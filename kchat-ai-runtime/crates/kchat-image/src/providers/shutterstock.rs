//! Shutterstock image search adapter.
//!
//! API docs: https://api-reference.shutterstock.com/
//! Endpoint: GET https://api.shutterstock.com/v2/images/search
//! Auth: `Authorization: Bearer <token>` header.
//! License: Commercial (and Editorial).
//! Rate limit: ~100 req/hour (free tier), higher on paid tiers.

use crate::error::ImageError;
use crate::provider::{retry_with_backoff, ImageSearchProvider};
use crate::types::{
    ApiKey, Attribution, ImageOrientation, ImageResult, ImageSearchRequest, ImageSearchResponse,
    ImageType, License, RateLimit,
};
use async_trait::async_trait;
use serde::Deserialize;

const SHUTTERSTOCK_ENDPOINT: &str = "https://api.shutterstock.com/v2/images/search";

pub struct ShutterstockProvider {
    client: reqwest::Client,
}

impl ShutterstockProvider {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(15))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        }
    }
}

impl Default for ShutterstockProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize)]
struct ShutterstockSearchResponse {
    #[serde(default)]
    total_count: u64,
    #[serde(default)]
    page: u32,
    #[serde(default)]
    per_page: u32,
    data: Vec<ShutterstockImage>,
}

#[derive(Debug, Deserialize)]
struct ShutterstockImage {
    id: String,
    #[serde(default)]
    aspect_ratio: Option<f32>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    image_type: Option<String>,
    assets: ShutterstockAssets,
    contributor: ShutterstockContributor,
}

#[derive(Debug, Deserialize)]
struct ShutterstockAssets {
    preview: ShutterstockAsset,
    #[serde(default)]
    large_thumb: Option<ShutterstockAsset>,
}

#[derive(Debug, Deserialize)]
struct ShutterstockAsset {
    url: String,
    width: u32,
    height: u32,
}

#[derive(Debug, Deserialize)]
struct ShutterstockContributor {
    id: String,
    #[serde(default)]
    display_name: Option<String>,
}

#[async_trait]
impl ImageSearchProvider for ShutterstockProvider {
    fn id(&self) -> &'static str {
        "shutterstock"
    }

    fn name(&self) -> &'static str {
        "Shutterstock"
    }

    fn env_var_name(&self) -> &'static str {
        "SHUTTERSTOCK_API_TOKEN"
    }

    fn rate_limit(&self) -> RateLimit {
        RateLimit::shutterstock()
    }

    async fn search(
        &self,
        req: &ImageSearchRequest,
        key: &ApiKey,
    ) -> Result<ImageSearchResponse, ImageError> {
        if key.is_empty() {
            return Err(ImageError::KeyMissing(self.env_var_name()));
        }

        let per_page = req.per_page.min(100).max(1);
        let provider_id = self.id();

        retry_with_backoff(provider_id, || async move {
            let mut q = self
                .client
                .get(SHUTTERSTOCK_ENDPOINT)
                .header("Authorization", format!("Bearer {}", key.as_str()))
                .query(&[("query", req.query.as_str())])
                .query(&[("per_page", &per_page.to_string())])
                .query(&[("page", &req.page.to_string())])
                .query(&[(
                    "keyword_safe_search",
                    if req.safesearch { "true" } else { "false" },
                )]);

            let img_type = req.image_type.unwrap_or(ImageType::Photo);
            q = q.query(&[("image_type", img_type.shutterstock())]);

            if let Some(o) = req.orientation {
                q = q.query(&[("orientation", o.shutterstock())]);
            }
            if let Some(c) = &req.color {
                q = q.query(&[("color", c.as_str())]);
            }
            if let Some(l) = &req.locale {
                q = q.query(&[("language", l.as_str())]);
            }
            if let Some(s) = req.sort {
                q = q.query(&[("sort", s.shutterstock())]);
            }

            let resp = q
                .send()
                .await
                .map_err(|e| ImageError::Network(provider_id, e.to_string()))?;

            let status = resp.status();
            if !status.is_success() {
                let body = resp.text().await.unwrap_or_default();
                if status.as_u16() == 401 {
                    return Err(ImageError::AuthFailed(provider_id, body));
                }
                if status.as_u16() == 429 {
                    return Err(ImageError::RateLimited(provider_id, 60));
                }
                return Err(ImageError::HttpStatus(
                    provider_id,
                    status.as_u16(),
                    body.chars().take(500).collect(),
                ));
            }

            let parsed: ShutterstockSearchResponse = resp
                .json()
                .await
                .map_err(|e| ImageError::Parse(provider_id, e.to_string()))?;

            let results = parsed
                .data
                .into_iter()
                .map(|img| {
                    let preview = &img.assets.preview;
                    let orientation = ImageOrientation::from_dims(preview.width, preview.height);
                    let thumb = img
                        .assets
                        .large_thumb
                        .as_ref()
                        .map(|a| a.url.clone())
                        .unwrap_or_else(|| preview.url.clone());
                    let photographer = img
                        .contributor
                        .display_name
                        .clone()
                        .unwrap_or_else(|| img.contributor.id.clone());
                    let license = if img
                        .image_type
                        .as_deref()
                        .map(|t| t.eq_ignore_ascii_case("editorial"))
                        .unwrap_or(false)
                    {
                        License::Editorial
                    } else {
                        License::Commercial
                    };
                    ImageResult {
                        id: img.id.clone(),
                        provider: provider_id.into(),
                        url: preview.url.clone(),
                        thumb_url: thumb,
                        width: preview.width,
                        height: preview.height,
                        orientation,
                        alt_text: img.description.unwrap_or_default(),
                        attribution: Attribution {
                            photographer,
                            photographer_url: format!(
                                "https://www.shutterstock.com/g/{}",
                                img.contributor.id
                            ),
                            source_url: format!(
                                "https://www.shutterstock.com/image/{}",
                                img.id
                            ),
                        },
                        license,
                        color: None,
                    }
                })
                .collect();

            Ok(ImageSearchResponse {
                results,
                total: parsed.total_count,
                page: if parsed.page == 0 { req.page } else { parsed.page },
                per_page: if parsed.per_page == 0 {
                    per_page
                } else {
                    parsed.per_page
                },
                provider: provider_id.into(),
            })
        })
        .await
    }
}

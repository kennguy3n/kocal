//! Pixabay image search adapter.
//!
//! API docs: https://pixabay.com/api/docs/
//! Endpoint: GET https://pixabay.com/api/
//! Auth: `key` query parameter.
//! License: Free, attribution required (Pixabay License).
//! Rate limit: ~100 req/hour, 5000 req/month (sustained).

use crate::error::ImageError;
use crate::provider::{retry_with_backoff, ImageSearchProvider};
use crate::types::{
    ApiKey, Attribution, ImageOrientation, ImageResult, ImageSearchRequest, ImageSearchResponse,
    ImageType, License, RateLimit,
};
use async_trait::async_trait;
use serde::Deserialize;

const PIXABAY_ENDPOINT: &str = "https://pixabay.com/api/";

pub struct PixabayProvider {
    client: reqwest::Client,
}

impl PixabayProvider {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(15))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        }
    }
}

impl Default for PixabayProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize)]
struct PixabayResponse {
    #[serde(default)]
    #[allow(dead_code)]
    total: u64,
    #[serde(rename = "totalHits", default)]
    total_hits: u64,
    hits: Vec<PixabayHit>,
}

#[derive(Debug, Deserialize)]
struct PixabayHit {
    id: i64,
    #[serde(rename = "webformatURL")]
    webformat_url: String,
    #[serde(rename = "previewURL")]
    preview_url: String,
    #[serde(rename = "imageWidth")]
    image_width: u32,
    #[serde(rename = "imageHeight")]
    image_height: u32,
    #[serde(rename = "imageSize")]
    #[allow(dead_code)]
    image_size: u64,
    tags: String,
    user: String,
    #[serde(rename = "user_id")]
    user_id: i64,
    #[serde(rename = "pageURL")]
    page_url: String,
    #[serde(default)]
    #[serde(rename = "imageType")]
    image_type: Option<String>,
}

#[async_trait]
impl ImageSearchProvider for PixabayProvider {
    fn id(&self) -> &'static str {
        "pixabay"
    }

    fn name(&self) -> &'static str {
        "Pixabay"
    }

    fn env_var_name(&self) -> &'static str {
        "PIXABAY_API_KEY"
    }

    fn rate_limit(&self) -> RateLimit {
        RateLimit::pixabay()
    }

    async fn search(
        &self,
        req: &ImageSearchRequest,
        key: &ApiKey,
    ) -> Result<ImageSearchResponse, ImageError> {
        if key.is_empty() {
            return Err(ImageError::KeyMissing(self.env_var_name()));
        }

        let per_page = req.per_page.min(200).max(3);
        let provider_id = self.id();

        retry_with_backoff(provider_id, || async move {
            let mut q = self
                .client
                .get(PIXABAY_ENDPOINT)
                .query(&[("key", key.as_str())])
                .query(&[("q", req.query.as_str())])
                .query(&[("per_page", &per_page.to_string())])
                .query(&[("page", &req.page.to_string())])
                .query(&[("safesearch", if req.safesearch { "true" } else { "false" })]);

            let img_type = req.image_type.unwrap_or(ImageType::All);
            q = q.query(&[("image_type", img_type.pixabay())]);

            if let Some(o) = req.orientation {
                let p = o.pixabay();
                if p != "all" {
                    q = q.query(&[("orientation", p)]);
                }
            }
            if let Some(c) = &req.color {
                q = q.query(&[("colors", c.as_str())]);
            }
            if let Some(mw) = req.min_width {
                q = q.query(&[("min_width", &mw.to_string())]);
            }
            if let Some(mh) = req.min_height {
                q = q.query(&[("min_height", &mh.to_string())]);
            }
            if let Some(l) = &req.locale {
                q = q.query(&[("lang", l.as_str())]);
            }
            if let Some(s) = req.sort {
                q = q.query(&[("order", s.pixabay())]);
            }

            let resp = q
                .send()
                .await
                .map_err(|e| ImageError::Network(provider_id, e.to_string()))?;

            let status = resp.status();
            if !status.is_success() {
                let body = resp.text().await.unwrap_or_default();
                if status.as_u16() == 401 || status.as_u16() == 403 {
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

            let parsed: PixabayResponse = resp
                .json()
                .await
                .map_err(|e| ImageError::Parse(provider_id, e.to_string()))?;

            let results = parsed
                .hits
                .into_iter()
                .map(|h| {
                    let orientation = ImageOrientation::from_dims(h.image_width, h.image_height);
                    let photographer = h.user;
                    let photographer_url = format!(
                        "https://pixabay.com/users/{}-{}",
                        photographer, h.user_id
                    );
                    ImageResult {
                        id: h.id.to_string(),
                        provider: provider_id.into(),
                        url: h.webformat_url,
                        thumb_url: h.preview_url,
                        width: h.image_width,
                        height: h.image_height,
                        orientation,
                        alt_text: h.tags,
                        attribution: Attribution {
                            photographer,
                            photographer_url,
                            source_url: h.page_url,
                        },
                        license: License::FreeWithAttribution,
                        color: None,
                    }
                })
                .collect();

            Ok(ImageSearchResponse {
                results,
                total: parsed.total_hits,
                page: req.page,
                per_page,
                provider: provider_id.into(),
            })
        })
        .await
    }
}

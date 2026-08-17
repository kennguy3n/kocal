//! Pexels image search adapter.
//!
//! API docs: https://www.pexels.com/api/documentation/
//! Endpoint: GET https://api.pexels.com/v1/search
//! Auth: Authorization header with API key.
//! License: Free, no attribution required (though appreciated).
//! Rate limit: 200 req/hour, 20000 req/month.

use crate::error::ImageError;
use crate::provider::{retry_with_backoff, ImageSearchProvider};
use crate::types::{
    ApiKey, Attribution, ImageOrientation, ImageResult, ImageSearchRequest, ImageSearchResponse,
    License, RateLimit,
};
use async_trait::async_trait;
use serde::Deserialize;

const PEXELS_ENDPOINT: &str = "https://api.pexels.com/v1/search";

/// Pexels provider.
pub struct PexelsProvider {
    client: reqwest::Client,
}

impl PexelsProvider {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(15))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        }
    }
}

impl Default for PexelsProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize)]
struct PexelsResponse {
    total_results: u64,
    page: u32,
    per_page: u32,
    photos: Vec<PexelsPhoto>,
}

#[derive(Debug, Deserialize)]
struct PexelsPhoto {
    id: i64,
    width: u32,
    height: u32,
    alt: Option<String>,
    #[serde(default)]
    avg_color: Option<String>,
    photographer: String,
    photographer_url: String,
    url: String,
    src: PexelsSrc,
}

#[derive(Debug, Deserialize)]
struct PexelsSrc {
    original: String,
    large: String,
    medium: String,
    small: String,
    #[serde(default)]
    portrait: Option<String>,
    #[serde(default)]
    landscape: Option<String>,
}

#[async_trait]
impl ImageSearchProvider for PexelsProvider {
    fn id(&self) -> &'static str {
        "pexels"
    }

    fn name(&self) -> &'static str {
        "Pexels"
    }

    fn env_var_name(&self) -> &'static str {
        "PEXELS_API_KEY"
    }

    fn rate_limit(&self) -> RateLimit {
        RateLimit::pexels()
    }

    async fn search(
        &self,
        req: &ImageSearchRequest,
        key: &ApiKey,
    ) -> Result<ImageSearchResponse, ImageError> {
        if key.is_empty() {
            return Err(ImageError::KeyMissing(self.env_var_name()));
        }

        let per_page = req.per_page.min(80).max(1);
        let provider_id = self.id();

        retry_with_backoff(provider_id, || async move {
            let mut q = self
                .client
                .get(PEXELS_ENDPOINT)
                .header("Authorization", key.as_str())
                .query(&[("query", req.query.as_str())])
                .query(&[("per_page", &per_page.to_string())])
                .query(&[("page", &req.page.to_string())]);

            if let Some(o) = req.orientation {
                q = q.query(&[("orientation", o.pexels())]);
            }
            if let Some(c) = &req.color {
                q = q.query(&[("color", c.as_str())]);
            }
            if let Some(l) = &req.locale {
                q = q.query(&[("locale", l.as_str())]);
            }

            let resp = q.send().await.map_err(|e| {
                ImageError::Network(provider_id, e.to_string())
            })?;

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

            let parsed: PexelsResponse = resp
                .json()
                .await
                .map_err(|e| ImageError::Parse(provider_id, e.to_string()))?;

            let results = parsed
                .photos
                .into_iter()
                .map(|p| {
                    let orientation = ImageOrientation::from_dims(p.width, p.height);
                    ImageResult {
                        id: p.id.to_string(),
                        provider: provider_id.into(),
                        url: p.src.original,
                        thumb_url: p.src.large,
                        width: p.width,
                        height: p.height,
                        orientation,
                        alt_text: p.alt.unwrap_or_default(),
                        attribution: Attribution {
                            photographer: p.photographer,
                            photographer_url: p.photographer_url,
                            source_url: p.url,
                        },
                        license: License::FreeNoAttribution,
                        color: p.avg_color,
                    }
                })
                .collect();

            Ok(ImageSearchResponse {
                results,
                total: parsed.total_results,
                page: parsed.page,
                per_page: parsed.per_page,
                provider: provider_id.into(),
            })
        })
        .await
    }
}

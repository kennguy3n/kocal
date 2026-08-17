//! Unsplash image search adapter.
//!
//! API docs: https://unsplash.com/documentation
//! Endpoint: GET https://api.unsplash.com/search/photos
//! Auth: `Authorization: Client-ID <ACCESS_KEY>` header.
//! License: Free, attribution required (Unsplash License).
//! Rate limit: 50 req/hour (demo), 5000 req/hour (approved production).

use crate::error::ImageError;
use crate::provider::{retry_with_backoff, ImageSearchProvider};
use crate::types::{
    ApiKey, Attribution, ImageOrientation, ImageResult, ImageSearchRequest, ImageSearchResponse,
    License, RateLimit,
};
use async_trait::async_trait;
use serde::Deserialize;

const UNSPLASH_ENDPOINT: &str = "https://api.unsplash.com/search/photos";

pub struct UnsplashProvider {
    client: reqwest::Client,
}

impl UnsplashProvider {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(15))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        }
    }
}

impl Default for UnsplashProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize)]
struct UnsplashSearchResponse {
    total: u64,
    total_pages: u64,
    results: Vec<UnsplashPhoto>,
}

#[derive(Debug, Deserialize)]
struct UnsplashPhoto {
    id: String,
    width: u32,
    height: u32,
    #[serde(default)]
    alt_description: Option<String>,
    #[serde(default)]
    description: Option<String>,
    urls: UnsplashUrls,
    user: UnsplashUser,
    links: UnsplashLinks,
    #[serde(default)]
    color: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UnsplashUrls {
    regular: String,
    small: String,
    thumb: String,
    #[serde(default)]
    raw: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UnsplashUser {
    name: String,
    #[serde(default)]
    username: String,
    links: UnsplashUserLinks,
}

#[derive(Debug, Deserialize)]
struct UnsplashUserLinks {
    html: String,
}

#[derive(Debug, Deserialize)]
struct UnsplashLinks {
    html: String,
}

#[async_trait]
impl ImageSearchProvider for UnsplashProvider {
    fn id(&self) -> &'static str {
        "unsplash"
    }

    fn name(&self) -> &'static str {
        "Unsplash"
    }

    fn env_var_name(&self) -> &'static str {
        "UNSPLASH_ACCESS_KEY"
    }

    fn rate_limit(&self) -> RateLimit {
        RateLimit::unsplash()
    }

    async fn search(
        &self,
        req: &ImageSearchRequest,
        key: &ApiKey,
    ) -> Result<ImageSearchResponse, ImageError> {
        if key.is_empty() {
            return Err(ImageError::KeyMissing(self.env_var_name()));
        }

        let per_page = req.per_page.min(30).max(1);
        let provider_id = self.id();

        retry_with_backoff(provider_id, || async move {
            let mut q = self
                .client
                .get(UNSPLASH_ENDPOINT)
                .header("Authorization", format!("Client-ID {}", key.as_str()))
                .header("Accept-Version", "v1")
                .query(&[("query", req.query.as_str())])
                .query(&[("per_page", &per_page.to_string())])
                .query(&[("page", &req.page.to_string())])
                // Default to high content filter for safety.
                .query(&[("content_filter", if req.safesearch { "high" } else { "low" })]);

            if let Some(o) = req.orientation {
                q = q.query(&[("orientation", o.unsplash())]);
            }
            if let Some(c) = &req.color {
                q = q.query(&[("color", c.as_str())]);
            }
            if let Some(s) = req.sort {
                q = q.query(&[("order_by", s.unsplash())]);
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

            let parsed: UnsplashSearchResponse = resp
                .json()
                .await
                .map_err(|e| ImageError::Parse(provider_id, e.to_string()))?;

            let results = parsed
                .results
                .into_iter()
                .map(|p| {
                    let orientation = ImageOrientation::from_dims(p.width, p.height);
                    let alt = p
                        .alt_description
                        .or(p.description)
                        .unwrap_or_default();
                    ImageResult {
                        id: p.id,
                        provider: provider_id.into(),
                        url: p.urls.regular,
                        thumb_url: p.urls.thumb,
                        width: p.width,
                        height: p.height,
                        orientation,
                        alt_text: alt,
                        attribution: Attribution {
                            photographer: p.user.name,
                            photographer_url: p.user.links.html,
                            source_url: p.links.html,
                        },
                        license: License::FreeWithAttribution,
                        color: p.color,
                    }
                })
                .collect();

            Ok(ImageSearchResponse {
                results,
                total: parsed.total,
                page: req.page,
                per_page,
                provider: provider_id.into(),
            })
        })
        .await
    }
}

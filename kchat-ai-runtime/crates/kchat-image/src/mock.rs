//! Mock image search provider for offline testing.
//!
//! Returns deterministic `ImageResult`s from a fixture map keyed by query.
//! Used by crate unit tests and by the eval harness when no API keys are set.

use crate::error::ImageError;
use crate::provider::ImageSearchProvider;
use crate::types::{
    ApiKey, Attribution, ImageOrientation, ImageResult, ImageSearchRequest, ImageSearchResponse,
    License, RateLimit,
};
use async_trait::async_trait;
use std::collections::HashMap;

/// A mock provider that returns canned results.
pub struct MockProvider {
    fixtures: HashMap<String, Vec<ImageResult>>,
}

impl MockProvider {
    pub fn new() -> Self {
        Self {
            fixtures: HashMap::new(),
        }
    }

    /// Register a fixture for a query.
    pub fn with_fixture(mut self, query: &str, results: Vec<ImageResult>) -> Self {
        self.fixtures.insert(query.to_lowercase(), results);
        self
    }

    /// Insert a fixture.
    pub fn add_fixture(&mut self, query: &str, results: Vec<ImageResult>) {
        self.fixtures.insert(query.to_lowercase(), results);
    }

    /// Build a default mock provider with a few standard fixtures.
    pub fn with_defaults() -> Self {
        let mut p = Self::new();
        p.add_fixture(
            "beach",
            vec![make_result("1", "beach at sunset", 1920, 1080, ImageOrientation::Landscape)],
        );
        p.add_fixture(
            "mountain",
            vec![
                make_result("2", "snowy mountain peak", 1080, 1920, ImageOrientation::Portrait),
                make_result("3", "mountain range panorama", 1920, 1080, ImageOrientation::Landscape),
            ],
        );
        p.add_fixture(
            "city",
            vec![
                make_result("4", "city skyline at night", 1920, 1080, ImageOrientation::Landscape),
                make_result("5", "city street portrait", 1080, 1920, ImageOrientation::Portrait),
                make_result("6", "city square", 800, 800, ImageOrientation::Square),
            ],
        );
        p
    }
}

impl Default for MockProvider {
    fn default() -> Self {
        Self::new()
    }
}

fn make_result(
    id: &str,
    alt: &str,
    w: u32,
    h: u32,
    o: ImageOrientation,
) -> ImageResult {
    ImageResult {
        id: id.into(),
        provider: "mock".into(),
        url: format!("https://mock.example.com/{}_{}x{}.jpg", id, w, h),
        thumb_url: format!("https://mock.example.com/thumb_{}.jpg", id),
        width: w,
        height: h,
        orientation: o,
        alt_text: alt.into(),
        attribution: Attribution {
            photographer: "Mock Photographer".into(),
            photographer_url: "https://mock.example.com/photographer".into(),
            source_url: format!("https://mock.example.com/photo/{}", id),
        },
        license: License::FreeNoAttribution,
        color: None,
    }
}

#[async_trait]
impl ImageSearchProvider for MockProvider {
    fn id(&self) -> &'static str {
        "mock"
    }

    fn name(&self) -> &'static str {
        "Mock"
    }

    fn env_var_name(&self) -> &'static str {
        "MOCK_IMAGE_API_KEY"
    }

    fn rate_limit(&self) -> RateLimit {
        RateLimit {
            per_hour: u32::MAX,
            per_month: u32::MAX,
        }
    }

    async fn search(
        &self,
        req: &ImageSearchRequest,
        _key: &ApiKey,
    ) -> Result<ImageSearchResponse, ImageError> {
        let key = req.query.to_lowercase();
        let results = self
            .fixtures
            .get(&key)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter(|r| r.matches_orientation(req.orientation))
            .collect();
        Ok(ImageSearchResponse {
            results,
            total: self.fixtures.get(&key).map(|v| v.len() as u64).unwrap_or(0),
            page: req.page,
            per_page: req.per_page,
            provider: self.id().into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_search_existing_query() {
        let p = MockProvider::with_defaults();
        let req = ImageSearchRequest::new("beach");
        let resp = p.search(&req, &ApiKey::default()).await.unwrap();
        assert_eq!(resp.results.len(), 1);
        assert_eq!(resp.results[0].alt_text, "beach at sunset");
    }

    #[tokio::test]
    async fn test_mock_search_missing_query() {
        let p = MockProvider::with_defaults();
        let req = ImageSearchRequest::new("nonexistent");
        let resp = p.search(&req, &ApiKey::default()).await.unwrap();
        assert_eq!(resp.results.len(), 0);
    }

    #[tokio::test]
    async fn test_mock_search_orientation_filter() {
        let p = MockProvider::with_defaults();
        let req = ImageSearchRequest::new("mountain")
            .with_orientation(ImageOrientation::Portrait);
        let resp = p.search(&req, &ApiKey::default()).await.unwrap();
        assert_eq!(resp.results.len(), 1);
        assert_eq!(resp.results[0].orientation, ImageOrientation::Portrait);
    }
}

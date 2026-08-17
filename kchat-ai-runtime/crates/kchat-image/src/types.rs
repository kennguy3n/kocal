//! Core types for image search across providers.
//!
//! All provider responses are normalized into `ImageResult` so the
//! slides skill surface and eval harness can treat results uniformly.

use serde::{Deserialize, Serialize};

/// Image orientation. Provider-specific values are normalized:
/// - Pexels: landscape / portrait / square
/// - Pixabay: horizontal / vertical
/// - Unsplash: landscape / portrait / squarish
/// - Shutterstock: horizontal / vertical
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageOrientation {
    Landscape,
    Portrait,
    Square,
}

impl ImageOrientation {
    /// Pexels API orientation string.
    pub fn pexels(self) -> &'static str {
        match self {
            ImageOrientation::Landscape => "landscape",
            ImageOrientation::Portrait => "portrait",
            ImageOrientation::Square => "square",
        }
    }

    /// Pixabay API orientation string.
    pub fn pixabay(self) -> &'static str {
        match self {
            ImageOrientation::Landscape => "horizontal",
            ImageOrientation::Portrait => "vertical",
            ImageOrientation::Square => "all",
        }
    }

    /// Unsplash API orientation string.
    pub fn unsplash(self) -> &'static str {
        match self {
            ImageOrientation::Landscape => "landscape",
            ImageOrientation::Portrait => "portrait",
            ImageOrientation::Square => "squarish",
        }
    }

    /// Shutterstock API orientation string.
    pub fn shutterstock(self) -> &'static str {
        match self {
            ImageOrientation::Landscape => "horizontal",
            ImageOrientation::Portrait => "vertical",
            // Shutterstock has no square; fall back to horizontal.
            ImageOrientation::Square => "horizontal",
        }
    }

    /// Infer orientation from width/height.
    pub fn from_dims(width: u32, height: u32) -> Self {
        if width == 0 || height == 0 {
            return ImageOrientation::Landscape;
        }
        let ratio = width as f32 / height as f32;
        if (0.9..=1.1).contains(&ratio) {
            ImageOrientation::Square
        } else if ratio > 1.0 {
            ImageOrientation::Landscape
        } else {
            ImageOrientation::Portrait
        }
    }
}

/// Image media type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageType {
    Photo,
    Illustration,
    Vector,
    All,
}

impl ImageType {
    pub fn pixabay(self) -> &'static str {
        match self {
            ImageType::Photo => "photo",
            ImageType::Illustration => "illustration",
            ImageType::Vector => "vector",
            ImageType::All => "all",
        }
    }

    pub fn shutterstock(self) -> &'static str {
        match self {
            ImageType::Photo => "photo",
            ImageType::Illustration => "illustration",
            ImageType::Vector => "vector",
            ImageType::All => "photo",
        }
    }
}

/// Sort order for search results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageSort {
    Relevant,
    Latest,
    Popular,
}

impl ImageSort {
    pub fn pixabay(self) -> &'static str {
        match self {
            ImageSort::Popular => "popular",
            ImageSort::Latest => "latest",
            ImageSort::Relevant => "popular",
        }
    }

    pub fn unsplash(self) -> &'static str {
        match self {
            ImageSort::Relevant => "relevant",
            ImageSort::Latest => "latest",
            ImageSort::Popular => "relevant",
        }
    }

    pub fn shutterstock(self) -> &'static str {
        match self {
            ImageSort::Popular => "popular",
            ImageSort::Latest => "newest",
            ImageSort::Relevant => "relevance",
        }
    }
}

/// A search request, provider-agnostic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSearchRequest {
    /// Search query (required).
    pub query: String,
    /// Desired orientation (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub orientation: Option<ImageOrientation>,
    /// Image type filter (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_type: Option<ImageType>,
    /// Color filter (hex string like "4F21EA" or named like "blue", "black_and_white").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    /// Results per page (1-200, clamped per provider).
    pub per_page: u32,
    /// Page number (1-indexed).
    pub page: u32,
    /// Enable safe search (default true).
    pub safesearch: bool,
    /// Minimum width in pixels.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_width: Option<u32>,
    /// Minimum height in pixels.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_height: Option<u32>,
    /// Locale / language code (e.g. "en", "vi", "ja").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub locale: Option<String>,
    /// Sort order.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sort: Option<ImageSort>,
}

impl Default for ImageSearchRequest {
    fn default() -> Self {
        Self {
            query: String::new(),
            orientation: None,
            image_type: None,
            color: None,
            per_page: 15,
            page: 1,
            safesearch: true,
            min_width: None,
            min_height: None,
            locale: None,
            sort: None,
        }
    }
}

impl ImageSearchRequest {
    /// Create a simple query-only request.
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            ..Default::default()
        }
    }

    /// Builder: set orientation.
    pub fn with_orientation(mut self, o: ImageOrientation) -> Self {
        self.orientation = Some(o);
        self
    }

    /// Builder: set image type.
    pub fn with_image_type(mut self, t: ImageType) -> Self {
        self.image_type = Some(t);
        self
    }

    /// Builder: set per_page.
    pub fn with_per_page(mut self, n: u32) -> Self {
        self.per_page = n;
        self
    }

    /// Builder: set safesearch.
    pub fn with_safesearch(mut self, s: bool) -> Self {
        self.safesearch = s;
        self
    }

    /// Compute a stable hash of the request for cache keys.
    pub fn cache_key(&self) -> String {
        let canonical = serde_json::to_string(self).unwrap_or_default();
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(canonical.as_bytes());
        hex::encode(hasher.finalize())
    }
}

/// Attribution info for an image.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Attribution {
    /// Photographer / contributor name.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub photographer: String,
    /// URL to the photographer's profile.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub photographer_url: String,
    /// URL to the source page on the provider.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub source_url: String,
}

/// License type for an image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum License {
    /// Free, no attribution required (Pexels).
    FreeNoAttribution,
    /// Free, attribution required (Pixabay, Unsplash).
    FreeWithAttribution,
    /// Commercial license (Shutterstock).
    Commercial,
    /// Editorial use only.
    Editorial,
}

impl License {
    pub fn requires_attribution(self) -> bool {
        matches!(self, License::FreeWithAttribution)
    }
}

/// A single normalized image result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageResult {
    /// Provider-assigned ID.
    pub id: String,
    /// Provider name ("pexels", "pixabay", "unsplash", "shutterstock").
    pub provider: String,
    /// Full-resolution image URL.
    pub url: String,
    /// Thumbnail URL.
    pub thumb_url: String,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Inferred orientation.
    pub orientation: ImageOrientation,
    /// Alt text / description.
    pub alt_text: String,
    /// Attribution metadata.
    pub attribution: Attribution,
    /// License type.
    pub license: License,
    /// Dominant color (hex or named), if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
}

impl ImageResult {
    /// Whether this result satisfies the orientation filter.
    pub fn matches_orientation(&self, requested: Option<ImageOrientation>) -> bool {
        match requested {
            None => true,
            Some(req) => {
                if req == ImageOrientation::Square {
                    // Square is approximate — accept near-square.
                    self.orientation == ImageOrientation::Square
                } else {
                    self.orientation == req
                }
            }
        }
    }
}

/// A page of search results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSearchResponse {
    /// Results on this page.
    pub results: Vec<ImageResult>,
    /// Total results available across all pages (provider-reported).
    pub total: u64,
    /// Current page number.
    pub page: u32,
    /// Per-page count.
    pub per_page: u32,
    /// Provider that produced this response.
    pub provider: String,
}

impl ImageSearchResponse {
    pub fn empty(provider: &str) -> Self {
        Self {
            results: Vec::new(),
            total: 0,
            page: 1,
            per_page: 0,
            provider: provider.into(),
        }
    }
}

/// API key wrapper that zeroizes the key on drop to avoid leaving
/// secrets in memory after the provider entry is deallocated.
#[derive(Debug, Clone)]
pub struct ApiKey(zeroize::Zeroizing<String>);

impl ApiKey {
    pub fn new(key: impl Into<String>) -> Self {
        Self(zeroize::Zeroizing::new(key.into()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Default for ApiKey {
    fn default() -> Self {
        Self(zeroize::Zeroizing::new(String::new()))
    }
}

/// Per-provider rate limit descriptor.
#[derive(Debug, Clone, Copy)]
pub struct RateLimit {
    pub per_hour: u32,
    pub per_month: u32,
}

impl RateLimit {
    pub fn pexels() -> Self {
        Self {
            per_hour: 200,
            per_month: 20000,
        }
    }
    pub fn pixabay() -> Self {
        Self {
            per_hour: 100,
            per_month: 5000,
        }
    }
    pub fn unsplash() -> Self {
        Self {
            per_hour: 50,
            per_month: 5000,
        }
    }
    pub fn shutterstock() -> Self {
        Self {
            per_hour: 100,
            per_month: 1000,
        }
    }
}

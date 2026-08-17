//! kchat-image: Unified image search across Pexels, Pixabay, Unsplash, Shutterstock.
//!
//! Provides a single `ImageSearchProvider` trait implemented by 4 stock photo
//! providers, plus a `MockProvider` for offline testing. The
//! `ImageSearchRegistry` merges results across providers with fallback,
//! deduplication, safety filtering, and an in-memory cache.
//!
//! ## Providers
//!
//! | Provider | Endpoint | Auth | License | Rate limit |
//! |----------|----------|------|---------|------------|
//! | Pexels | `api.pexels.com/v1/search` | `Authorization` header | Free, no attribution | 200/hr |
//! | Pixabay | `pixabay.com/api/` | `key` query param | Free, attribution required | 100/hr |
//! | Unsplash | `api.unsplash.com/search/photos` | `Client-ID` header | Free, attribution required | 50/hr |
//! | Shutterstock | `api.shutterstock.com/v2/images/search` | `Bearer` header | Commercial | 100/hr |
//!
//! ## Usage
//!
//! ```no_run
//! # use kchat_image::*;
//! # async fn run() -> Result<(), ImageError> {
//! let registry = ImageSearchRegistry::from_env();
//! let req = ImageSearchRequest::new("mountain landscape")
//!     .with_orientation(ImageOrientation::Landscape)
//!     .with_per_page(10);
//! let resp = registry.search(&req).await?;
//! for r in &resp.results {
//!     println!("{}: {} ({}x{})", r.provider, r.alt_text, r.width, r.height);
//! }
//! # Ok(())
//! # }
//! ```

pub mod cache;
pub mod error;
pub mod mock;
pub mod provider;
pub mod providers;
pub mod registry;
pub mod safety;
pub mod types;

pub use cache::ImageCache;
pub use error::ImageError;
pub use mock::MockProvider;
pub use provider::ImageSearchProvider;
pub use providers::{PexelsProvider, PixabayProvider, ShutterstockProvider, UnsplashProvider};
pub use registry::ImageSearchRegistry;
pub use types::{
    ApiKey, Attribution, ImageOrientation, ImageResult, ImageSearchRequest, ImageSearchResponse,
    ImageSort, ImageType, License, RateLimit,
};

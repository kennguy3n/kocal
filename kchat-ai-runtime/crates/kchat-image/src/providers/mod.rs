//! Provider adapters — concrete implementations of `ImageSearchProvider`.

pub mod pexels;
pub mod pixabay;
pub mod shutterstock;
pub mod unsplash;

pub use pexels::PexelsProvider;
pub use pixabay::PixabayProvider;
pub use shutterstock::ShutterstockProvider;
pub use unsplash::UnsplashProvider;

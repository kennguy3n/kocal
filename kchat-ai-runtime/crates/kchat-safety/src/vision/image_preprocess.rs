//! Image bytes → MobileCLIP-preprocessed CHW f32 tensor.
//!
//! Ported from slm-guardrail's `encoder/vision/image_preprocess.rs`.
//! Pure-Rust port of MobileCLIP-S2's official image preprocessing:
//!
//! 1. Decode raw bytes (PNG/JPEG/WebP) to RGB
//! 2. Aspect-preserving resize to 256 (bilinear), then center-crop to square
//! 3. u8 → f32 / 255.0 → per-channel normalize (no-op for MobileCLIP-S2)
//! 4. Reshape HWC → CHW for ONNX session input

use image::imageops::FilterType;
use image::{GenericImageView, ImageReader, Limits};
use std::io::Cursor;

use super::{MOBILECLIP_IMAGE_SIZE, MOBILECLIP_PIXEL_MEAN, MOBILECLIP_PIXEL_STD};

/// Hard cap on each input dimension for decoded images.
const MAX_INPUT_DIMENSION: u32 = 16_384;

/// Hard cap on decoder's transient allocation (256 MiB).
const MAX_DECODE_ALLOC_BYTES: u64 = 256 * 1024 * 1024;

/// Hard cap on the long edge after aspect-preserving resize.
const MAX_RESIZED_LONG_EDGE: u32 = 32_768;

/// Errors raised by [`preprocess_image`].
#[derive(Debug)]
pub enum VisionImagePreprocessError {
    DecodeFailed { reason: String },
    EmptyImage { width: u32, height: u32 },
    ImageTooLarge { width: u32, height: u32, reason: &'static str },
}

impl std::fmt::Display for VisionImagePreprocessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DecodeFailed { reason } => {
                write!(f, "image preprocess: decode failed: {reason}")
            }
            Self::EmptyImage { width, height } => write!(
                f,
                "image preprocess: decoded image has empty dimensions {width}x{height}",
            ),
            Self::ImageTooLarge { width, height, reason } => write!(
                f,
                "image preprocess: decoded image {width}x{height} exceeds bound ({reason})",
            ),
        }
    }
}

impl std::error::Error for VisionImagePreprocessError {}

/// Banker's rounding (round half to even) matching Python 3's `round()`.
fn round_half_even_u32(x: f64) -> u32 {
    let floor = x.floor();
    let fract = x - floor;
    let on_half = (fract - 0.5).abs() < f64::EPSILON * 4.0;
    let rounded = if on_half {
        let floor_i = floor as i64;
        if floor_i % 2 == 0 { floor } else { floor + 1.0 }
    } else if fract < 0.5 {
        floor
    } else {
        floor + 1.0
    };
    rounded as u32
}

/// CLIP image preprocessing.
///
/// Returns a flat CHW `Vec<f32>` of length `3 * 256 * 256` (196608).
pub fn preprocess_image(bytes: &[u8]) -> Result<Vec<f32>, VisionImagePreprocessError> {
    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| VisionImagePreprocessError::DecodeFailed {
            reason: format!("format guess: {e}"),
        })?;
    let mut limits = Limits::default();
    limits.max_image_width = Some(MAX_INPUT_DIMENSION);
    limits.max_image_height = Some(MAX_INPUT_DIMENSION);
    limits.max_alloc = Some(MAX_DECODE_ALLOC_BYTES);
    reader.limits(limits);
    let dynamic_image = reader
        .decode()
        .map_err(|e| VisionImagePreprocessError::DecodeFailed {
            reason: format!("decode: {e}"),
        })?;

    let (orig_w, orig_h) = dynamic_image.dimensions();
    if orig_w == 0 || orig_h == 0 {
        return Err(VisionImagePreprocessError::EmptyImage {
            width: orig_w,
            height: orig_h,
        });
    }
    if orig_w > MAX_INPUT_DIMENSION || orig_h > MAX_INPUT_DIMENSION {
        return Err(VisionImagePreprocessError::ImageTooLarge {
            width: orig_w,
            height: orig_h,
            reason: "input dimension exceeds MAX_INPUT_DIMENSION",
        });
    }

    let target = MOBILECLIP_IMAGE_SIZE as u32;
    let scale = f64::from(target) / f64::from(orig_w.min(orig_h));
    let resized_w = round_half_even_u32(f64::from(orig_w) * scale).max(target);
    let resized_h = round_half_even_u32(f64::from(orig_h) * scale).max(target);

    if resized_w > MAX_RESIZED_LONG_EDGE || resized_h > MAX_RESIZED_LONG_EDGE {
        return Err(VisionImagePreprocessError::ImageTooLarge {
            width: orig_w,
            height: orig_h,
            reason: "aspect ratio would exceed MAX_RESIZED_LONG_EDGE after resize",
        });
    }

    let rgb_source = dynamic_image.to_rgb8();
    let resized = if resized_w == orig_w && resized_h == orig_h {
        rgb_source
    } else {
        image::imageops::resize(&rgb_source, resized_w, resized_h, FilterType::Triangle)
    };

    let crop_x = (resized_w - target) / 2;
    let crop_y = (resized_h - target) / 2;
    let mut chw = vec![0.0_f32; 3 * (target as usize) * (target as usize)];
    let stride: usize = (target as usize) * (target as usize);

    for y in 0..target {
        for x in 0..target {
            let pixel = resized.get_pixel(crop_x + x, crop_y + y);
            let [r, g, b] = pixel.0;
            let plane_offset = (y as usize) * (target as usize) + (x as usize);
            let r_f = (f32::from(r) / 255.0 - MOBILECLIP_PIXEL_MEAN[0]) / MOBILECLIP_PIXEL_STD[0];
            let g_f = (f32::from(g) / 255.0 - MOBILECLIP_PIXEL_MEAN[1]) / MOBILECLIP_PIXEL_STD[1];
            let b_f = (f32::from(b) / 255.0 - MOBILECLIP_PIXEL_MEAN[2]) / MOBILECLIP_PIXEL_STD[2];
            chw[plane_offset] = r_f;
            chw[stride + plane_offset] = g_f;
            chw[2 * stride + plane_offset] = b_f;
        }
    }
    Ok(chw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, ImageFormat, Rgb};
    use std::io::Cursor as IoCursor;

    fn synthetic_png(width: u32, height: u32) -> Vec<u8> {
        let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(width, height);
        for (x, y, pixel) in img.enumerate_pixels_mut() {
            *pixel = Rgb([(x % 256) as u8, (y % 256) as u8, 128]);
        }
        let mut buf: Vec<u8> = Vec::new();
        img.write_to(&mut IoCursor::new(&mut buf), ImageFormat::Png)
            .expect("encode png");
        buf
    }

    #[test]
    fn preprocess_png_returns_chw_tensor_of_expected_length() {
        let bytes = synthetic_png(512, 384);
        let tensor = preprocess_image(&bytes).expect("preprocess png");
        let expected_len = 3 * MOBILECLIP_IMAGE_SIZE * MOBILECLIP_IMAGE_SIZE;
        assert_eq!(tensor.len(), expected_len);
        for v in &tensor {
            assert!(v.is_finite(), "tensor element must be finite, got {v}");
        }
    }

    #[test]
    fn preprocess_garbage_bytes_returns_decode_failed() {
        let err = preprocess_image(b"this is not a valid image").expect_err("decode must fail");
        assert!(matches!(err, VisionImagePreprocessError::DecodeFailed { .. }));
    }

    #[test]
    fn preprocess_portrait_image_center_crops_to_square() {
        let bytes = synthetic_png(256, 512);
        let tensor = preprocess_image(&bytes).expect("preprocess portrait");
        assert_eq!(tensor.len(), 3 * MOBILECLIP_IMAGE_SIZE * MOBILECLIP_IMAGE_SIZE);
    }

    #[test]
    fn preprocess_landscape_image_center_crops_to_square() {
        let bytes = synthetic_png(512, 256);
        let tensor = preprocess_image(&bytes).expect("preprocess landscape");
        assert_eq!(tensor.len(), 3 * MOBILECLIP_IMAGE_SIZE * MOBILECLIP_IMAGE_SIZE);
    }

    #[test]
    fn preprocess_smaller_than_target_upscales() {
        let bytes = synthetic_png(100, 100);
        let tensor = preprocess_image(&bytes).expect("preprocess upscale");
        assert_eq!(tensor.len(), 3 * MOBILECLIP_IMAGE_SIZE * MOBILECLIP_IMAGE_SIZE);
    }

    #[test]
    fn round_half_even_u32_matches_python_round() {
        assert_eq!(round_half_even_u32(234.5), 234);
        assert_eq!(round_half_even_u32(235.5), 236);
        assert_eq!(round_half_even_u32(236.5), 236);
        assert_eq!(round_half_even_u32(237.5), 238);
        assert_eq!(round_half_even_u32(0.5), 0);
        assert_eq!(round_half_even_u32(1.5), 2);
    }
}

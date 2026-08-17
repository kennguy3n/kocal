//! Safety filter for image search results.
//!
//! Post-filters provider responses to enforce:
//! - Attribution presence when the license requires it
//! - alt_text / tag blocklist (adult, violence, etc.)
//! - URL scheme allowlist (https only)
//! - Empty-result rejection

use crate::types::{ImageResult, License};

/// Embedded safety blocklist for alt text. Kept small and standalone —
/// the heavier safety pipeline lives in `kchat-safety`.
const ALT_BLOCKLIST: &[&str] = &[
    "nude",
    "nudity",
    "nsfw",
    "porn",
    "explicit",
    "gore",
    "violence graphic",
    "weapon suicide",
    "self-harm",
    "terrorist",
    "isis",
];

/// Filter a list of results, dropping unsafe or incomplete entries.
/// Returns the kept results and the count of dropped results.
pub fn filter_results(results: Vec<ImageResult>) -> (Vec<ImageResult>, usize) {
    let mut kept = Vec::with_capacity(results.len());
    let mut dropped = 0;
    for r in results {
        if !is_safe(&r) {
            dropped += 1;
            continue;
        }
        kept.push(r);
    }
    (kept, dropped)
}

/// Check whether a single result passes the safety filter.
pub fn is_safe(r: &ImageResult) -> bool {
    // URL must be https.
    if !r.url.starts_with("https://") || !r.thumb_url.starts_with("https://") {
        return false;
    }
    // SSRF protection: reject URLs pointing to private/loopback/link-local IP ranges.
    if is_private_url(&r.url) || is_private_url(&r.thumb_url) {
        return false;
    }
    // Attribution required for Pixabay / Unsplash.
    if r.license.requires_attribution() {
        if r.attribution.photographer.is_empty() && r.attribution.source_url.is_empty() {
            return false;
        }
    }
    // Blocklist on alt text (case-insensitive substring).
    let alt_lower = r.alt_text.to_lowercase();
    if ALT_BLOCKLIST.iter().any(|b| alt_lower.contains(b)) {
        return false;
    }
    // Reject zero-dimension images.
    if r.width == 0 || r.height == 0 {
        return false;
    }
    true
}

/// Check whether a URL's host is a private, loopback, or link-local address.
/// This prevents SSRF attacks where a malicious provider returns URLs
/// pointing to internal services (e.g. 169.254.169.254 for AWS metadata).
///
/// Checks both literal IP addresses in the hostname and well-known
/// internal hostnames. Does NOT perform DNS resolution (to avoid blocking
/// legitimate CDN hostnames that resolve to private IPs in some networks).
fn is_private_url(url: &str) -> bool {
    let host = match extract_host(url) {
        Some(h) => h,
        None => return false,
    };

    // Check for literal IPv4 address in host.
    if let Some(ip) = parse_ipv4(&host) {
        return is_private_ipv4(ip);
    }

    // Check for literal IPv6 address in host.
    if host.starts_with('[') && host.ends_with(']') {
        let inner = &host[1..host.len()-1];
        if let Ok(ip) = inner.parse::<std::net::Ipv6Addr>() {
            return is_private_ipv6(ip);
        }
    }

    // Check for well-known internal hostnames.
    let host_lower = host.to_lowercase();
    if host_lower == "localhost"
        || host_lower == "metadata.google.internal"
        || host_lower.ends_with(".internal")
        || host_lower.ends_with(".local")
    {
        return true;
    }

    false
}

/// Extract the hostname from a URL string (no external URL parser needed).
fn extract_host(url: &str) -> Option<String> {
    // Strip scheme.
    let after_scheme = url.split("://").nth(1)?;
    // Take everything before the first / or : or ?.
    let host_end = after_scheme
        .find(|c: char| c == '/' || c == ':' || c == '?' || c == '#')
        .unwrap_or(after_scheme.len());
    let host = &after_scheme[..host_end];
    if host.is_empty() {
        return None;
    }
    Some(host.to_string())
}

/// Parse an IPv4 address from a string, returning [u8; 4].
fn parse_ipv4(s: &str) -> Option<[u8; 4]> {
    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() != 4 {
        return None;
    }
    let mut octets = [0u8; 4];
    for (i, p) in parts.iter().enumerate() {
        octets[i] = p.parse().ok()?;
    }
    Some(octets)
}

/// Check if an IPv4 address is in a private/reserved range.
fn is_private_ipv4(ip: [u8; 4]) -> bool {
    let [a, b, _c, _d] = ip;
    // 10.0.0.0/8
    if a == 10 { return true; }
    // 172.16.0.0/12
    if a == 172 && (16..=31).contains(&b) { return true; }
    // 192.168.0.0/16
    if a == 192 && b == 168 { return true; }
    // 127.0.0.0/8 (loopback)
    if a == 127 { return true; }
    // 169.254.0.0/16 (link-local, includes AWS metadata 169.254.169.254)
    if a == 169 && b == 254 { return true; }
    // 0.0.0.0/8
    if a == 0 { return true; }
    // 100.64.0.0/10 (CGNAT)
    if a == 100 && (64..=127).contains(&b) { return true; }
    false
}

/// Check if an IPv6 address is private/reserved.
fn is_private_ipv6(ip: std::net::Ipv6Addr) -> bool {
    let segs = ip.segments();
    ip.is_loopback()           // ::1
        || ip.is_unspecified() // ::
        || ip.is_unique_local() // fc00::/7
        // fe80::/10 (link-local): first byte = 0xfe, second byte's top 2 bits = 10
        || (segs[0] & 0xffc0) == 0xfe80
}

/// Check whether a license is acceptable for the given use.
pub fn license_acceptable(license: License, allow_commercial: bool) -> bool {
    match license {
        License::FreeNoAttribution | License::FreeWithAttribution => true,
        License::Commercial | License::Editorial => allow_commercial,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Attribution, ImageOrientation, License};

    fn make_result(alt: &str, license: License, photographer: &str) -> ImageResult {
        ImageResult {
            id: "1".into(),
            provider: "test".into(),
            url: "https://example.com/img.jpg".into(),
            thumb_url: "https://example.com/thumb.jpg".into(),
            width: 800,
            height: 600,
            orientation: ImageOrientation::Landscape,
            alt_text: alt.into(),
            attribution: Attribution {
                photographer: photographer.into(),
                photographer_url: String::new(),
                source_url: String::new(),
            },
            license,
            color: None,
        }
    }

    #[test]
    fn test_safe_result_passes() {
        let r = make_result("a calm beach", License::FreeNoAttribution, "Alice");
        assert!(is_safe(&r));
    }

    #[test]
    fn test_blocklist_drops_unsafe() {
        let r = make_result("explicit content", License::FreeNoAttribution, "Alice");
        assert!(!is_safe(&r));
    }

    #[test]
    fn test_attribution_required_for_pixabay_license() {
        let r = make_result("a forest", License::FreeWithAttribution, "");
        assert!(!is_safe(&r), "should drop unattributed Pixabay-style result");
    }

    #[test]
    fn test_http_url_dropped() {
        let mut r = make_result("a forest", License::FreeNoAttribution, "Alice");
        r.url = "http://insecure.com/img.jpg".into();
        assert!(!is_safe(&r));
    }

    #[test]
    fn test_zero_dims_dropped() {
        let mut r = make_result("a forest", License::FreeNoAttribution, "Alice");
        r.width = 0;
        assert!(!is_safe(&r));
    }

    #[test]
    fn test_license_acceptable() {
        assert!(license_acceptable(License::FreeNoAttribution, false));
        assert!(license_acceptable(License::FreeWithAttribution, false));
        assert!(!license_acceptable(License::Commercial, false));
        assert!(license_acceptable(License::Commercial, true));
    }

    #[test]
    fn test_filter_results_counts_dropped() {
        let results = vec![
            make_result("beach", License::FreeNoAttribution, "A"),
            make_result("explicit", License::FreeNoAttribution, "B"),
            make_result("forest", License::FreeWithAttribution, "C"),
        ];
        let (kept, dropped) = filter_results(results);
        assert_eq!(kept.len(), 2);
        assert_eq!(dropped, 1);
    }

    #[test]
    fn test_ssrf_private_ipv4_dropped() {
        let mut r = make_result("forest", License::FreeNoAttribution, "Alice");
        r.url = "https://10.0.0.1/internal.jpg".into();
        assert!(!is_safe(&r), "10.x.x.x should be blocked by SSRF filter");

        let mut r = make_result("forest", License::FreeNoAttribution, "Alice");
        r.url = "https://169.254.169.254/metadata.jpg".into();
        assert!(!is_safe(&r), "169.254.x.x (AWS metadata) should be blocked");

        let mut r = make_result("forest", License::FreeNoAttribution, "Alice");
        r.url = "https://192.168.1.1/router.jpg".into();
        assert!(!is_safe(&r), "192.168.x.x should be blocked");

        let mut r = make_result("forest", License::FreeNoAttribution, "Alice");
        r.url = "https://127.0.0.1/localhost.jpg".into();
        assert!(!is_safe(&r), "127.x.x.x should be blocked");
    }

    #[test]
    fn test_ssrf_localhost_hostname_dropped() {
        let mut r = make_result("forest", License::FreeNoAttribution, "Alice");
        r.url = "https://localhost/admin.jpg".into();
        assert!(!is_safe(&r), "localhost should be blocked");

        let mut r = make_result("forest", License::FreeNoAttribution, "Alice");
        r.url = "https://service.internal/secret.jpg".into();
        assert!(!is_safe(&r), ".internal should be blocked");
    }

    #[test]
    fn test_ssrf_public_url_passes() {
        let r = make_result("forest", License::FreeNoAttribution, "Alice");
        // Default URL is https://example.com — should pass SSRF check.
        assert!(is_safe(&r));
    }
}

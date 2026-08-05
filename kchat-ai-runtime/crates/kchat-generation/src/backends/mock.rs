//! Mock backend — used for testing when no real model is available.
//!
//! Returns deterministic, predictable output. Useful for unit tests and
//! for builds without the `llamacpp` feature.

use crate::backend::{
    BackendAdapter, BackendConfig, BackendError, BackendType, GenerationConfig, GenerationResult,
};
use crate::stream::StreamHandle;
use parking_lot::Mutex;

/// A mock backend that returns deterministic output.
pub struct MockBackend {
    loaded: Mutex<bool>,
}

impl MockBackend {
    pub fn new() -> Self {
        Self {
            loaded: Mutex::new(false),
        }
    }
}

impl Default for MockBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl BackendAdapter for MockBackend {
    fn load(&self, _config: &BackendConfig) -> Result<(), BackendError> {
        *self.loaded.lock() = true;
        Ok(())
    }

    fn unload(&self) -> Result<(), BackendError> {
        *self.loaded.lock() = false;
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        *self.loaded.lock()
    }

    fn generate(
        &self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<GenerationResult, BackendError> {
        if !self.is_loaded() {
            return Err(BackendError::NotLoaded);
        }

        // Deterministic mock output — echo the prompt + a fixed suffix
        let text = format!("{} [mock generated {} tokens]", prompt, config.max_tokens);

        Ok(GenerationResult {
            text,
            prompt_tokens: prompt.len() as u32 / 4,
            completion_tokens: config.max_tokens as u32,
            ttft_ms: 50,
            total_ms: 100,
            tokens_per_second: 50.0,
            backend: "mock".into(),
            grammar_valid: true,
        })
    }

    fn generate_stream(
        &self,
        prompt: &str,
        config: &GenerationConfig,
        stream: &StreamHandle,
    ) -> Result<GenerationResult, BackendError> {
        if !self.is_loaded() {
            return Err(BackendError::NotLoaded);
        }

        // Simulate streaming: push the prompt as "tokens"
        for word in prompt.split_whitespace() {
            if stream.is_cancelled() {
                break;
            }
            stream.push_token(format!("{} ", word));
        }

        if !stream.is_cancelled() {
            stream.push_token("[mock]");
            stream.complete(config.max_tokens as u32, 100);
        }

        Ok(GenerationResult {
            text: format!("{} [mock]", prompt),
            prompt_tokens: prompt.len() as u32 / 4,
            completion_tokens: config.max_tokens as u32,
            ttft_ms: 50,
            total_ms: 100,
            tokens_per_second: 50.0,
            backend: "mock".into(),
            grammar_valid: true,
        })
    }

    fn backend_type(&self) -> BackendType {
        BackendType::LlamaCppCpu
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kchat_core::tier::DeviceTier;

    #[test]
    fn test_mock_backend_load_unload() {
        let backend = MockBackend::new();
        assert!(!backend.is_loaded());

        let config = BackendConfig::for_tier(
            BackendType::LlamaCppCpu,
            "mock",
            "/dev/null",
            DeviceTier::Medium,
            "macos",
        );
        backend.load(&config).unwrap();
        assert!(backend.is_loaded());

        backend.unload().unwrap();
        assert!(!backend.is_loaded());
    }

    #[test]
    fn test_mock_backend_generate_not_loaded() {
        let backend = MockBackend::new();
        let config = GenerationConfig::default();
        let result = backend.generate("hello", &config);
        assert!(matches!(result, Err(BackendError::NotLoaded)));
    }

    #[test]
    fn test_mock_backend_generate_loaded() {
        let backend = MockBackend::new();
        let config = BackendConfig::for_tier(
            BackendType::LlamaCppCpu,
            "mock",
            "/dev/null",
            DeviceTier::Medium,
            "macos",
        );
        backend.load(&config).unwrap();

        let gen_config = GenerationConfig::default();
        let result = backend.generate("hello world", &gen_config).unwrap();
        assert!(result.text.contains("hello world"));
        assert_eq!(result.backend, "mock");
    }
}

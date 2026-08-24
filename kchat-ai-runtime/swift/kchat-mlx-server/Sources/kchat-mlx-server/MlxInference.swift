import Foundation
import MLX
import MLXNN
import MLXLLM
import MLXLMCommon
import Tokenizers

// MARK: - Tokenizer Loader (bridges swift-transformers to MLXLMCommon)

struct LocalTokenizerLoader: MLXLMCommon.TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        return TokenizerBridge(tokenizer: tokenizer)
    }
}

struct TokenizerBridge: MLXLMCommon.Tokenizer {
    let tokenizer: Tokenizers.Tokenizer

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenizer.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? {
        tokenizer.convertTokenToId(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        tokenizer.convertIdToToken(id)
    }

    var bosToken: String? { tokenizer.bosToken }
    var eosToken: String? { tokenizer.eosToken }
    var unknownToken: String? { tokenizer.unknownToken }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        try tokenizer.applyChatTemplate(messages: messages, tools: tools, additionalContext: additionalContext)
    }
}

// MARK: - MLX Inference Engine

class MlxInference {
    let modelContainer: ModelContainer

    /// Lock protecting `_currentAdapter` and `_currentAdapterPath`.
    /// These are accessed from HTTP handler threads (via Task.detached) and
    /// must be synchronized to prevent data races.
    private let adapterLock = NSLock()
    /// Currently loaded LoRA adapter (nil = base model only).
    private var _currentAdapter: LoRAContainer?
    /// Path of the currently loaded adapter (for logging + status).
    private var _currentAdapterPath: String?

    /// Thread-safe accessor for the current adapter path.
    var currentAdapterPath: String? {
        adapterLock.lock()
        defer { adapterLock.unlock() }
        return _currentAdapterPath
    }

    init(modelPath: String, loraPath: String? = nil) async throws {
        let url = URL(fileURLWithPath: modelPath)
        self.modelContainer = try await LLMModelFactory.shared.loadContainer(
            from: url,
            using: LocalTokenizerLoader()
        )

        if let loraPath = loraPath {
            try await loadAdapter(loraPath)
        }
    }

    // MARK: - LoRA Adapter Management

    /// Load a LoRA adapter from a directory and apply it to the model.
    /// If an adapter is already loaded, it is unloaded first.
    func loadAdapter(_ adapterPath: String) async throws {
        let adapterURL = URL(fileURLWithPath: adapterPath)

        // Verify the directory exists and contains the expected files
        let configURL = adapterURL.appending(component: "adapter_config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw NSError(
                domain: "MlxInference", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "adapter_config.json not found in \(adapterPath)"])
        }

        // Unload existing adapter before loading the new one
        if currentAdapterPath != nil {
            try await detachAdapter()
        }

        // Load the adapter container from the directory
        let adapter = try LoRAContainer.from(directory: adapterURL)

        // Apply the adapter to the model — replaces target layers + loads weights
        try await modelContainer.perform { context in
            try adapter.load(into: context.model)
        }

        adapterLock.lock()
        _currentAdapter = adapter
        _currentAdapterPath = adapterPath
        adapterLock.unlock()
        fputs("  LoRA adapter loaded: \(adapterPath)\n", stderr)
    }

    /// Detach the current LoRA adapter, reverting the model to its base form.
    func detachAdapter() async throws {
        adapterLock.lock()
        guard let adapter = _currentAdapter else {
            adapterLock.unlock()
            return
        }
        adapterLock.unlock()

        await modelContainer.perform { context in
            adapter.unload(from: context.model)
        }

        adapterLock.lock()
        _currentAdapter = nil
        _currentAdapterPath = nil
        adapterLock.unlock()
        fputs("  LoRA adapter detached\n", stderr)
    }

    /// Swap to a new LoRA adapter (convenience: unload + load in one call).
    func swapAdapter(_ adapterPath: String) async throws {
        try await loadAdapter(adapterPath)
    }

    func generate(
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        topP: Float,
        seed: UInt64
    ) async throws -> GenerateResult {
        let promptStart = Date()

        // Wrap raw prompts in chat template format for models that expect it
        // (e.g. macaw/LFM2.5 uses <|im_start|>...<|im_end|> format with thinking)
        let chatPrompt: String
        if prompt.contains("<|im_start|>") {
            chatPrompt = prompt
        } else if prompt.contains("JSON") || prompt.contains("json") || prompt.contains("ToolPlan") {
            // For JSON tasks, add a system prompt instructing the model to output only JSON
            chatPrompt = "<|im_start|>system\nYou are a JSON generator. Output ONLY valid JSON. No explanation, no markdown, no thinking, no extra text. Start your response with { and end with }.<|im_end|>\n<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n"
        } else {
            chatPrompt = "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n"
        }
        let userInput: UserInput = .init(prompt: chatPrompt)
        let lmInput = try await modelContainer.prepare(input: userInput)

        let promptMs = Date().timeIntervalSince(promptStart) * 1000.0

        // Generate
        let decodeStart = Date()
        var resultText = ""
        var tokenCount = 0

        let params = GenerateParameters(
            temperature: temperature,
            topP: topP,
            seed: seed
        )

        let stream = try await modelContainer.generate(
            input: lmInput,
            parameters: params
        )

        for await event in stream {
            if tokenCount >= maxTokens {
                break
            }
            switch event {
            case .chunk(let token):
                resultText += token
                tokenCount += 1
            case .toolCall:
                break
            case .info:
                break
            @unknown default:
                break
            }
        }

        let predictedMs = Date().timeIntervalSince(decodeStart) * 1000.0

        // Strip thinking portion from output — models like macaw/LFM2.5 generate
        // thinking...output... — we only want the output part
        var finalText = resultText
        if let outputRange = finalText.range(of: "") {
            finalText = String(finalText[outputRange.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
        }

        // MLX delivers chunks, not individual tokens. The chunk count undercounts
        // actual tokens (e.g. entire JSON output may arrive as 1 chunk).
        // Estimate actual token count from the RAW output text (before thinking
        // stripping) to avoid undercounting when thinking tokens are present.
        let estimatedTokens = max(tokenCount, resultText.count / 4)

        return GenerateResult(
            content: finalText,
            tokensPredicted: estimatedTokens,
            tokensEvaluated: 0,
            promptMs: promptMs,
            predictedMs: predictedMs
        )
    }
}

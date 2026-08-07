import Foundation

// MARK: - Request Types

struct CompletionRequest: Codable {
    let prompt: String
    let n_predict: Int?
    let temperature: Float?
    let top_p: Float?
    let top_k: Int?
    let seed: UInt64?
    let repeat_penalty: Float?
    // Accept json_schema as either a string or an object — we use it as a
    // signal to enable JSON extraction post-processing, not for grammar
    // constraining (MLX doesn't support that). Using a wrapper type avoids
    // Codable decode failures when the harness sends a JSON object.
    let json_schema: JSONSchemaField?
}

/// Wrapper that accepts either a string or a JSON object for json_schema.
enum JSONSchemaField: Codable {
    case string(String)
    case object(AnyCodable)

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let s = try? container.decode(String.self) {
            self = .string(s)
        } else if let obj = try? container.decode(AnyCodable.self) {
            self = .object(obj)
        } else {
            self = .string("")
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let s): try container.encode(s)
        case .object(let obj): try container.encode(obj)
        }
    }

    var isPresent: Bool {
        switch self {
        case .string(let s): return !s.isEmpty
        case .object: return true
        }
    }
}

/// Minimal type-erased Codable wrapper for arbitrary JSON objects.
struct AnyCodable: Codable {
    let value: Any

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let dict = try? container.decode([String: AnyCodable].self) {
            self.value = dict
        } else if let arr = try? container.decode([AnyCodable].self) {
            self.value = arr
        } else if let s = try? container.decode(String.self) {
            self.value = s
        } else if let n = try? container.decode(Double.self) {
            self.value = n
        } else if let b = try? container.decode(Bool.self) {
            self.value = b
        } else {
            self.value = NSNull()
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch value {
        case let v as [String: AnyCodable]:
            try container.encode(v)
        case let v as [AnyCodable]:
            try container.encode(v)
        case let v as String:
            try container.encode(v)
        case let v as Double:
            try container.encode(v)
        case let v as Bool:
            try container.encode(v)
        default:
            try container.encodeNil()
        }
    }
}

// MARK: - Response Types

struct CompletionResponse: Codable {
    let content: String
    let tokens_predicted: Int
    let tokens_evaluated: Int
    let prompt_ms: Double
    let predicted_ms: Double
    let predicted_per_token_ms: Double
    let prompt_per_token_ms: Double
}

struct HealthResponse: Codable {
    let status: String
}

struct ErrorResponse: Codable {
    let error: String
}

// MARK: - Generate Result

struct GenerateResult {
    let content: String
    let tokensPredicted: Int
    let tokensEvaluated: Int
    let promptMs: Double
    let predictedMs: Double
}

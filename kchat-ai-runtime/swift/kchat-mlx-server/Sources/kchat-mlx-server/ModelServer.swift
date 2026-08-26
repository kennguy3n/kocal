import Foundation
import Darwin

// MARK: - HTTP Server using POSIX sockets

class ModelServer {
    private var serverFd: Int32 = -1
    private var inference: MlxInference?
    private let port: UInt16
    private var modelReady = false
    private var running = false

    init(port: UInt16) throws {
        self.port = port

        serverFd = socket(AF_INET, SOCK_STREAM, 0)
        if serverFd < 0 {
            throw NSError(domain: "ModelServer", code: 1, userInfo: [NSLocalizedDescriptionKey: "socket() failed"])
        }

        // Allow address reuse
        var optval: Int32 = 1
        setsockopt(serverFd, SOL_SOCKET, SO_REUSEADDR, &optval, socklen_t(MemoryLayout<Int32>.size))

        // Bind to 127.0.0.1
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")
        let bindResult = withUnsafePointer(to: &addr) { ptr -> Int32 in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
                bind(serverFd, sa, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        if bindResult < 0 {
            close(serverFd)
            throw NSError(domain: "ModelServer", code: 2, userInfo: [NSLocalizedDescriptionKey: "bind() failed on port \(port)"])
        }

        // Listen
        if listen(serverFd, 16) < 0 {
            close(serverFd)
            throw NSError(domain: "ModelServer", code: 3, userInfo: [NSLocalizedDescriptionKey: "listen() failed"])
        }
    }

    func setInference(_ inference: MlxInference) {
        self.inference = inference
        self.modelReady = true
    }

    var isReady: Bool { modelReady }

    func start() {
        running = true
        // Accept loop on a background thread
        Thread.detachNewThread { [weak self] in
            self?.acceptLoop()
        }
    }

    func stop() {
        running = false
        if serverFd >= 0 {
            close(serverFd)
            serverFd = -1
        }
    }

    private func acceptLoop() {
        while running {
            var clientAddr = sockaddr_in()
            var clientLen = socklen_t(MemoryLayout<sockaddr_in>.size)
            let clientFd = withUnsafeMutablePointer(to: &clientAddr) { ptr -> Int32 in
                ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
                    accept(serverFd, sa, &clientLen)
                }
            }
            if clientFd < 0 {
                if running {
                    continue
                } else {
                    break
                }
            }
            // Handle each connection on its own thread
            Thread.detachNewThread { [weak self] in
                self?.handleClient(clientFd)
            }
        }
    }

    // MARK: - Connection Handling

    private func handleClient(_ fd: Int32) {
        defer { close(fd) }

        // Set a 30s read timeout so a stale client can't block this thread forever
        var tv = timeval(tv_sec: 30, tv_usec: 0)
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))

        // Read request data
        var allData = Data()
        var buf = [UInt8](repeating: 0, count: 65536)

        while true {
            let n = buf.withUnsafeMutableBufferPointer { ptr -> Int in
                read(fd, ptr.baseAddress, ptr.count)
            }
            if n <= 0 { break }
            allData.append(contentsOf: buf[0..<n])

            // Check if we have the full headers
            if let text = String(data: allData, encoding: .utf8) {
                if let headerEnd = text.range(of: "\r\n\r\n") {
                    let headerText = String(text[..<headerEnd.lowerBound])
                    // Check Content-Length
                    let contentLength: Int? = headerText
                        .components(separatedBy: "\r\n")
                        .first(where: { $0.lowercased().hasPrefix("content-length:") })
                        .flatMap { line -> Int? in
                            let parts = line.split(separator: ":", maxSplits: 1)
                            guard parts.count == 2 else { return nil }
                            return Int(parts[1].trimmingCharacters(in: .whitespaces))
                        }

                    let headerLen = headerText.utf8.count + 4 // +4 for \r\n\r\n
                    if let cl = contentLength {
                        let bodyReceived = allData.count - headerLen
                        if bodyReceived >= cl {
                            break // Full request received
                        }
                    } else {
                        break // No body, full request received
                    }
                }
            }
        }

        guard let request = HTTPRequest.parse(allData) else {
            sendResponse(fd, status: 400, body: Data("{\"error\":\"bad request\"}".utf8))
            return
        }

        handleRequest(request, fd: fd)
    }

    // MARK: - Request Routing

    private func handleRequest(_ request: HTTPRequest, fd: Int32) {
        switch (request.method, request.path) {
        case ("GET", "/health"):
            if modelReady {
                sendJsonResponse(fd, status: 200, body: HealthResponse(status: "ok"))
            } else {
                sendJsonResponse(fd, status: 503, body: HealthResponse(status: "loading"))
            }

        case ("POST", "/completion"):
            if !modelReady {
                sendJsonResponse(fd, status: 503, body: ErrorResponse(error: "model still loading"))
                return
            }
            handleCompletion(request, fd: fd)

        case ("POST", "/completion/stream"):
            if !modelReady {
                sendJsonResponse(fd, status: 503, body: ErrorResponse(error: "model still loading"))
                return
            }
            handleCompletionStream(request, fd: fd)

        case ("GET", "/v1/models"):
            struct ModelsList: Codable {
                let object: String
                let data: [String]
            }
            sendJsonResponse(fd, status: 200, body: ModelsList(object: "list", data: []))

        case ("POST", "/v1/chat/completions"):
            if !modelReady {
                sendJsonResponse(fd, status: 503, body: ErrorResponse(error: "model still loading"))
                return
            }
            handleChatCompletion(request, fd: fd)

        case ("POST", "/lora/load"):
            if !modelReady {
                sendJsonResponse(fd, status: 503, body: ErrorResponse(error: "model still loading"))
                return
            }
            handleLoraLoad(request, fd: fd)

        case ("POST", "/lora/detach"):
            if !modelReady {
                sendJsonResponse(fd, status: 503, body: ErrorResponse(error: "model still loading"))
                return
            }
            handleLoraDetach(request, fd: fd)

        case ("GET", "/lora/status"):
            handleLoraStatus(request, fd: fd)

        default:
            sendJsonResponse(fd, status: 404, body: ErrorResponse(error: "not found"))
        }
    }

    // MARK: - Completion Endpoint

    private func handleCompletion(_ request: HTTPRequest, fd: Int32) {
        guard !request.body.isEmpty else {
            sendJsonResponse(fd, status: 400, body: ErrorResponse(error: "empty body"))
            return
        }

        do {
            let req = try JSONDecoder().decode(CompletionRequest.self, from: request.body)

            let maxTokens = req.n_predict ?? 128
            let temperature = req.temperature ?? 0.7
            let topP = req.top_p ?? 0.9
            let topK = req.top_k ?? 40
            let repeatPenalty = req.repeat_penalty ?? 1.1
            let seed = req.seed ?? 42
            let stopSequences = req.stop ?? []

            // Bridge async generate() to sync using a semaphore with timeout.
            // Task.detached runs on the Swift Concurrency cooperative pool — NOT
            // on this thread — so sem.wait() won't deadlock with the task.
            // The 120s timeout prevents infinite hangs if the cooperative pool
            // is exhausted or the model hangs internally.
            let sem = DispatchSemaphore(value: 0)
            var responseResult: Result<GenerateResult, Error>!

            Task.detached {
                do {
                    let result = try await self.inference!.generate(
                        prompt: req.prompt,
                        maxTokens: maxTokens,
                        temperature: temperature,
                        topP: topP,
                        topK: topK,
                        repeatPenalty: repeatPenalty,
                        seed: seed,
                        stopSequences: stopSequences
                    )
                    responseResult = .success(result)
                } catch {
                    responseResult = .failure(error)
                }
                sem.signal()
            }

            let waitResult = sem.wait(timeout: .now() + .seconds(120))
            if waitResult == .timedOut {
                sendJsonResponse(fd, status: 504, body: ErrorResponse(error: "generation timed out (120s)"))
                return
            }

            switch responseResult! {
            case .success(let result):
                var content = result.content
                // If json_schema was requested, extract just the JSON from the output
                if let schema = req.json_schema, schema.isPresent {
                    content = extractJSON(from: content)
                }
                let response = CompletionResponse(
                    content: content,
                    tokens_predicted: result.tokensPredicted,
                    tokens_evaluated: result.tokensEvaluated,
                    prompt_ms: result.promptMs,
                    predicted_ms: result.predictedMs,
                    predicted_per_token_ms: result.predictedMs / Double(max(result.tokensPredicted, 1)),
                    prompt_per_token_ms: result.promptMs / Double(max(result.tokensEvaluated, 1))
                )
                sendJsonResponse(fd, status: 200, body: response)
            case .failure(let error):
                sendJsonResponse(fd, status: 500, body: ErrorResponse(error: "\(error)"))
            }

        } catch {
            sendJsonResponse(fd, status: 400, body: ErrorResponse(error: "invalid JSON: \(error)"))
        }
    }

    // MARK: - Streaming Completion Endpoint (SSE)

    private func handleCompletionStream(_ request: HTTPRequest, fd: Int32) {
        guard !request.body.isEmpty else {
            sendJsonResponse(fd, status: 400, body: ErrorResponse(error: "empty body"))
            return
        }

        do {
            let req = try JSONDecoder().decode(CompletionRequest.self, from: request.body)

            let maxTokens = req.n_predict ?? 128
            let temperature = req.temperature ?? 0.7
            let topP = req.top_p ?? 0.9
            let topK = req.top_k ?? 40
            let repeatPenalty = req.repeat_penalty ?? 1.1
            let seed = req.seed ?? 42
            let stopSequences = req.stop ?? []

            // Send SSE headers immediately so the client can start reading
            let header = "HTTP/1.1 200 OK\r\n" +
                         "Content-Type: text/event-stream\r\n" +
                         "Cache-Control: no-cache\r\n" +
                         "Connection: close\r\n" +
                         "Access-Control-Allow-Origin: *\r\n" +
                         "\r\n"
            let headerData = Data(header.utf8)
            _ = headerData.withUnsafeBytes { ptr in
                write(fd, ptr.baseAddress, ptr.count)
            }

            // Stream tokens via generateStream callback.
            // Task.detached runs on the cooperative pool; we use a semaphore
            // to bridge back to this thread for writing each SSE event.
            let sem = DispatchSemaphore(value: 0)
            var finalResult: GenerateResult?
            var genError: Error?

            Task.detached {
                do {
                    let result = try await self.inference!.generateStream(
                        prompt: req.prompt,
                        maxTokens: maxTokens,
                        temperature: temperature,
                        topP: topP,
                        topK: topK,
                        repeatPenalty: repeatPenalty,
                        seed: seed,
                        stopSequences: stopSequences
                    ) { token in
                        // Encode token as SSE data event and write immediately.
                        // JSON-encode the token string to handle newlines/special chars.
                        let encoded = (try? JSONEncoder().encode(token)) ?? Data("\"\"".utf8)
                        let sseEvent = "data: " + String(data: encoded, encoding: .utf8)! + "\n\n"
                        let eventData = Data(sseEvent.utf8)
                        eventData.withUnsafeBytes { ptr in
                            _ = write(fd, ptr.baseAddress, ptr.count)
                        }
                    }
                    finalResult = result
                } catch {
                    genError = error
                }
                sem.signal()
            }

            // Wait for generation to complete (120s timeout)
            let waitResult = sem.wait(timeout: .now() + .seconds(120))
            if waitResult == .timedOut {
                let errEvent = "data: {\"error\":\"generation timed out\"}\n\n"
                _ = write(fd, errEvent, errEvent.count)
                return
            }

            if let error = genError {
                let errEvent = "data: {\"error\":\"\(error)\"}\n\n"
                _ = write(fd, errEvent, errEvent.count)
                return
            }

            // Send final result as a [DONE] event
            if let result = finalResult {
                let finalJson = "{\"content\":\"\(result.content.replacingOccurrences(of: "\"", with: "\\\""))\",\"tokens_predicted\":\(result.tokensPredicted),\"tokens_evaluated\":\(result.tokensEvaluated),\"prompt_ms\":\(result.promptMs),\"predicted_ms\":\(result.predictedMs)}"
                let doneEvent = "data: " + finalJson + "\n\ndata: [DONE]\n\n"
                _ = write(fd, doneEvent, doneEvent.count)
            }

        } catch {
            sendJsonResponse(fd, status: 400, body: ErrorResponse(error: "invalid JSON: \(error)"))
        }
    }

    // MARK: - Chat Completion Endpoint (OpenAI-compatible)

    private func handleChatCompletion(_ request: HTTPRequest, fd: Int32) {
        guard !request.body.isEmpty else {
            sendJsonResponse(fd, status: 400, body: ErrorResponse(error: "empty body"))
            return
        }

        struct ChatRequest: Codable {
            let messages: [ChatMessage]
            let max_tokens: Int?
            let temperature: Float?
            let top_p: Float?
            let seed: UInt64?
        }
        struct ChatMessage: Codable {
            let role: String
            let content: String
        }

        do {
            let req = try JSONDecoder().decode(ChatRequest.self, from: request.body)

            let prompt = req.messages.map { msg in
                switch msg.role {
                case "system": return "<|im_start|>system\n\(msg.content)<|im_end|>"
                case "user": return "<|im_start|>user\n\(msg.content)<|im_end|>"
                case "assistant": return "<|im_start|>assistant\n\(msg.content)<|im_end|>"
                default: return "<|im_start|>\(msg.role)\n\(msg.content)<|im_end|>"
                }
            }.joined(separator: "\n") + "\n<|im_start|>assistant\n"

            let maxTokens = req.max_tokens ?? 128
            let temperature = req.temperature ?? 0.7
            let topP = req.top_p ?? 0.9
            let seed = req.seed ?? 42

            let sem = DispatchSemaphore(value: 0)
            var responseResult: Result<GenerateResult, Error>!

            Task.detached {
                do {
                    let result = try await self.inference!.generate(
                        prompt: prompt,
                        maxTokens: maxTokens,
                        temperature: temperature,
                        topP: topP,
                        seed: seed
                    )
                    responseResult = .success(result)
                } catch {
                    responseResult = .failure(error)
                }
                sem.signal()
            }

            let waitResult = sem.wait(timeout: .now() + .seconds(120))
            if waitResult == .timedOut {
                sendJsonResponse(fd, status: 504, body: ErrorResponse(error: "generation timed out (120s)"))
                return
            }

            switch responseResult! {
            case .success(let result):
                let response: [String: Any] = [
                    "id": "chatcmpl-\(UUID().uuidString.prefix(8))",
                    "object": "chat.completion",
                    "choices": [[
                        "index": 0,
                        "message": [
                            "role": "assistant",
                            "content": result.content
                        ],
                        "finish_reason": "stop"
                    ]],
                    "usage": [
                        "prompt_tokens": result.tokensEvaluated,
                        "completion_tokens": result.tokensPredicted,
                        "total_tokens": result.tokensEvaluated + result.tokensPredicted
                    ]
                ]
                let jsonData = try JSONSerialization.data(withJSONObject: response)
                sendResponse(fd, status: 200, body: jsonData)
            case .failure(let error):
                sendJsonResponse(fd, status: 500, body: ErrorResponse(error: "\(error)"))
            }

        } catch {
            sendJsonResponse(fd, status: 400, body: ErrorResponse(error: "invalid JSON: \(error)"))
        }
    }

    // MARK: - LoRA Endpoints

    private func handleLoraLoad(_ request: HTTPRequest, fd: Int32) {
        guard !request.body.isEmpty else {
            sendJsonResponse(fd, status: 400, body: ErrorResponse(error: "empty body"))
            return
        }

        do {
            let req = try JSONDecoder().decode(LoraLoadRequest.self, from: request.body)
            let adapterPath = req.adapter_path

            guard !adapterPath.isEmpty else {
                sendJsonResponse(fd, status: 400, body: ErrorResponse(error: "adapter_path required"))
                return
            }

            guard FileManager.default.fileExists(atPath: adapterPath) else {
                sendJsonResponse(fd, status: 404, body: ErrorResponse(error: "adapter directory not found: \(adapterPath)"))
                return
            }

            // Bridge async swapAdapter() to sync using a semaphore.
            let sem = DispatchSemaphore(value: 0)
            var loadError: Error?

            Task.detached {
                do {
                    try await self.inference!.swapAdapter(adapterPath)
                } catch {
                    loadError = error
                }
                sem.signal()
            }

            let waitResult = sem.wait(timeout: .now() + .seconds(60))
            if waitResult == .timedOut {
                sendJsonResponse(fd, status: 504, body: ErrorResponse(error: "LoRA load timed out (60s)"))
                return
            }

            if let error = loadError {
                sendJsonResponse(fd, status: 500, body: ErrorResponse(error: "failed to load adapter: \(error)"))
                return
            }

            sendJsonResponse(fd, status: 200, body: LoraLoadResponse(status: "ok", adapter: adapterPath))

        } catch {
            sendJsonResponse(fd, status: 400, body: ErrorResponse(error: "invalid JSON: \(error)"))
        }
    }

    private func handleLoraDetach(_ request: HTTPRequest, fd: Int32) {
        let sem = DispatchSemaphore(value: 0)
        var detachError: Error?

        Task.detached {
            do {
                try await self.inference!.detachAdapter()
            } catch {
                detachError = error
            }
            sem.signal()
        }

        let waitResult = sem.wait(timeout: .now() + .seconds(30))
        if waitResult == .timedOut {
            sendJsonResponse(fd, status: 504, body: ErrorResponse(error: "LoRA detach timed out (30s)"))
            return
        }

        if let error = detachError {
            sendJsonResponse(fd, status: 500, body: ErrorResponse(error: "failed to detach adapter: \(error)"))
            return
        }

        sendJsonResponse(fd, status: 200, body: LoraDetachResponse(status: "ok", adapter: nil))
    }

    private func handleLoraStatus(_ request: HTTPRequest, fd: Int32) {
        let adapterPath = inference?.currentAdapterPath
        let response = LoraLoadResponse(status: adapterPath != nil ? "loaded" : "none", adapter: adapterPath)
        sendJsonResponse(fd, status: 200, body: response)
    }

    // MARK: - Response Helpers

    private func sendJsonResponse<T: Encodable>(_ fd: Int32, status: Int, body: T) {
        do {
            let jsonData = try JSONEncoder().encode(body)
            sendResponse(fd, status: status, body: jsonData)
        } catch {
            sendResponse(fd, status: 500, body: Data("{\"error\":\"internal error\"}".utf8))
        }
    }

    private func sendResponse(_ fd: Int32, status: Int, body: Data) {
        let statusText = HTTPStatus.text(for: status)
        let header = "HTTP/1.1 \(status) \(statusText)\r\n" +
                     "Content-Type: application/json\r\n" +
                     "Content-Length: \(body.count)\r\n" +
                     "Connection: close\r\n" +
                     "Access-Control-Allow-Origin: *\r\n" +
                     "\r\n"
        var responseData = Data(header.utf8)
        responseData.append(body)

        responseData.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) in
            var remaining = ptr.count
            var offset = 0
            while remaining > 0 {
                let written = write(fd, ptr.baseAddress!.advanced(by: offset), remaining)
                if written <= 0 { break }
                remaining -= written
                offset += written
            }
        }
    }
}

// MARK: - HTTP Request Parsing

struct HTTPRequest {
    let method: String
    let path: String
    let headers: [String: String]
    var body: Data
    var contentLength: Int? {
        headers["Content-Length"].flatMap { Int($0) }
    }

    static func parse(_ data: Data) -> HTTPRequest? {
        guard let text = String(data: data, encoding: .utf8) else { return nil }

        // Find header/body boundary
        guard let headerEnd = text.range(of: "\r\n\r\n") else { return nil }
        let headerText = String(text[..<headerEnd.lowerBound])
        let bodyText = String(text[headerEnd.upperBound...])

        let lines = headerText.components(separatedBy: "\r\n")
        guard let firstLine = lines.first else { return nil }

        let parts = firstLine.components(separatedBy: " ")
        guard parts.count >= 2 else { return nil }

        let method = parts[0]
        let path = parts[1]

        var headers: [String: String] = [:]
        for line in lines.dropFirst() {
            if let colonIdx = line.firstIndex(of: ":") {
                let key = String(line[..<colonIdx]).trimmingCharacters(in: .whitespaces)
                let value = String(line[line.index(after: colonIdx)...]).trimmingCharacters(in: .whitespaces)
                headers[key] = value
            }
        }

        let body = bodyText.data(using: .utf8) ?? Data()
        return HTTPRequest(method: method, path: path, headers: headers, body: body)
    }
}

// MARK: - JSON Extraction

func extractJSON(from text: String) -> String {
    var trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)

    // Strip markdown code fences: ```json ... ``` or ``` ... ```
    if trimmed.hasPrefix("```") {
        // Remove opening fence line
        if let firstNewline = trimmed.firstIndex(of: "\n") {
            trimmed = String(trimmed[trimmed.index(after: firstNewline)...])
        }
        // Remove closing fence
        if let closingFence = trimmed.range(of: "```") {
            trimmed = String(trimmed[..<closingFence.lowerBound])
        }
        trimmed = trimmed.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // Try parsing the whole string as JSON first
    if let _ = try? JSONSerialization.jsonObject(with: Data(trimmed.utf8), options: []) {
        return trimmed
    }

    // Find the first { or [ and try to extract balanced JSON
    for (i, c) in trimmed.enumerated() {
        if c == "{" || c == "[" {
            let substring = String(trimmed.dropFirst(i))
            if let endIdx = findJSONEnd(substring) {
                let candidate = String(substring[substring.startIndex..<substring.index(substring.startIndex, offsetBy: endIdx)])
                if let _ = try? JSONSerialization.jsonObject(with: Data(candidate.utf8), options: []) {
                    return candidate
                }
            }
        }
    }

    return trimmed
}

private func findJSONEnd(_ s: String) -> Int? {
    var depth = 0
    var inString = false
    var escape = false
    var byteIdx = 0

    for ch in s {
        if escape {
            escape = false
            byteIdx += String(ch).utf8.count
            continue
        }
        if ch == "\\" && inString {
            escape = true
            byteIdx += 1
            continue
        }
        if ch == "\"" {
            inString = !inString
            byteIdx += 1
            continue
        }
        if inString {
            byteIdx += String(ch).utf8.count
            continue
        }
        switch ch {
        case "{", "[":
            depth += 1
        case "}", "]":
            depth -= 1
            if depth == 0 {
                return byteIdx + 1
            }
        default:
            break
        }
        byteIdx += String(ch).utf8.count
    }

    return nil
}

// MARK: - HTTP Status

enum HTTPStatus {
    static func text(for code: Int) -> String {
        switch code {
        case 200: return "OK"
        case 400: return "Bad Request"
        case 404: return "Not Found"
        case 500: return "Internal Server Error"
        case 503: return "Service Unavailable"
        case 504: return "Gateway Timeout"
        default: return "Unknown"
        }
    }
}

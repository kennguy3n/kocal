import Foundation

// MARK: - Argument Parsing

struct CliArgs {
    var modelPath: String?
    var port: UInt16 = 18888
    var host: String = "127.0.0.1"
}

func parseArgs() -> CliArgs {
    var args = CliArgs()
    let argv = CommandLine.arguments

    var i = 1
    while i < argv.count {
        let arg = argv[i]
        switch arg {
        case "--model", "-m":
            i += 1
            if i < argv.count {
                args.modelPath = argv[i]
            }
        case "--port", "-p":
            i += 1
            if i < argv.count, let port = UInt16(argv[i]) {
                args.port = port
            }
        case "--host":
            i += 1
            if i < argv.count {
                args.host = argv[i]
            }
        case "--help", "-h":
            print("""
            kchat-mlx-server — MLX inference server for kchat eval

            Usage: kchat-mlx-server --model <path> [--port <port>] [--host <host>]

            Options:
              --model <path>   Path to MLX model directory (containing config.json, safetensors)
              --port <port>    Port to listen on (default: 18888)
              --host <host>    Host to bind to (default: 127.0.0.1)
              --help           Show this help message

            Endpoints:
              GET  /health              — Health check
              POST /completion          — llama-server compatible completion API
              POST /v1/chat/completions — OpenAI-compatible chat completion API
            """)
            exit(0)
        default:
            break
        }
        i += 1
    }

    return args
}

// MARK: - Main Entry Point

@main
struct Main {
    static func main() async {
        let args = parseArgs()

        guard let modelPath = args.modelPath else {
            fputs("error: --model is required\n", stderr)
            fputs("usage: kchat-mlx-server --model <path> [--port <port>] [--host <host>]\n", stderr)
            exit(1)
        }

        // Verify model directory exists
        guard FileManager.default.fileExists(atPath: modelPath) else {
            fputs("error: model directory not found: \(modelPath)\n", stderr)
            exit(1)
        }

        // Start server first so health checks respond immediately
        let server: ModelServer
        do {
            server = try ModelServer(port: args.port)
        } catch {
            fputs("error: failed to start server on port \(args.port): \(error)\n", stderr)
            exit(1)
        }

        server.start()
        fputs("kchat-mlx-server listening on http://\(args.host):\(args.port)\n", stderr)
        fputs("  loading model from \(modelPath)...\n", stderr)

        // Load model on a background thread — health returns 503 until this completes
        // Task.detached is critical: Task {} inherits the main actor, which would block
        // NWListener from accepting connections during model loading.
        Task.detached {
            do {
                let inference = try await MlxInference(modelPath: modelPath)
                server.setInference(inference)
                fputs("  model loaded — ready\n", stderr)
            } catch {
                fputs("error: failed to load model: \(error)\n", stderr)
                exit(1)
            }
        }

        fputs("  GET  /health              — health check\n", stderr)
        fputs("  POST /completion          — completion API\n", stderr)
        fputs("  POST /v1/chat/completions — chat completion API\n", stderr)

        // Keep running until interrupted
        signal(SIGINT) { _ in
            fputs("\nshutting down...\n", stderr)
            exit(0)
        }
        signal(SIGTERM) { _ in
            exit(0)
        }

        // Block forever — dispatchMain() starts the main run loop and never returns.
        // This properly integrates with Swift Concurrency and GCD, unlike
        // RunLoop.current.run() which can starve the cooperative thread pool.
        // The POSIX socket accept loop runs on its own detached Thread.
        dispatchMain()
    }
}

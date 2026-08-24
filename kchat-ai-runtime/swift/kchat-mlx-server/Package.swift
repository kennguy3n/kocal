// swift-tools-version: 5.12
import PackageDescription

let package = Package(
    name: "kchat-mlx-server",
    platforms: [.macOS(.v14)],
    dependencies: [
        // PrismML mlx-swift fork: adds 1-bit quantization Metal kernels for Bonsai models
        // Based on mlx-swift 0.31.6 + 1-bit kernel support + M5/gen-17 NAX fix
        .package(url: "https://github.com/PrismML-Eng/mlx-swift.git", branch: "v0.31.6_prism"),
        // mlx-swift-lm: ModelContainer, LLMModelFactory, generation pipeline
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", branch: "main"),
        // swift-transformers: AutoTokenizer for loading tokenizer.json
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        .executableTarget(
            name: "kchat-mlx-server",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/kchat-mlx-server"
        )
    ]
)

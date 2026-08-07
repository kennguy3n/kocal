// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "kchat-mlx-server",
    platforms: [.macOS(.v14)],
    dependencies: [
        // mlx-swift: core MLX arrays + neural network modules
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.22.0"),
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

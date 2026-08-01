// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "TurboFieldfare",
    platforms: [
        .macOS(.v26),
        .iOS(.v26),
    ],
    products: [
        .library(name: "TurboFieldfare", targets: ["TurboFieldfare"]),
        .executable(name: "TurboFieldfareRepack", targets: ["TurboFieldfareRepack"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "TurboFieldfare",
            dependencies: [
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/TurboFieldfare",
            resources: [
                .copy("Metal"),
            ]
        ),
        .target(
            name: "TurboFieldfareRepackCore",
            path: "Sources/TurboFieldfareRepack/Core"
        ),
        .executableTarget(
            name: "TurboFieldfareRepack",
            dependencies: ["TurboFieldfareRepackCore"],
            path: "Sources/TurboFieldfareRepack/Command"
        ),
        .testTarget(
            name: "TurboFieldfareRepackTests",
            dependencies: ["TurboFieldfareRepackCore"],
            path: "Tests/TurboFieldfareRepack"
        ),
    ]
)

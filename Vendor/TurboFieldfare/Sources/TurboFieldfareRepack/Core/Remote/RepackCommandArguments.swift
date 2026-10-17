import Foundation

public struct RepackCommandArguments: Equatable, Sendable {
    public enum Mode: Equatable, Sendable {
        case install
        case writeVision
        case discardPartial
        case verifyInstall
    }

    public var mode: Mode = .install
    public var model = SupportedModelSource.default
    public var output: String?
    public var textModel: String?
    public var overwrite = false
    public var resume = false
    public var dryRun = false
    public var inputGTurbo: String?

    public static let usage = """
    Usage:
      TurboFieldfareRepack [--model <gemma4|gemma4-qat|qwen36|ornith15|lfm25>] --output <model.gturbo> [--overwrite] [--resume]
      TurboFieldfareRepack [--model <gemma4|gemma4-qat|qwen36|ornith15|lfm25>] --dry-run --output <model.gturbo>
      TurboFieldfareRepack --write-vision --model <gemma4|gemma4-qat> --text-model <text.gturbo> --output <name.vision.gturbo> [--overwrite] [--resume]
      TurboFieldfareRepack --discard-partial --output <model.gturbo>
      TurboFieldfareRepack --verify-install --input-gturbo <model.gturbo>
      TurboFieldfareRepack --help

    The installer streams the selected checkpoint (default: the supported Gemma 4
    checkpoint) from Hugging Face and repackages it without materializing the
    source checkpoint on disk. Set HF_TOKEN only if Hugging Face requests
    authentication. A cancelled or interrupted download can be continued with
    --resume or removed with --discard-partial.

    Vision-write mode streams only the 358 vision tensors from a pinned Gemma
    revision into a companion directory bound to an existing text .gturbo. Pass
    --text-model for the text pack and --output for the companion
    (Name.vision.gturbo). --model must be gemma4 or gemma4-qat; every other
    model is rejected. Use --resume or --overwrite the same way as a text
    install.
    """

    public static func parse(_ values: [String]) throws -> RepackCommandArguments {
        var parsed = RepackCommandArguments()
        var index = 1
        while index < values.count {
            let flag = values[index]
            switch flag {
            case "--help":
                throw RepackCommandParseError.help
            case "--overwrite":
                parsed.overwrite = true
                index += 1
            case "--resume":
                parsed.resume = true
                index += 1
            case "--discard-partial":
                parsed.mode = .discardPartial
                index += 1
            case "--verify-install":
                parsed.mode = .verifyInstall
                index += 1
            case "--write-vision":
                parsed.mode = .writeVision
                index += 1
            case "--dry-run":
                parsed.dryRun = true
                index += 1
            case "--model", "--source":
                guard index + 1 < values.count else {
                    throw RepackCommandParseError.missingValue(flag)
                }
                guard let source = SupportedModelSource.named(values[index + 1]) else {
                    throw RepackCommandParseError.invalidMode(
                        "unknown model \"\(values[index + 1])\"; supported: "
                        + SupportedModelSource.all.map(\.name).joined(separator: ", "))
                }
                parsed.model = source
                index += 2
            case "--output", "--input-gturbo", "--text-model":
                guard index + 1 < values.count else {
                    throw RepackCommandParseError.missingValue(flag)
                }
                switch flag {
                case "--output":
                    parsed.output = values[index + 1]
                case "--input-gturbo":
                    parsed.inputGTurbo = values[index + 1]
                default:
                    parsed.textModel = values[index + 1]
                }
                index += 2
            default:
                throw RepackCommandParseError.unknown(flag)
            }
        }
        try parsed.validate()
        return parsed
    }

    private func validate() throws {
        guard !(resume && mode == .discardPartial) else {
            throw RepackCommandParseError.invalidMode(
                "--resume and --discard-partial are mutually exclusive")
        }
        switch mode {
        case .discardPartial:
            guard output != nil else {
                throw RepackCommandParseError.missingRequired("--output")
            }
            guard inputGTurbo == nil, textModel == nil, !overwrite else {
                throw RepackCommandParseError.invalidMode("--discard-partial only accepts --output")
            }
            guard !dryRun else {
                throw RepackCommandParseError.invalidMode("--discard-partial does not accept --dry-run")
            }
        case .verifyInstall:
            guard inputGTurbo != nil else {
                throw RepackCommandParseError.missingRequired("--input-gturbo")
            }
            guard output == nil, textModel == nil, !overwrite, !resume else {
                throw RepackCommandParseError.invalidMode("verification accepts only --input-gturbo")
            }
            guard !dryRun else {
                throw RepackCommandParseError.invalidMode("verification does not accept --dry-run")
            }
        case .writeVision:
            guard model.supportsVisionCompanion else {
                throw RepackCommandParseError.invalidMode(
                    "vision-write accepts only --model gemma4 or gemma4-qat")
            }
            guard textModel != nil else {
                throw RepackCommandParseError.missingRequired("--text-model")
            }
            guard output != nil else {
                throw RepackCommandParseError.missingRequired("--output")
            }
            guard inputGTurbo == nil else {
                throw RepackCommandParseError.invalidMode(
                    "--input-gturbo is not used with --write-vision")
            }
            guard !dryRun else {
                throw RepackCommandParseError.invalidMode("vision-write does not accept --dry-run")
            }
        case .install:
            guard output != nil else {
                throw RepackCommandParseError.missingRequired("--output")
            }
            guard inputGTurbo == nil else {
                throw RepackCommandParseError.invalidMode("--input-gturbo requires --verify-install")
            }
            guard textModel == nil else {
                throw RepackCommandParseError.invalidMode("--text-model requires --write-vision")
            }
        }
    }
}

public enum RepackCommandParseError: Error, Equatable, CustomStringConvertible {
    case help
    case unknown(String)
    case missingValue(String)
    case missingRequired(String)
    case invalidMode(String)

    public var description: String {
        switch self {
        case .help: return "help"
        case .unknown(let flag): return "unknown argument: \(flag)"
        case .missingValue(let flag): return "missing value for \(flag)"
        case .missingRequired(let flag): return "missing required argument: \(flag)"
        case .invalidMode(let message): return message
        }
    }
}

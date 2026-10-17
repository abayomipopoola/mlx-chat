import Foundation
import TurboFieldfareRepackCore

private func printError(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

private func run(_ values: [String]) async -> Int32 {
    let arguments: RepackCommandArguments
    do {
        arguments = try RepackCommandArguments.parse(values)
    } catch RepackCommandParseError.help {
        print(RepackCommandArguments.usage)
        return 0
    } catch {
        printError("error: \(error)\n\n\(RepackCommandArguments.usage)")
        return 2
    }

    if arguments.mode == .discardPartial, let output = arguments.output {
        do {
            try RemoteStreamingRepacker.discardPartial(outputDirectory: output)
            print("Discarded saved download for \(output)")
            return 0
        } catch {
            printError("discard failed: \(error)")
            return 1
        }
    }

    if arguments.mode == .verifyInstall, let input = arguments.inputGTurbo {
        do {
            let result = try VerifiedInstallTool.run(
                options: VerifyInstallOptions(inputGTurbo: input))
            print("Verified \(result.fileCount) files (\(result.bytesVerified) bytes)")
            print("Receipt: \(result.receiptPath)")
            return 0
        } catch {
            printError("verification failed: \(error)")
            return 1
        }
    }

    if arguments.mode == .writeVision {
        guard let output = arguments.output, let textModel = arguments.textModel else {
            return 2
        }
        do {
            let options = try arguments.model.visionCompanionOptions(
                textModelDirectory: URL(fileURLWithPath: textModel),
                outputDirectory: URL(fileURLWithPath: output),
                overwrite: arguments.overwrite,
                token: ProcessInfo.processInfo.environment["HF_TOKEN"],
                resume: arguments.resume)
            let result = try await RemoteVisionCompanionInstaller(options: options).run()
            print("Installed vision companion for \(arguments.model.displayName)")
            print("Source revision: \(result.resolvedCommit)")
            print("Download bytes: \(result.remoteBytesToDownload)")
            print("Companion: \(result.outputDir)")
            return 0
        } catch {
            printError("vision-write failed: \(error)")
            return 1
        }
    }

    guard let output = arguments.output else { return 2 }
    let source = arguments.model
    let options = source.installOptions(
        outputDirectory: URL(fileURLWithPath: output),
        overwrite: arguments.overwrite,
        token: ProcessInfo.processInfo.environment["HF_TOKEN"],
        resume: arguments.resume,
        dryRun: arguments.dryRun)
    do {
        let result = try await RemoteStreamingRepacker(options: options).run()
        print(arguments.dryRun ? "Validated \(source.displayName)" : "Installed \(source.displayName)")
        print("Source revision: \(result.resolvedCommit)")
        print("Download bytes: \(result.remoteBytesToDownload)")
        print("Model: \(result.outputDir)")
        return 0
    } catch {
        printError("install failed: \(error)")
        return 1
    }
}

exit(await run(CommandLine.arguments))

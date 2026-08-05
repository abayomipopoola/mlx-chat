import Foundation

/// Repairs known compatibility differences in third-party MLX snapshots.
///
/// Some LFM2.5 conversions wrap the text model in `language_model` and keep
/// that wrapper in every safetensors key. MLX Swift's text-only LFM2 module
/// expects the same weights without that wrapper. Some conversions also omit
/// `block_ff_dim`, which MLX Swift needs to size the feed-forward layers. The
/// tensor bytes and shapes are already correct, so only metadata is repaired.
enum ModelWeightNormalizer {
    enum NormalizationError: LocalizedError {
        case malformedHeader(String)
        case keyCollision(String)
        case enlargedHeader(String)

        var errorDescription: String? {
            switch self {
            case .malformedHeader(let file):
                return "Couldn't read the safetensors header in \(file)."
            case .keyCollision(let key):
                return "Weight normalization would overwrite \(key)."
            case .enlargedHeader(let file):
                return "The normalized safetensors header no longer fits in \(file)."
            }
        }
    }

    private static let wrappedPrefix = "language_model."
    private static let maximumHeaderBytes = 64 * 1024 * 1024
    private static let backupDirectoryName = ".mlxchat-weight-header-backups"

    static func normalizeIfNeeded(in directory: URL) throws {
        let configURL = directory.appendingPathComponent("config.json")
        guard let configData = try? Data(contentsOf: configURL),
              let config = try? JSONSerialization.jsonObject(with: configData) as? [String: Any],
              (config["model_type"] as? String)?.lowercased() == "lfm2"
        else { return }

        try normalizeConfigIfNeeded(configData, at: configURL)

        let fileManager = FileManager.default
        let files = try fileManager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles])
            .filter { $0.pathExtension == "safetensors" }

        var repairedAnyHeader = false
        for file in files {
            repairedAnyHeader = try normalizeSafetensorsFile(file) || repairedAnyHeader
        }

        if repairedAnyHeader {
            try normalizeIndexIfPresent(in: directory)
        }
    }

    /// Pure header transformation used by focused tests.
    static func normalizedHeaderJSON(_ data: Data) throws -> Data? {
        guard var dictionary = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { throw NormalizationError.malformedHeader("header") }
        guard try normalizeKeys(in: &dictionary) else { return nil }
        return try JSONSerialization.data(withJSONObject: dictionary, options: [.sortedKeys])
    }

    /// Pure index transformation used by focused tests.
    static func normalizedIndexJSON(_ data: Data) throws -> Data? {
        guard var dictionary = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              var weightMap = dictionary["weight_map"] as? [String: Any]
        else { return nil }
        guard try normalizeKeys(in: &weightMap) else { return nil }
        dictionary["weight_map"] = weightMap
        return try JSONSerialization.data(withJSONObject: dictionary, options: [.sortedKeys])
    }

    /// Pure config transformation used by focused tests.
    static func normalizedConfigJSON(_ data: Data) throws -> Data? {
        guard var dictionary = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        guard dictionary["block_ff_dim"] == nil,
              let intermediateSize = dictionary["intermediate_size"] as? NSNumber
        else { return nil }
        dictionary["block_ff_dim"] = intermediateSize
        return try JSONSerialization.data(withJSONObject: dictionary, options: [.sortedKeys])
    }

    private static func normalizeSafetensorsFile(_ file: URL) throws -> Bool {
        let handle = try FileHandle(forReadingFrom: file)
        let lengthData = try handle.read(upToCount: 8) ?? Data()
        guard lengthData.count == 8 else {
            try handle.close()
            throw NormalizationError.malformedHeader(file.lastPathComponent)
        }

        let headerLength = lengthData.enumerated().reduce(UInt64(0)) { value, byte in
            value | (UInt64(byte.element) << UInt64(byte.offset * 8))
        }
        guard headerLength > 0,
              headerLength <= UInt64(maximumHeaderBytes),
              headerLength <= UInt64(Int.max)
        else {
            try handle.close()
            throw NormalizationError.malformedHeader(file.lastPathComponent)
        }

        let header = try handle.read(upToCount: Int(headerLength)) ?? Data()
        try handle.close()
        guard header.count == Int(headerLength),
              let normalized = try normalizedHeaderJSON(header)
        else { return false }
        guard normalized.count <= header.count else {
            throw NormalizationError.enlargedHeader(file.lastPathComponent)
        }

        try preserveOriginal(lengthData + header, named: file.lastPathComponent + ".header", beside: file)

        var padded = normalized
        padded.append(Data(repeating: 0x20, count: header.count - normalized.count))
        let writer = try FileHandle(forWritingTo: file)
        try writer.seek(toOffset: 8)
        try writer.write(contentsOf: padded)
        try writer.synchronize()
        try writer.close()
        return true
    }

    private static func normalizeIndexIfPresent(in directory: URL) throws {
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        guard FileManager.default.fileExists(atPath: indexURL.path) else { return }
        let original = try Data(contentsOf: indexURL)
        guard let normalized = try normalizedIndexJSON(original) else { return }
        guard normalized.count <= original.count else {
            throw NormalizationError.enlargedHeader(indexURL.lastPathComponent)
        }

        try preserveOriginal(original, named: indexURL.lastPathComponent, beside: indexURL)
        var padded = normalized
        padded.append(Data(repeating: 0x20, count: original.count - normalized.count))
        let writer = try FileHandle(forWritingTo: indexURL)
        try writer.seek(toOffset: 0)
        try writer.write(contentsOf: padded)
        try writer.synchronize()
        try writer.close()
    }

    private static func normalizeConfigIfNeeded(_ original: Data, at configURL: URL) throws {
        guard let normalized = try normalizedConfigJSON(original) else { return }
        try preserveOriginal(original, named: configURL.lastPathComponent, beside: configURL)

        let writer = try FileHandle(forWritingTo: configURL)
        try writer.seek(toOffset: 0)
        try writer.write(contentsOf: normalized)
        try writer.truncate(atOffset: UInt64(normalized.count))
        try writer.synchronize()
        try writer.close()
    }

    private static func normalizeKeys(in dictionary: inout [String: Any]) throws -> Bool {
        var changed = false
        for key in Array(dictionary.keys) where key.hasPrefix(wrappedPrefix) {
            let normalizedKey = String(key.dropFirst(wrappedPrefix.count))
            guard dictionary[normalizedKey] == nil else {
                throw NormalizationError.keyCollision(normalizedKey)
            }
            guard let value = dictionary.removeValue(forKey: key) else { continue }
            dictionary[normalizedKey] = value
            changed = true
        }
        return changed
    }

    private static func preserveOriginal(_ data: Data, named name: String, beside file: URL) throws {
        let directory = file.deletingLastPathComponent()
            .appendingPathComponent(backupDirectoryName, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let backup = directory.appendingPathComponent(name)
        guard !FileManager.default.fileExists(atPath: backup.path) else { return }
        try data.write(to: backup, options: .withoutOverwriting)
    }
}

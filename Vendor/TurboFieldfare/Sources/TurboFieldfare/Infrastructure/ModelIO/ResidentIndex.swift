import Foundation
import Darwin
import TurboFieldfareFormat

/// On-disk header. `indexSize` is the full byte size of the leading index
/// region — it INCLUDES the 24-byte header itself, the entry table, the
/// string table, and the writer's 16 KB page padding. The resident tensor
/// region starts at file byte `indexSize`.
struct ResidentIndexHeader: Sendable, Equatable {
    let indexSize: UInt64
    let residentSize: UInt64
    let entryCount: UInt64
}

public struct ResidentIndexEntry: Sendable, Equatable {
    public let name: String
    public let dtype: UInt8
    /// Absolute file offset of the packed weight bytes (≥ `indexSize`).
    public let fileOffset: UInt64
    public let sizeBytes: UInt64
    public let shape: (UInt32, UInt32, UInt32, UInt32)
    public let scaleOffset: UInt64
    public let scaleSize: UInt64
    public let biasOffset: UInt64
    public let biasSize: UInt64

    public static func == (a: ResidentIndexEntry, b: ResidentIndexEntry) -> Bool {
        a.name == b.name && a.dtype == b.dtype
            && a.fileOffset == b.fileOffset && a.sizeBytes == b.sizeBytes
            && a.shape.0 == b.shape.0 && a.shape.1 == b.shape.1
            && a.shape.2 == b.shape.2 && a.shape.3 == b.shape.3
            && a.scaleOffset == b.scaleOffset && a.scaleSize == b.scaleSize
            && a.biasOffset == b.biasOffset && a.biasSize == b.biasSize
    }
}

/// Parsed resident index: header + name-keyed entries.
struct ResidentIndex: Sendable {
    let header: ResidentIndexHeader
    let entries: [String: ResidentIndexEntry]
}

enum ResidentIndexReader {
    static let headerBytes = GTurboFormatV1.residentHeaderBytes
    static let entryBytes = GTurboFormatV1.residentEntryBytes
    static let defaultMaxBytes = GTurboFormatV1.residentIndexMaxBytes

    /// `pread` the header + index region out of `model_weights.bin`. The
    /// tensor data region (starting at byte `header.indexSize`) is **not**
    /// read here — that's the resident-buffer materialization job.
    static func load(fileURL: URL,
                            maxBytes: UInt64 = defaultMaxBytes) throws -> ResidentIndex {
        let fd = open(fileURL.path, O_RDONLY | O_NONBLOCK | O_CLOEXEC)
        guard fd >= 0 else {
            throw ModelError.posixFailed(call: "open(\(fileURL.path))", errno: errno)
        }
        defer { close(fd) }
        var info = stat()
        guard fstat(fd, &info) == 0 else {
            throw ModelError.posixFailed(call: "fstat(\(fileURL.path))", errno: errno)
        }
        guard (info.st_mode & S_IFMT) == S_IFREG else {
            throw ModelError.indexCorrupt(detail: "resident index is not a regular file")
        }

        return try load(fileDescriptor: fd, displayPath: fileURL.path,
                        maxBytes: maxBytes)
    }

    package static func load(fileDescriptor fd: Int32,
                             displayPath: String,
                             maxBytes: UInt64 = defaultMaxBytes) throws -> ResidentIndex {
        var headerBuf = [UInt8](repeating: 0, count: headerBytes)
        try headerBuf.withUnsafeMutableBytes {
            try preadExactly(fd: fd, into: $0, offset: 0,
                             field: "IndexHeader")
        }
        let regionLen: Int
        let wireHeader: GTurboResidentIndexHeaderV1?
        do {
            let decoded = try headerBuf.withUnsafeBytes {
                try GTurboResidentIndexCodec.decodeHeader($0)
            }
            guard decoded.indexSize <= maxBytes else {
                throw ModelError.indexCorrupt(
                    detail: "indexSize \(decoded.indexSize) exceeds metadata cap \(maxBytes)")
            }
            guard decoded.indexSize >= UInt64(headerBytes) else {
                throw ModelError.indexCorrupt(detail: "indexSize is smaller than the header")
            }
            var st = stat()
            guard fstat(fd, &st) == 0, st.st_size >= 0 else {
                throw ModelError.posixFailed(call: "fstat(\(displayPath))", errno: errno)
            }
            guard decoded.indexSize <= UInt64(st.st_size),
                  decoded.indexSize <= UInt64(Int.max) else {
                throw ModelError.indexCorrupt(detail: "index region exceeds file size")
            }
            regionLen = Int(decoded.indexSize)
            wireHeader = decoded
        } catch let error as ModelError {
            throw error
        } catch {
            // Dual-read: fall back to the pre-#87 header walk.
            let header = headerBuf.withUnsafeBytes { raw -> ResidentIndexHeader in
                let base = raw.baseAddress!
                return ResidentIndexHeader(
                    indexSize:    decodeU64LE(base, 0),
                    residentSize: decodeU64LE(base, 8),
                    entryCount:   decodeU64LE(base, 16))
            }
            guard header.indexSize >= UInt64(headerBytes) else {
                throw ModelError.indexCorrupt(
                    detail: "indexSize \(header.indexSize) < header size \(headerBytes)")
            }
            guard header.indexSize <= maxBytes else {
                throw ModelError.indexCorrupt(
                    detail: "indexSize \(header.indexSize) exceeds metadata cap \(maxBytes)")
            }
            regionLen = Int(header.indexSize)
            wireHeader = nil
            return try loadLegacyRegion(
                fd: fd, headerBuf: headerBuf, header: header, regionLen: regionLen)
        }

        var indexBuf = [UInt8](repeating: 0, count: regionLen)
        indexBuf.replaceSubrange(0..<headerBytes, with: headerBuf)
        try indexBuf.withUnsafeMutableBytes { raw in
            try preadExactly(
                fd: fd,
                into: UnsafeMutableRawBufferPointer(
                    rebasing: raw[headerBytes..<regionLen]),
                offset: off_t(headerBytes),
                field: "index region")
        }

        if let wireHeader {
            do {
                let wireEntries = try indexBuf.withUnsafeBytes {
                    try GTurboResidentIndexCodec.decodeRegion($0, header: wireHeader)
                }
                let header = ResidentIndexHeader(indexSize: wireHeader.indexSize,
                                                 residentSize: wireHeader.residentSize,
                                                 entryCount: wireHeader.entryCount)
                let entries = Dictionary(uniqueKeysWithValues: wireEntries.map { wire in
                    let shape = wire.shape
                    return (wire.name, ResidentIndexEntry(
                        name: wire.name, dtype: wire.dtype,
                        fileOffset: wire.fileOffset, sizeBytes: wire.sizeBytes,
                        shape: (shape[0], shape[1], shape[2], shape[3]),
                        scaleOffset: wire.scaleOffset, scaleSize: wire.scaleSize,
                        biasOffset: wire.biasOffset, biasSize: wire.biasSize))
                })
                return ResidentIndex(header: header, entries: entries)
            } catch {
                let header = ResidentIndexHeader(indexSize: wireHeader.indexSize,
                                                 residentSize: wireHeader.residentSize,
                                                 entryCount: wireHeader.entryCount)
                return try parseLegacyEntries(indexBuf: indexBuf, header: header,
                                              regionLen: regionLen)
            }
        }
        throw ModelError.indexCorrupt(detail: "resident index decode failed")
    }

    private static func loadLegacyRegion(
        fd: Int32,
        headerBuf: [UInt8],
        header: ResidentIndexHeader,
        regionLen: Int
    ) throws -> ResidentIndex {
        let expectedEntryTableEnd = UInt64(headerBytes) + UInt64(header.entryCount) * UInt64(entryBytes)
        guard expectedEntryTableEnd <= header.indexSize else {
            throw ModelError.indexCorrupt(
                detail: "header+entries (\(expectedEntryTableEnd)) > indexSize \(header.indexSize)")
        }
        var indexBuf = [UInt8](repeating: 0, count: regionLen)
        indexBuf.replaceSubrange(0..<headerBytes, with: headerBuf)
        try indexBuf.withUnsafeMutableBytes { raw in
            try preadExactly(
                fd: fd,
                into: UnsafeMutableRawBufferPointer(
                    rebasing: raw[headerBytes..<regionLen]),
                offset: off_t(headerBytes),
                field: "index region")
        }
        return try parseLegacyEntries(indexBuf: indexBuf, header: header, regionLen: regionLen)
    }

    private static func parseLegacyEntries(
        indexBuf: [UInt8],
        header: ResidentIndexHeader,
        regionLen: Int
    ) throws -> ResidentIndex {
        var entries: [String: ResidentIndexEntry] = [:]
        entries.reserveCapacity(Int(header.entryCount))
        try indexBuf.withUnsafeBytes { raw in
            let base = raw.baseAddress!
            for i in 0..<Int(header.entryCount) {
                let p = base.advanced(by: headerBytes + i * entryBytes)
                let nameOffset = Int(decodeU32LE(p, 0))
                let nameLength = Int(decodeU16LE(p, 4))
                let dtype = decodeU8(p, 6)
                let fileOffset = decodeU64LE(p, 8)
                let sizeBytes  = decodeU64LE(p, 16)
                let s0 = decodeU32LE(p, 24)
                let s1 = decodeU32LE(p, 28)
                let s2 = decodeU32LE(p, 32)
                let s3 = decodeU32LE(p, 36)
                let scaleOffset = decodeU64LE(p, 40)
                let scaleSize   = decodeU64LE(p, 48)
                let biasOffset  = decodeU64LE(p, 56)
                let biasSize    = decodeU64LE(p, 64)

                guard nameOffset >= headerBytes,
                      nameOffset + nameLength <= regionLen else {
                    throw ModelError.indexCorrupt(
                        detail: "entry \(i) name range [\(nameOffset), \(nameOffset + nameLength)) " +
                                "out of index region [\(headerBytes), \(regionLen))")
                }
                let namePtr = base.advanced(by: nameOffset)
                let name = String(decoding: UnsafeRawBufferPointer(
                    start: namePtr, count: nameLength), as: UTF8.self)

                let entry = ResidentIndexEntry(
                    name: name, dtype: dtype,
                    fileOffset: fileOffset, sizeBytes: sizeBytes,
                    shape: (s0, s1, s2, s3),
                    scaleOffset: scaleOffset, scaleSize: scaleSize,
                    biasOffset: biasOffset, biasSize: biasSize)
                if entries[name] != nil {
                    throw ModelError.indexCorrupt(detail: "duplicate tensor name \(name)")
                }
                entries[name] = entry
            }
        }
        return ResidentIndex(header: header, entries: entries)
    }

    private static func preadExactly(fd: Int32,
                                     into buffer: UnsafeMutableRawBufferPointer,
                                     offset: off_t,
                                     field: String) throws {
        var total = 0
        while total < buffer.count {
            let count = pread(fd, buffer.baseAddress!.advanced(by: total),
                              buffer.count - total, offset + off_t(total))
            if count < 0, errno == EINTR { continue }
            guard count > 0 else {
                throw ModelError.indexCorrupt(
                    detail: "short read for \(field) (\(total)/\(buffer.count))")
            }
            total += count
        }
    }

    @inline(__always)
    static func decodeU64LE(_ base: UnsafeRawPointer, _ off: Int) -> UInt64 {
        let p = base.advanced(by: off).assumingMemoryBound(to: UInt8.self)
        var v: UInt64 = 0
        for i in (0..<8).reversed() { v = (v << 8) | UInt64(p[i]) }
        return v
    }

    @inline(__always)
    static func decodeU32LE(_ base: UnsafeRawPointer, _ off: Int) -> UInt32 {
        let p = base.advanced(by: off).assumingMemoryBound(to: UInt8.self)
        var v: UInt32 = 0
        for i in (0..<4).reversed() { v = (v << 8) | UInt32(p[i]) }
        return v
    }

    @inline(__always)
    static func decodeU16LE(_ base: UnsafeRawPointer, _ off: Int) -> UInt16 {
        let p = base.advanced(by: off).assumingMemoryBound(to: UInt8.self)
        return UInt16(p[0]) | (UInt16(p[1]) << 8)
    }

    @inline(__always)
    static func decodeU8(_ base: UnsafeRawPointer, _ off: Int) -> UInt8 {
        base.advanced(by: off).assumingMemoryBound(to: UInt8.self)[0]
    }
}

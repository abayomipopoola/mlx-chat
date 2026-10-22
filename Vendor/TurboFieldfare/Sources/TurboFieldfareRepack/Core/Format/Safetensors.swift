import Darwin
import Foundation

/// Parses bounded safetensors header bytes fetched by the remote installer.
/// Tensor payload coordinates remain absolute source-file offsets so later
/// range requests can copy only the required tiles.
enum Safetensors {
    static let maxHeaderBytes: UInt64 = 1 << 24  // 16 MB — generous; observed ~95 KB

    struct Header {
        let tensors: [SourceTensor]

        init(tensors: [SourceTensor]) {
            self.tensors = tensors
        }
    }

    static func parseHeader(path: String) throws -> Header {
        let fd = try Posix.openRead(path)
        defer { close(fd) }
        var st = stat()
        if fstat(fd, &st) != 0 {
            throw RepackError.fileStatFailed(path: path, errno: errno)
        }
        let fileSize = UInt64(st.st_size)
        if fileSize < 8 {
            throw RepackError.safetensorsHeaderInvalid(path: path, detail: "file too short")
        }

        var headerSizeLE: UInt64 = 0
        try withUnsafeMutableBytes(of: &headerSizeLE) { raw in
            try Posix.preadAll(fd: fd, path: path, buf: raw.baseAddress!, count: 8, offset: 0)
        }
        let headerSize = UInt64(littleEndian: headerSizeLE)
        if headerSize > maxHeaderBytes || headerSize > fileSize - 8 {
            throw RepackError.safetensorsHeaderTooLarge(path: path, size: headerSize)
        }

        let buf = UnsafeMutableRawBufferPointer.allocate(byteCount: Int(headerSize), alignment: 16)
        defer { buf.deallocate() }
        try Posix.preadAll(fd: fd, path: path, buf: buf.baseAddress!, count: Int(headerSize), offset: 8)
        let data = Data(bytesNoCopy: buf.baseAddress!, count: Int(headerSize), deallocator: .none)
        return try parseHeaderBytes(path: path, fileSize: fileSize, headerBytes: data)
    }

    static func parseHeaderBytes(path: String,
                                        fileSize: UInt64,
                                        headerBytes data: Data) throws -> Header {
        let headerSize = UInt64(data.count)
        if headerSize > maxHeaderBytes || headerSize > fileSize - 8 {
            throw RepackError.safetensorsHeaderTooLarge(path: path, size: headerSize)
        }
        let rawObj = try JSONSerialization.jsonObject(with: data, options: [])
        guard let dict = rawObj as? [String: Any] else {
            throw RepackError.safetensorsHeaderInvalid(path: path, detail: "header is not a JSON object")
        }

        let payloadBase = UInt64(8) + headerSize
        var tensors: [SourceTensor] = []
        tensors.reserveCapacity(dict.count)
        for (name, value) in dict {
            if name == "__metadata__" { continue }
            guard let entry = value as? [String: Any] else {
                throw RepackError.safetensorsHeaderInvalid(path: path,
                                                           detail: "entry for \(name) is not a dict")
            }
            guard let dtypeStr = entry["dtype"] as? String else {
                throw RepackError.safetensorsHeaderInvalid(path: path,
                                                           detail: "entry for \(name) has no dtype")
            }
            var dtype: SourceTensor.Dtype
            let storedAsBytes: Bool
            switch dtypeStr {
            case "U32":  dtype = .u32; storedAsBytes = false
            case "U8":   dtype = .u32; storedAsBytes = true
            case "BF16": dtype = .bf16; storedAsBytes = false
            case "F16":  dtype = .fp16; storedAsBytes = false
            case "F32":  dtype = .fp32; storedAsBytes = false
            default: throw RepackError.safetensorsUnknownDtype(path: path, dtype: dtypeStr)
            }
            guard let shape = entry["shape"] as? [Any] else {
                throw RepackError.safetensorsHeaderInvalid(path: path,
                                                           detail: "entry for \(name) has no shape")
            }
            var shapeU64: [UInt64] = try shape.map { e in
                if let n = e as? NSNumber { return n.uint64Value }
                if let n = e as? Int { return UInt64(n) }
                throw RepackError.safetensorsHeaderInvalid(path: path,
                                                           detail: "entry for \(name) has non-integer shape entry")
            }
            guard let offs = entry["data_offsets"] as? [Any], offs.count == 2,
                  let begin = (offs[0] as? NSNumber)?.uint64Value ?? (offs[0] as? Int).map({ UInt64($0) }),
                  let end   = (offs[1] as? NSNumber)?.uint64Value ?? (offs[1] as? Int).map({ UInt64($0) })
            else {
                throw RepackError.safetensorsHeaderInvalid(path: path,
                                                           detail: "entry for \(name) has bad data_offsets")
            }
            let abs = payloadBase + begin
            let size = end - begin
            let endAbs = abs + size
            if endAbs > fileSize {
                throw RepackError.safetensorsTensorOutOfRange(path: path, name: name,
                                                              end: endAbs, fileSize: fileSize)
            }
            // MLX writes affine weights as raw bytes (U8) of the little-endian
            // uint32 packing. Fold each 4 bytes back into one U32 so the
            // planner's word-shaped logical dims stay correct. The byte
            // check uses the stored width, before that fold.
            let storedBytes: UInt64 = storedAsBytes ? 1 : UInt64(dtype.elementBytes)
            let storedElements = shapeU64.reduce(UInt64(1), *)
            if storedElements * storedBytes != size {
                throw RepackError.shapeMismatch(name: name,
                                                detail: "shape product \(storedElements)*\(storedBytes) != size \(size)")
            }
            if storedAsBytes {
                guard name.hasSuffix(".weight"),
                      let last = shapeU64.last, last % 4 == 0, !shapeU64.isEmpty else {
                    throw RepackError.safetensorsHeaderInvalid(
                        path: path,
                        detail: "U8 tensor \(name) is not a packed weight row")
                }
                shapeU64[shapeU64.count - 1] = last / 4
            }
            let elemBytes = UInt64(dtype.elementBytes)
            let elements = shapeU64.reduce(UInt64(1), *)
            if elements * elemBytes != size {
                throw RepackError.shapeMismatch(name: name,
                                                detail: "shape product \(elements)*\(elemBytes) != size \(size)")
            }
            tensors.append(SourceTensor(name: name, shardPath: path, dtype: dtype,
                                        shape: shapeU64,
                                        absoluteOffset: abs, sizeBytes: size))
        }
        return Header(tensors: tensors)
    }
}

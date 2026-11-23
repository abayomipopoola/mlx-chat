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

    private static func strictUInt64(_ raw: Any) -> UInt64? {
        if let number = raw as? NSNumber {
            if CFGetTypeID(number) == CFBooleanGetTypeID() {
                return nil
            }
            let objCType = String(cString: number.objCType)
            if objCType == "f" || objCType == "d" {
                let d = number.doubleValue
                guard d.isFinite, d >= 0, d.rounded(.towardZero) == d,
                      d < Double(UInt64.max) else {
                    return nil
                }
                return UInt64(d)
            }
            if number.int64Value < 0 {
                // Distinguish negative signed integers from unsigned > Int64.max.
                if objCType == "Q" || objCType == "L" || objCType == "I"
                    || objCType == "S" || objCType == "C" {
                    return number.uint64Value
                }
                return nil
            }
            return number.uint64Value
        }
        if let intVal = raw as? Int {
            guard intVal >= 0 else { return nil }
            return UInt64(intVal)
        }
        return nil
    }

    static func parseHeaderBytes(path: String,
                                        fileSize: UInt64,
                                        headerBytes data: Data) throws -> Header {
        let headerSize = UInt64(data.count)
        guard fileSize >= 8 else {
            throw RepackError.safetensorsHeaderTooLarge(path: path, size: headerSize)
        }
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
                guard let dim = strictUInt64(e) else {
                    throw RepackError.safetensorsHeaderInvalid(path: path,
                                                               detail: "entry for \(name) has invalid shape entry")
                }
                return dim
            }
            guard let offs = entry["data_offsets"] as? [Any], offs.count == 2,
                  let begin = strictUInt64(offs[0]),
                  let end   = strictUInt64(offs[1]),
                  end >= begin
            else {
                throw RepackError.safetensorsHeaderInvalid(path: path,
                                                           detail: "entry for \(name) has bad data_offsets")
            }
            let (abs, absOverflow) = payloadBase.addingReportingOverflow(begin)
            let size = end - begin
            let (endAbs, endOverflow) = abs.addingReportingOverflow(size)
            if absOverflow || endOverflow || endAbs > fileSize {
                throw RepackError.safetensorsTensorOutOfRange(path: path, name: name,
                                                              end: endOverflow ? UInt64.max : endAbs,
                                                              fileSize: fileSize)
            }
            // MLX writes affine weights as raw bytes (U8) of the little-endian
            // uint32 packing. Fold each 4 bytes back into one U32 so the
            // planner's word-shaped logical dims stay correct. The byte
            // check uses the stored width, before that fold.
            let storedBytes: UInt64 = storedAsBytes ? 1 : UInt64(dtype.elementBytes)
            var storedElements: UInt64 = 1
            for dim in shapeU64 {
                let (next, overflow) = storedElements.multipliedReportingOverflow(by: dim)
                if overflow {
                    throw RepackError.safetensorsHeaderInvalid(path: path,
                                                               detail: "shape product overflows UInt64 for \(name)")
                }
                storedElements = next
            }
            let (expectedBytes, byteOverflow) = storedElements.multipliedReportingOverflow(by: storedBytes)
            if byteOverflow || expectedBytes != size {
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
            var elements: UInt64 = 1
            for dim in shapeU64 {
                let (next, overflow) = elements.multipliedReportingOverflow(by: dim)
                if overflow {
                    throw RepackError.safetensorsHeaderInvalid(path: path,
                                                               detail: "logical shape product overflows UInt64 for \(name)")
                }
                elements = next
            }
            let (logicalBytes, logicalOverflow) = elements.multipliedReportingOverflow(by: elemBytes)
            if logicalOverflow || logicalBytes != size {
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

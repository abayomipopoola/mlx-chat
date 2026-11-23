import Darwin
import Foundation
import MLX
import MLXNN

// MARK: - Streaming configuration, stats, and errors

public enum Inkling2EvictionPolicy: String, Sendable, Codable {
    case sharedLRU
    case sharedSLRU
    case perPassLRU
}

public struct Inkling2StreamingConfig: Sendable, Equatable {
    /// Locked empirically from Check (a) on `mlx-community/Inkling-2-Mini-4bit`:
    /// shared SLRU wins every feasible budget, loop-weighted 2x core allocation cuts slot RAM
    /// without losing hit rate, and 512-token chunks halve prefill SSD re-reads.
    public static let defaultEvictionPolicy: Inkling2EvictionPolicy = .sharedSLRU
    public static let defaultLoopWeightedSlots: Bool = true
    public static let defaultPrefillChunkSize: Int = 512
    public static let defaultMaxConcurrentReads: Int = 8
    public static let defaultBypassPageCache: Bool = false

    public let modelDirectory: URL
    public let expertCacheSlots: Int
    public let evictionPolicy: Inkling2EvictionPolicy
    public let prefillChunkSize: Int
    public let maxConcurrentReads: Int
    public let loopWeightedSlots: Bool
    public let bypassPageCache: Bool

    public init(
        modelDirectory: URL,
        expertCacheSlots: Int,
        evictionPolicy: Inkling2EvictionPolicy = Self.defaultEvictionPolicy,
        prefillChunkSize: Int = Self.defaultPrefillChunkSize,
        maxConcurrentReads: Int = Self.defaultMaxConcurrentReads,
        loopWeightedSlots: Bool = Self.defaultLoopWeightedSlots,
        bypassPageCache: Bool = Self.defaultBypassPageCache
    ) {
        self.modelDirectory = modelDirectory.standardizedFileURL
        self.expertCacheSlots = expertCacheSlots
        self.evictionPolicy = evictionPolicy
        self.prefillChunkSize = max(1, prefillChunkSize)
        self.maxConcurrentReads = max(1, maxConcurrentReads)
        self.loopWeightedSlots = loopWeightedSlots
        self.bypassPageCache = bypassPageCache
    }

    public static func slotsPerLayer(
        textConfig: Inkling2TextConfiguration,
        budgetSlots: Int,
        loopWeighted: Bool
    ) -> [Int] {
        let minSlots = max(1, textConfig.expertsPerToken)
        let coreRange = textConfig.loop?.core
        return (0 ..< textConfig.hiddenLayers).map { layer in
            let isCore = coreRange?.contains(layer) ?? false
            let raw = (loopWeighted && coreRange != nil && !isCore) ? (budgetSlots / 2) : budgetSlots
            return max(minSlots, min(raw, textConfig.experts))
        }
    }
}

public struct Inkling2StreamingStats: Sendable, Equatable {
    public let totalRequests: Int
    public let cacheHits: Int
    public let cacheMisses: Int
    public let bytesRead: UInt64
    public let alignedBytesRead: UInt64
    public let ioNanoseconds: UInt64
    public let residentSlotBytes: UInt64

    public var hitRate: Double {
        totalRequests > 0 ? Double(cacheHits) / Double(totalRequests) : 0
    }
}

public enum Inkling2StreamingError: LocalizedError, Equatable {
    case invalidSafetensorsHeader(String)
    case missingExpertTensor(String)
    case nonContiguousExperts(String)
    case zeroCopyAliasingFailed
    case invalidConfiguration(String)
    case shortRead(String)
    case ioError(path: String, errno: Int32)
    case unexpectedResidentExperts(Int)

    public var errorDescription: String? {
        switch self {
        case .invalidSafetensorsHeader(let detail):
            return "Invalid safetensors header: \(detail)"
        case .missingExpertTensor(let key):
            return "Missing expert tensor in checkpoint: \(key)"
        case .nonContiguousExperts(let key):
            return "Expert tensor is not contiguous along axis 0: \(key)"
        case .zeroCopyAliasingFailed:
            return "Metal zero-copy buffer aliasing check failed."
        case .invalidConfiguration(let detail):
            return "Invalid SSD streaming configuration: \(detail)"
        case .shortRead(let detail):
            return "Unexpected EOF while reading expert weights from SSD: \(detail)"
        case .ioError(let path, let code):
            return "SSD read failed for \(path) (errno \(code)): \(String(cString: strerror(code)))"
        case .unexpectedResidentExperts(let count):
            return "Fail-closed guard: \(count) resident expert tensor(s) bypassed excludedCheckpointNamespaces."
        }
    }
}

// MARK: - Safetensors Header-Only Expert Index

struct Inkling2TensorSliceMeta: Sendable {
    let shardIndex: Int
    let baseFileOffset: UInt64
    let perExpertBytes: UInt64
}

struct Inkling2ProjectionFileMeta: Sendable {
    let format: Inkling2ExpertFormat
    let outputDims: Int
    let packedCols: Int
    let groupCols: Int
    let weight: Inkling2TensorSliceMeta
    let scales: Inkling2TensorSliceMeta
    let biases: Inkling2TensorSliceMeta?
    let fullGlobalScale: [Float]?
}

struct Inkling2LayerFileMeta: Sendable {
    let gateUp: Inkling2ProjectionFileMeta
    let down: Inkling2ProjectionFileMeta
}

struct Inkling2SafetensorsExpertIndex {
    let shardURLs: [URL]
    let layers: [Inkling2LayerFileMeta]
    let maxAlignedSliceBytes: Int

    private struct RawTensorMeta {
        let shardIndex: Int
        let dataStart: UInt64
        let byteOffset: UInt64
        let byteLength: UInt64
        let dtype: String
        let shape: [Int]
    }

    static func parse(
        modelDirectory: URL,
        textConfig: Inkling2TextConfiguration
    ) throws -> Inkling2SafetensorsExpertIndex {
        let fm = FileManager.default
        let indexURL = modelDirectory.appendingPathComponent("model.safetensors.index.json")
        let shardNames: [String]
        if fm.fileExists(atPath: indexURL.path) {
            struct WeightIndex: Decodable {
                let weight_map: [String: String]
            }
            let data = try Data(contentsOf: indexURL)
            let decoded = try JSONDecoder().decode(WeightIndex.self, from: data)
            shardNames = Array(Set(decoded.weight_map.values)).sorted()
        } else {
            shardNames = try fm.contentsOfDirectory(atPath: modelDirectory.path)
                .filter { $0.hasSuffix(".safetensors") }
                .sorted()
        }
        guard !shardNames.isEmpty else {
            throw Inkling2StreamingError.invalidSafetensorsHeader(
                "No .safetensors files in \(modelDirectory.path)")
        }

        let shardURLs = shardNames.map { modelDirectory.appendingPathComponent($0) }
        var tensors: [String: RawTensorMeta] = [:]

        for (shardIndex, url) in shardURLs.enumerated() {
            let fh = try FileHandle(forReadingFrom: url)
            defer { try? fh.close() }
            guard let lenData = try fh.read(upToCount: 8), lenData.count == 8 else {
                throw Inkling2StreamingError.invalidSafetensorsHeader(
                    "Truncated header in \(url.lastPathComponent)")
            }
            let headerLen = lenData.withUnsafeBytes {
                $0.loadUnaligned(as: UInt64.self)
            }.littleEndian
            guard headerLen > 0, headerLen <= 512 * 1024 * 1024,
                  let headerData = try fh.read(upToCount: Int(headerLen)),
                  headerData.count == Int(headerLen),
                  let json = try JSONSerialization.jsonObject(with: headerData) as? [String: Any]
            else {
                throw Inkling2StreamingError.invalidSafetensorsHeader(
                    "Malformed header JSON in \(url.lastPathComponent)")
            }
            let dataStart = 8 + headerLen
            for (key, value) in json where key != "__metadata__" {
                guard let dict = value as? [String: Any],
                      let dtype = dict["dtype"] as? String,
                      let shape = dict["shape"] as? [Int],
                      let offsets = dict["data_offsets"] as? [NSNumber],
                      offsets.count == 2
                else { continue }
                let begin = offsets[0].uint64Value
                let end = offsets[1].uint64Value
                guard end >= begin else {
                    throw Inkling2StreamingError.invalidSafetensorsHeader(
                        "Negative offset span for \(key)")
                }
                tensors[key] = RawTensorMeta(
                    shardIndex: shardIndex,
                    dataStart: dataStart,
                    byteOffset: begin,
                    byteLength: end - begin,
                    dtype: dtype,
                    shape: shape)
            }
        }

        var shardHandles: [FileHandle] = []
        defer { for fh in shardHandles { try? fh.close() } }
        shardHandles.reserveCapacity(shardURLs.count)
        for url in shardURLs {
            shardHandles.append(try FileHandle(forReadingFrom: url))
        }

        func lookup(_ layer: Int, _ proj: String, _ tensorName: String) -> RawTensorMeta? {
            let suffix = "\(proj).\(tensorName)"
            return tensors["model.language_model.layers.\(layer).mlp.experts.\(suffix)"]
                ?? tensors["language_model.layers.\(layer).mlp.experts.\(suffix)"]
        }

        func readFloat32Array(_ meta: RawTensorMeta, expectedCount: Int) throws -> [Float] {
            guard meta.dtype == "F32",
                  meta.shape == [expectedCount],
                  meta.byteLength == UInt64(expectedCount * MemoryLayout<Float>.size)
            else {
                throw Inkling2StreamingError.invalidSafetensorsHeader(
                    "Unexpected global_scale shape/dtype \(meta.dtype) \(meta.shape)")
            }
            let handle = shardHandles[meta.shardIndex]
            try handle.seek(toOffset: meta.dataStart + meta.byteOffset)
            guard let bytes = try handle.read(upToCount: Int(meta.byteLength)),
                  bytes.count == Int(meta.byteLength)
            else {
                throw Inkling2StreamingError.shortRead("global_scale")
            }
            var floats = [Float](repeating: 0, count: expectedCount)
            floats.withUnsafeMutableBytes { dst in
                _ = bytes.copyBytes(to: dst)
            }
            return floats
        }

        var maxAlignedSlice = 16384
        var layerMetas: [Inkling2LayerFileMeta] = []
        layerMetas.reserveCapacity(textConfig.hiddenLayers)

        func parseProjection(
            layer: Int,
            proj: String,
            inputDims: Int,
            outputDims: Int
        ) throws -> Inkling2ProjectionFileMeta {
            guard let weightMeta = lookup(layer, proj, "weight") else {
                throw Inkling2StreamingError.missingExpertTensor(
                    "layers.\(layer).mlp.experts.\(proj).weight")
            }
            guard let scalesMeta = lookup(layer, proj, "scales") else {
                throw Inkling2StreamingError.missingExpertTensor(
                    "layers.\(layer).mlp.experts.\(proj).scales")
            }
            let biasesMeta = lookup(layer, proj, "biases")
            let globalScaleMeta = lookup(layer, proj, "global_scale")

            guard weightMeta.dtype == "U32",
                  weightMeta.shape.count == 3,
                  weightMeta.shape[0] == textConfig.experts,
                  weightMeta.shape[1] == outputDims,
                  weightMeta.byteLength % UInt64(textConfig.experts) == 0
            else {
                throw Inkling2StreamingError.nonContiguousExperts(
                    "layers.\(layer).mlp.experts.\(proj).weight (\(weightMeta.dtype) \(weightMeta.shape))")
            }

            let packedCols = weightMeta.shape[2]
            let expectedWeightBytes = UInt64(textConfig.experts) * UInt64(outputDims) * UInt64(packedCols) * 4
            guard weightMeta.byteLength == expectedWeightBytes,
                  packedCols > 0,
                  (packedCols * 32) % inputDims == 0
            else {
                throw Inkling2StreamingError.nonContiguousExperts(
                    "layers.\(layer).mlp.experts.\(proj).weight byte length mismatch")
            }
            let bits = packedCols * 32 / inputDims

            guard scalesMeta.shape.count == 3,
                  scalesMeta.shape[0] == textConfig.experts,
                  scalesMeta.shape[1] == outputDims,
                  scalesMeta.byteLength % UInt64(textConfig.experts) == 0
            else {
                throw Inkling2StreamingError.nonContiguousExperts(
                    "layers.\(layer).mlp.experts.\(proj).scales (\(scalesMeta.dtype) \(scalesMeta.shape))")
            }
            let groupCols = scalesMeta.shape[2]
            guard groupCols > 0, inputDims % groupCols == 0 else {
                throw Inkling2StreamingError.invalidSafetensorsHeader(
                    "Invalid group count \(groupCols) for inputDims \(inputDims)")
            }
            let groupSize = inputDims / groupCols

            let format: Inkling2ExpertFormat
            let scaleElemBytes: Int
            if let biasesMeta {
                guard scalesMeta.dtype == "BF16", biasesMeta.dtype == "BF16",
                      biasesMeta.shape == scalesMeta.shape,
                      biasesMeta.byteLength == UInt64(textConfig.experts * outputDims * groupCols * 2)
                else {
                    throw Inkling2StreamingError.nonContiguousExperts(
                        "layers.\(layer).mlp.experts.\(proj).biases (\(biasesMeta.dtype) \(biasesMeta.shape))")
                }
                scaleElemBytes = 2
                format = Inkling2ExpertFormat(
                    mode: .affine, bits: bits, groupSize: groupSize, hasGlobalScale: false)
            } else {
                guard scalesMeta.dtype == "U8" else {
                    throw Inkling2StreamingError.invalidSafetensorsHeader(
                        "Unexpected scales dtype \(scalesMeta.dtype) at layer \(layer)")
                }
                scaleElemBytes = 1
                let mode: QuantizationMode
                switch (bits, groupSize) {
                case (4, 16): mode = .nvfp4
                case (4, 32): mode = .mxfp4
                case (8, 32): mode = .mxfp8
                default:
                    throw Inkling2StreamingError.invalidConfiguration(
                        "Unsupported expert quant (bits=\(bits), groupSize=\(groupSize)) at layer \(layer)")
                }
                format = Inkling2ExpertFormat(
                    mode: mode,
                    bits: bits,
                    groupSize: groupSize,
                    hasGlobalScale: globalScaleMeta != nil)
            }

            let expectedScaleBytes = UInt64(textConfig.experts * outputDims * groupCols * scaleElemBytes)
            guard scalesMeta.byteLength == expectedScaleBytes else {
                throw Inkling2StreamingError.nonContiguousExperts(
                    "layers.\(layer).mlp.experts.\(proj).scales byte length mismatch")
            }

            let perExpertWeight = weightMeta.byteLength / UInt64(textConfig.experts)
            let perExpertScales = scalesMeta.byteLength / UInt64(textConfig.experts)
            let wSlice = Inkling2TensorSliceMeta(
                shardIndex: weightMeta.shardIndex,
                baseFileOffset: weightMeta.dataStart + weightMeta.byteOffset,
                perExpertBytes: perExpertWeight)
            let sSlice = Inkling2TensorSliceMeta(
                shardIndex: scalesMeta.shardIndex,
                baseFileOffset: scalesMeta.dataStart + scalesMeta.byteOffset,
                perExpertBytes: perExpertScales)
            let bSlice: Inkling2TensorSliceMeta?
            if let biasesMeta {
                let perExpertBiases = biasesMeta.byteLength / UInt64(textConfig.experts)
                bSlice = Inkling2TensorSliceMeta(
                    shardIndex: biasesMeta.shardIndex,
                    baseFileOffset: biasesMeta.dataStart + biasesMeta.byteOffset,
                    perExpertBytes: perExpertBiases)
                maxAlignedSlice = max(maxAlignedSlice, alignedUpperBound(Int(perExpertBiases)))
            } else {
                bSlice = nil
            }
            maxAlignedSlice = max(maxAlignedSlice, alignedUpperBound(Int(perExpertWeight)))
            maxAlignedSlice = max(maxAlignedSlice, alignedUpperBound(Int(perExpertScales)))

            let gsArray = try globalScaleMeta.map {
                try readFloat32Array($0, expectedCount: textConfig.experts)
            }

            return Inkling2ProjectionFileMeta(
                format: format,
                outputDims: outputDims,
                packedCols: packedCols,
                groupCols: groupCols,
                weight: wSlice,
                scales: sSlice,
                biases: bSlice,
                fullGlobalScale: gsArray)
        }

        for layer in 0 ..< textConfig.hiddenLayers {
            let gateUp = try parseProjection(
                layer: layer,
                proj: "gate_up_proj",
                inputDims: textConfig.hiddenSize,
                outputDims: 2 * textConfig.intermediateSize)
            let down = try parseProjection(
                layer: layer,
                proj: "down_proj",
                inputDims: textConfig.intermediateSize,
                outputDims: textConfig.hiddenSize)
            layerMetas.append(Inkling2LayerFileMeta(gateUp: gateUp, down: down))
        }

        return Inkling2SafetensorsExpertIndex(
            shardURLs: shardURLs,
            layers: layerMetas,
            maxAlignedSliceBytes: maxAlignedSlice)
    }

    private static func alignedUpperBound(_ byteCount: Int) -> Int {
        // Any unaligned offset can span at most 16 KB - 1 before and after.
        ((byteCount + 32767) & ~16383)
    }
}

// MARK: - Concurrent Coalescing Shard Reader

final class Inkling2AlignedShardReader: @unchecked Sendable {
    struct ReadTask {
        let shardIndex: Int
        let fileOffset: UInt64
        let byteCount: Int
        let destination: UnsafeMutableRawPointer
    }

    private let stateCondition = NSCondition()
    private var activeReaders = 0
    private var fds: [Int32]
    private let fileSizes: [UInt64]
    private let shardPaths: [String]
    private let bypassPageCache: Bool
    private let maxWorkers: Int
    private var bounceBuffers: [UnsafeMutableRawPointer]
    private let bounceCapacity: Int
    private var isClosed = false

    #if DEBUG
    /// Test hook: inject an I/O failure before any pread completes for a path containing this substring.
    var debugInjectFailureOnShardSubstring: String?
    #endif

    init(
        shardURLs: [URL],
        maxAlignedSliceBytes: Int,
        workerCount: Int,
        bypassPageCache: Bool = false
    ) throws {
        let alignment = 16384
        self.bypassPageCache = bypassPageCache
        self.maxWorkers = max(1, workerCount)
        self.bounceCapacity = bypassPageCache
            ? max(alignment, ((maxAlignedSliceBytes + alignment - 1) / alignment) * alignment)
            : 0
        self.shardPaths = shardURLs.map(\.path)

        var localFds = Array(repeating: Int32(-1), count: shardURLs.count)
        var measuredSizes = Array(repeating: UInt64(0), count: shardURLs.count)
        var localBuffers: [UnsafeMutableRawPointer] = []

        var cleanupOnThrow = true
        defer {
            if cleanupOnThrow {
                for fd in localFds where fd >= 0 {
                    Darwin.close(fd)
                }
                for ptr in localBuffers {
                    free(ptr)
                }
            }
        }

        for (index, url) in shardURLs.enumerated() {
            let fd = open(url.path, O_RDONLY)
            guard fd >= 0 else {
                throw Inkling2StreamingError.ioError(path: url.path, errno: errno)
            }
            localFds[index] = fd
            var st = stat()
            guard fstat(fd, &st) == 0, st.st_size >= 0 else {
                throw Inkling2StreamingError.ioError(path: url.path, errno: errno)
            }
            measuredSizes[index] = UInt64(st.st_size)
            if bypassPageCache {
                _ = fcntl(fd, F_NOCACHE, 1)
            } else {
                // Keep OS unified page cache enabled for warm re-reads, without speculative readahead.
                _ = fcntl(fd, F_RDAHEAD, 0)
            }
        }

        if bypassPageCache {
            for _ in 0 ..< maxWorkers {
                var ptr: UnsafeMutableRawPointer?
                let rc = posix_memalign(&ptr, alignment, bounceCapacity)
                guard rc == 0, let allocated = ptr else {
                    throw Inkling2StreamingError.ioError(path: "posix_memalign", errno: rc)
                }
                localBuffers.append(allocated)
            }
        }

        self.fds = localFds
        self.fileSizes = measuredSizes
        self.bounceBuffers = localBuffers
        cleanupOnThrow = false
    }

    deinit {
        close()
    }

    func close() {
        stateCondition.lock()
        guard !isClosed else {
            stateCondition.unlock()
            return
        }
        isClosed = true
        while activeReaders > 0 {
            stateCondition.wait()
        }
        for i in fds.indices where fds[i] >= 0 {
            Darwin.close(fds[i])
            fds[i] = -1
        }
        for ptr in bounceBuffers {
            free(ptr)
        }
        bounceBuffers.removeAll()
        stateCondition.unlock()
    }

    func readSlicesConcurrently(_ tasks: [ReadTask]) throws -> (rawBytes: UInt64, alignedBytes: UInt64) {
        guard !tasks.isEmpty else { return (0, 0) }
        let sorted = tasks.sorted {
            if $0.shardIndex != $1.shardIndex { return $0.shardIndex < $1.shardIndex }
            return $0.fileOffset < $1.fileOffset
        }

        stateCondition.lock()
        if isClosed {
            stateCondition.unlock()
            throw Inkling2StreamingError.ioError(path: "closed-reader", errno: EBADF)
        }
        activeReaders += 1
        let activeFds = fds
        let activeBuffers = bounceBuffers
        #if DEBUG
        let injectedFailure = debugInjectFailureOnShardSubstring
        #else
        let injectedFailure: String? = nil
        #endif
        stateCondition.unlock()

        defer {
            stateCondition.lock()
            activeReaders -= 1
            if activeReaders == 0 && isClosed {
                stateCondition.broadcast()
            }
            stateCondition.unlock()
        }

        guard !activeFds.isEmpty, (!bypassPageCache || !activeBuffers.isEmpty) else {
            throw Inkling2StreamingError.ioError(path: "closed-reader", errno: EBADF)
        }

        if let injectedFailure {
            for task in sorted where shardPaths[task.shardIndex].contains(injectedFailure) {
                throw Inkling2StreamingError.ioError(path: shardPaths[task.shardIndex], errno: EIO)
            }
        }

        let maxCoalescedBytes = 32 * 1024 * 1024
        var workItems: [ReadTask] = []
        workItems.reserveCapacity(sorted.count)
        var totalBytes = 0

        for task in sorted {
            let fileSize = fileSizes[task.shardIndex]
            let path = shardPaths[task.shardIndex]
            guard task.byteCount >= 0,
                  task.fileOffset <= fileSize,
                  UInt64(task.byteCount) <= fileSize - task.fileOffset
            else {
                throw Inkling2StreamingError.shortRead(
                    "\(path) slice [\(task.fileOffset), +\(task.byteCount)) exceeds file size \(fileSize)")
            }
            guard task.byteCount > 0 else { continue }
            totalBytes += task.byteCount

            if let prev = workItems.last,
               prev.shardIndex == task.shardIndex,
               prev.fileOffset + UInt64(prev.byteCount) == task.fileOffset,
               prev.destination.advanced(by: prev.byteCount) == task.destination,
               prev.byteCount + task.byteCount <= maxCoalescedBytes,
               (!bypassPageCache || (prev.byteCount + task.byteCount + 32768) <= bounceCapacity)
            {
                workItems[workItems.count - 1] = ReadTask(
                    shardIndex: prev.shardIndex,
                    fileOffset: prev.fileOffset,
                    byteCount: prev.byteCount + task.byteCount,
                    destination: prev.destination)
            } else {
                workItems.append(task)
            }
        }

        guard !workItems.isEmpty else { return (0, 0) }

        if workItems.count == 1 || maxWorkers <= 1 || (!bypassPageCache && totalBytes < 2 * 1024 * 1024) {
            var rawTotal: UInt64 = 0
            var alignedTotal: UInt64 = 0
            let bounce = activeBuffers.first
            for item in workItems {
                let aligned = try Self.readItem(
                    fd: activeFds[item.shardIndex],
                    fileSize: fileSizes[item.shardIndex],
                    path: shardPaths[item.shardIndex],
                    item: item,
                    bypassPageCache: bypassPageCache,
                    bounce: bounce,
                    bounceCapacity: bounceCapacity)
                rawTotal += UInt64(item.byteCount)
                alignedTotal += aligned
            }
            return (rawTotal, alignedTotal)
        }

        workItems.sort { $0.byteCount > $1.byteCount }
        let sizeWorkers = max(1, (totalBytes + (2 * 1024 * 1024 - 1)) / (2 * 1024 * 1024))
        let workerCount = bypassPageCache
            ? min(activeBuffers.count, workItems.count)
            : min(maxWorkers, min(workItems.count, sizeWorkers))
        var perWorkerRaw = [UInt64](repeating: 0, count: workerCount)
        var perWorkerAligned = [UInt64](repeating: 0, count: workerCount)
        var perWorkerError = [Error?](repeating: nil, count: workerCount)

        perWorkerRaw.withUnsafeMutableBufferPointer { rawBuf in
            perWorkerAligned.withUnsafeMutableBufferPointer { alignedBuf in
                perWorkerError.withUnsafeMutableBufferPointer { errBuf in
                    DispatchQueue.concurrentPerform(iterations: workerCount) { workerIdx in
                        let bounce = bypassPageCache ? activeBuffers[workerIdx] : nil
                        var idx = workerIdx
                        while idx < workItems.count {
                            let item = workItems[idx]
                            do {
                                let aligned = try Self.readItem(
                                    fd: activeFds[item.shardIndex],
                                    fileSize: fileSizes[item.shardIndex],
                                    path: shardPaths[item.shardIndex],
                                    item: item,
                                    bypassPageCache: bypassPageCache,
                                    bounce: bounce,
                                    bounceCapacity: bounceCapacity)
                                rawBuf[workerIdx] += UInt64(item.byteCount)
                                alignedBuf[workerIdx] += aligned
                            } catch {
                                errBuf[workerIdx] = error
                                return
                            }
                            idx += workerCount
                        }
                    }
                }
            }
        }

        if let firstErr = perWorkerError.compactMap({ $0 }).first {
            throw firstErr
        }
        return (perWorkerRaw.reduce(0, +), perWorkerAligned.reduce(0, +))
    }

    private static func readItem(
        fd: Int32,
        fileSize: UInt64,
        path: String,
        item: ReadTask,
        bypassPageCache: Bool,
        bounce: UnsafeMutableRawPointer?,
        bounceCapacity: Int
    ) throws -> UInt64 {
        let alignMask = UInt64(16383)
        let alignedStart = item.fileOffset & ~alignMask
        let leading = Int(item.fileOffset - alignedStart)
        let alignedLength = (leading + item.byteCount + 16383) & ~16383
        // Clamp EOF reads to the file's real end because safetensors shards are not 16 KB padded.
        let clampedEnd = min(alignedStart + UInt64(alignedLength), fileSize)
        let readLength = Int(clampedEnd - alignedStart)
        guard readLength >= leading + item.byteCount else {
            throw Inkling2StreamingError.shortRead(
                "\(path) clamped read \(readLength) < required \(leading + item.byteCount)")
        }

        if !bypassPageCache {
            try preadFull(
                fd: fd,
                path: path,
                buffer: item.destination,
                length: item.byteCount,
                offset: off_t(item.fileOffset))
            return UInt64(readLength)
        }

        if readLength > bounceCapacity || bounce == nil {
            var tempPtr: UnsafeMutableRawPointer?
            let rc = posix_memalign(&tempPtr, 16384, alignedLength)
            guard rc == 0, let temp = tempPtr else {
                throw Inkling2StreamingError.ioError(path: path, errno: rc)
            }
            defer { free(temp) }
            try preadFull(fd: fd, path: path, buffer: temp, length: readLength, offset: off_t(alignedStart))
            memcpy(item.destination, temp.advanced(by: leading), item.byteCount)
            return UInt64(readLength)
        }

        let alignedBuffer = bounce!
        try preadFull(fd: fd, path: path, buffer: alignedBuffer, length: readLength, offset: off_t(alignedStart))
        memcpy(item.destination, alignedBuffer.advanced(by: leading), item.byteCount)
        return UInt64(readLength)
    }

    private static func preadFull(
        fd: Int32,
        path: String,
        buffer: UnsafeMutableRawPointer,
        length: Int,
        offset: off_t
    ) throws {
        var remaining = length
        var currentOffset = offset
        var dstOffset = 0
        while remaining > 0 {
            let n = pread(fd, buffer.advanced(by: dstOffset), remaining, currentOffset)
            if n < 0 {
                let code = errno
                if code == EINTR { continue }
                throw Inkling2StreamingError.ioError(path: path, errno: code)
            }
            if n == 0 {
                throw Inkling2StreamingError.shortRead("\(path) at offset \(currentOffset)")
            }
            remaining -= n
            currentOffset += off_t(n)
            dstOffset += n
        }
    }
}

// MARK: - Shared Slot Buffer Set (Persistent Cache & Prefill Staging)

private final class Inkling2SlotBufferSet: @unchecked Sendable {
    let slotCount: Int
    let gateUpWeight: MLXArray
    let gateUpScales: MLXArray
    let gateUpBiases: MLXArray?
    let downWeight: MLXArray
    let downScales: MLXArray
    let downBiases: MLXArray?

    let gateUpWeightPtr: UnsafeMutableRawPointer
    let gateUpScalesPtr: UnsafeMutableRawPointer
    let gateUpBiasesPtr: UnsafeMutableRawPointer?
    let downWeightPtr: UnsafeMutableRawPointer
    let downScalesPtr: UnsafeMutableRawPointer
    let downBiasesPtr: UnsafeMutableRawPointer?

    var gateUpGlobalScale: MLXArray?
    var downGlobalScale: MLXArray?
    let gateUpGlobalScalePtr: UnsafeMutablePointer<Float>?
    let downGlobalScalePtr: UnsafeMutablePointer<Float>?
    private var gateUpGlobalScaleCPU: [Float]
    private var downGlobalScaleCPU: [Float]
    private var gateUpGlobalScaleDirty = false
    private var downGlobalScaleDirty = false

    init(slotCount: Int, meta: Inkling2LayerFileMeta) throws {
        self.slotCount = slotCount
        let gu = meta.gateUp
        let dn = meta.down

        let guW = MLXArray.zeros([slotCount, gu.outputDims, gu.packedCols], dtype: .uint32)
        let guSDType: DType = gu.format.mode == .affine ? .bfloat16 : .uint8
        let guS = MLXArray.zeros([slotCount, gu.outputDims, gu.groupCols], dtype: guSDType)
        let guB: MLXArray? = gu.format.mode == .affine
            ? MLXArray.zeros([slotCount, gu.outputDims, gu.groupCols], dtype: .bfloat16) : nil

        let dnW = MLXArray.zeros([slotCount, dn.outputDims, dn.packedCols], dtype: .uint32)
        let dnSDType: DType = dn.format.mode == .affine ? .bfloat16 : .uint8
        let dnS = MLXArray.zeros([slotCount, dn.outputDims, dn.groupCols], dtype: dnSDType)
        let dnB: MLXArray? = dn.format.mode == .affine
            ? MLXArray.zeros([slotCount, dn.outputDims, dn.groupCols], dtype: .bfloat16) : nil

        var arraysToEval: [MLXArray] = [guW, guS, dnW, dnS]
        if let guB { arraysToEval.append(guB) }
        if let dnB { arraysToEval.append(dnB) }

        var guGSCPU: [Float] = []
        var dnGSCPU: [Float] = []
        let guGSArray: MLXArray?
        let dnGSArray: MLXArray?

        if let values = gu.fullGlobalScale {
            let initial = values.first ?? 1
            if slotCount >= 5 {
                let arr = MLXArray.zeros([slotCount], dtype: .float32)
                arraysToEval.append(arr)
                guGSArray = arr
            } else {
                guGSCPU = [Float](repeating: initial, count: slotCount)
                let arr = MLXArray(guGSCPU)
                arraysToEval.append(arr)
                guGSArray = arr
            }
        } else {
            guGSArray = nil
        }

        if let values = dn.fullGlobalScale {
            let initial = values.first ?? 1
            if slotCount >= 5 {
                let arr = MLXArray.zeros([slotCount], dtype: .float32)
                arraysToEval.append(arr)
                dnGSArray = arr
            } else {
                dnGSCPU = [Float](repeating: initial, count: slotCount)
                let arr = MLXArray(dnGSCPU)
                arraysToEval.append(arr)
                dnGSArray = arr
            }
        } else {
            dnGSArray = nil
        }

        eval(arraysToEval)

        self.gateUpWeight = guW
        self.gateUpScales = guS
        self.gateUpBiases = guB
        self.downWeight = dnW
        self.downScales = dnS
        self.downBiases = dnB

        self.gateUpWeightPtr = try Self.writableBasePointer(guW)
        self.gateUpScalesPtr = try Self.writableBasePointer(guS)
        self.gateUpBiasesPtr = try guB.map { try Self.writableBasePointer($0) }
        self.downWeightPtr = try Self.writableBasePointer(dnW)
        self.downScalesPtr = try Self.writableBasePointer(dnS)
        self.downBiasesPtr = try dnB.map { try Self.writableBasePointer($0) }

        self.gateUpGlobalScale = guGSArray
        self.downGlobalScale = dnGSArray
        self.gateUpGlobalScaleCPU = guGSCPU
        self.downGlobalScaleCPU = dnGSCPU

        if slotCount >= 5, let guGSArray {
            let ptr = try Self.writableBasePointer(guGSArray).assumingMemoryBound(to: Float.self)
            let initial = gu.fullGlobalScale?.first ?? 1
            for i in 0 ..< slotCount { ptr[i] = initial }
            self.gateUpGlobalScalePtr = ptr
        } else {
            self.gateUpGlobalScalePtr = nil
        }

        if slotCount >= 5, let dnGSArray {
            let ptr = try Self.writableBasePointer(dnGSArray).assumingMemoryBound(to: Float.self)
            let initial = dn.fullGlobalScale?.first ?? 1
            for i in 0 ..< slotCount { ptr[i] = initial }
            self.downGlobalScalePtr = ptr
        } else {
            self.downGlobalScalePtr = nil
        }
    }

    static func writableBasePointer(_ array: MLXArray) throws -> UnsafeMutableRawPointer {
        let view = array.asData(access: .noCopy)
        guard view.data.count >= array.nbytes, array.nbytes > 16 else {
            // Buffers strictly > 16 bytes avoid MLX Small-array inline storage and keep the Metal shared pointer.
            throw Inkling2StreamingError.zeroCopyAliasingFailed
        }
        return try view.data.withUnsafeBytes { rawBuf in
            guard let base = rawBuf.baseAddress else {
                throw Inkling2StreamingError.zeroCopyAliasingFailed
            }
            return UnsafeMutableRawPointer(mutating: base)
        }
    }

    func updateGlobalScale(slot: Int, expert: Int, meta: Inkling2LayerFileMeta) {
        let idx = max(0, expert)
        if let fullGU = meta.gateUp.fullGlobalScale {
            let val = fullGU[idx]
            if let ptr = gateUpGlobalScalePtr {
                ptr[slot] = val
            } else {
                gateUpGlobalScaleCPU[slot] = val
                gateUpGlobalScaleDirty = true
            }
        }
        if let fullDN = meta.down.fullGlobalScale {
            let val = fullDN[idx]
            if let ptr = downGlobalScalePtr {
                ptr[slot] = val
            } else {
                downGlobalScaleCPU[slot] = val
                downGlobalScaleDirty = true
            }
        }
    }

    func updateGlobalScales(experts: [Int], meta: Inkling2LayerFileMeta) {
        for slot in 0 ..< experts.count {
            updateGlobalScale(slot: slot, expert: experts[slot], meta: meta)
        }
    }

    func makeTensors(meta: Inkling2LayerFileMeta) -> (Inkling2ExpertTensors, Inkling2ExpertTensors) {
        if gateUpGlobalScaleDirty {
            gateUpGlobalScale = MLXArray(gateUpGlobalScaleCPU)
            gateUpGlobalScaleDirty = false
        }
        if downGlobalScaleDirty {
            downGlobalScale = MLXArray(downGlobalScaleCPU)
            downGlobalScaleDirty = false
        }
        let gu = Inkling2ExpertTensors(
            weight: gateUpWeight,
            scales: gateUpScales,
            biases: gateUpBiases,
            globalScale: gateUpGlobalScale,
            format: meta.gateUp.format)
        let dn = Inkling2ExpertTensors(
            weight: downWeight,
            scales: downScales,
            biases: downBiases,
            globalScale: downGlobalScale,
            format: meta.down.format)
        return (gu, dn)
    }

    func appendReadTasks(
        forExpert expert: Int,
        intoSlot slot: Int,
        meta: Inkling2LayerFileMeta,
        tasks: inout [Inkling2AlignedShardReader.ReadTask]
    ) {
        Self.appendProjectionReadTasks(
            forExpert: expert,
            intoSlot: slot,
            projection: meta.gateUp,
            weightPtr: gateUpWeightPtr,
            scalesPtr: gateUpScalesPtr,
            biasesPtr: gateUpBiasesPtr,
            tasks: &tasks)
        Self.appendProjectionReadTasks(
            forExpert: expert,
            intoSlot: slot,
            projection: meta.down,
            weightPtr: downWeightPtr,
            scalesPtr: downScalesPtr,
            biasesPtr: downBiasesPtr,
            tasks: &tasks)
    }

    func copyExpert(
        from source: Inkling2SlotBufferSet,
        sourceSlot: Int,
        destinationSlot: Int,
        meta: Inkling2LayerFileMeta
    ) {
        Self.copyProjection(
            projection: meta.gateUp,
            srcWeight: source.gateUpWeightPtr,
            srcScales: source.gateUpScalesPtr,
            srcBiases: source.gateUpBiasesPtr,
            srcSlot: sourceSlot,
            dstWeight: gateUpWeightPtr,
            dstScales: gateUpScalesPtr,
            dstBiases: gateUpBiasesPtr,
            dstSlot: destinationSlot)
        Self.copyProjection(
            projection: meta.down,
            srcWeight: source.downWeightPtr,
            srcScales: source.downScalesPtr,
            srcBiases: source.downBiasesPtr,
            srcSlot: sourceSlot,
            dstWeight: downWeightPtr,
            dstScales: downScalesPtr,
            dstBiases: downBiasesPtr,
            dstSlot: destinationSlot)
    }

    private static func appendProjectionReadTasks(
        forExpert expert: Int,
        intoSlot slot: Int,
        projection: Inkling2ProjectionFileMeta,
        weightPtr: UnsafeMutableRawPointer,
        scalesPtr: UnsafeMutableRawPointer,
        biasesPtr: UnsafeMutableRawPointer?,
        tasks: inout [Inkling2AlignedShardReader.ReadTask]
    ) {
        let wBytes = projection.weight.perExpertBytes
        tasks.append(
            Inkling2AlignedShardReader.ReadTask(
                shardIndex: projection.weight.shardIndex,
                fileOffset: projection.weight.baseFileOffset + UInt64(expert) * wBytes,
                byteCount: Int(wBytes),
                destination: weightPtr.advanced(by: slot * Int(wBytes))))

        let sBytes = projection.scales.perExpertBytes
        tasks.append(
            Inkling2AlignedShardReader.ReadTask(
                shardIndex: projection.scales.shardIndex,
                fileOffset: projection.scales.baseFileOffset + UInt64(expert) * sBytes,
                byteCount: Int(sBytes),
                destination: scalesPtr.advanced(by: slot * Int(sBytes))))

        if let bMeta = projection.biases, let bPtr = biasesPtr {
            let bBytes = bMeta.perExpertBytes
            tasks.append(
                Inkling2AlignedShardReader.ReadTask(
                    shardIndex: bMeta.shardIndex,
                    fileOffset: bMeta.baseFileOffset + UInt64(expert) * bBytes,
                    byteCount: Int(bBytes),
                    destination: bPtr.advanced(by: slot * Int(bBytes))))
        }
    }

    private static func copyProjection(
        projection: Inkling2ProjectionFileMeta,
        srcWeight: UnsafeMutableRawPointer,
        srcScales: UnsafeMutableRawPointer,
        srcBiases: UnsafeMutableRawPointer?,
        srcSlot: Int,
        dstWeight: UnsafeMutableRawPointer,
        dstScales: UnsafeMutableRawPointer,
        dstBiases: UnsafeMutableRawPointer?,
        dstSlot: Int
    ) {
        let wBytes = Int(projection.weight.perExpertBytes)
        memcpy(dstWeight.advanced(by: dstSlot * wBytes), srcWeight.advanced(by: srcSlot * wBytes), wBytes)

        let sBytes = Int(projection.scales.perExpertBytes)
        memcpy(dstScales.advanced(by: dstSlot * sBytes), srcScales.advanced(by: srcSlot * sBytes), sBytes)

        if let bMeta = projection.biases, let sB = srcBiases, let dB = dstBiases {
            let bBytes = Int(bMeta.perExpertBytes)
            memcpy(dB.advanced(by: dstSlot * bBytes), sB.advanced(by: srcSlot * bBytes), bBytes)
        }
    }
}

private struct Inkling2PrefillStagingGeometry: Equatable {
    let gateUpOutputDims: Int
    let gateUpPackedCols: Int
    let gateUpGroupCols: Int
    let gateUpMode: QuantizationMode
    let gateUpHasBiases: Bool
    let gateUpHasGlobalScale: Bool
    let downOutputDims: Int
    let downPackedCols: Int
    let downGroupCols: Int
    let downMode: QuantizationMode
    let downHasBiases: Bool
    let downHasGlobalScale: Bool

    init(meta: Inkling2LayerFileMeta) {
        self.gateUpOutputDims = meta.gateUp.outputDims
        self.gateUpPackedCols = meta.gateUp.packedCols
        self.gateUpGroupCols = meta.gateUp.groupCols
        self.gateUpMode = meta.gateUp.format.mode
        self.gateUpHasBiases = meta.gateUp.biases != nil
        self.gateUpHasGlobalScale = meta.gateUp.fullGlobalScale != nil
        self.downOutputDims = meta.down.outputDims
        self.downPackedCols = meta.down.packedCols
        self.downGroupCols = meta.down.groupCols
        self.downMode = meta.down.format.mode
        self.downHasBiases = meta.down.biases != nil
        self.downHasGlobalScale = meta.down.fullGlobalScale != nil
    }
}

private final class Inkling2PrefillStagingBuffer: @unchecked Sendable {
    let geometry: Inkling2PrefillStagingGeometry
    let buffers: Inkling2SlotBufferSet

    init(geometry: Inkling2PrefillStagingGeometry, slotCount: Int, meta: Inkling2LayerFileMeta) throws {
        self.geometry = geometry
        self.buffers = try Inkling2SlotBufferSet(slotCount: slotCount, meta: meta)
    }
}

// MARK: - Per-Layer Pre-Allocated Slot Cache + GPU Reader Sync Barrier

final class Inkling2LayerExpertCache: @unchecked Sendable {
    let layer: Int
    let numExperts: Int
    let topK: Int
    let slotCapacity: Int
    let policy: Inkling2EvictionPolicy
    let passes: Int
    let isLoopCore: Bool
    let residentBytes: UInt64

    private let meta: Inkling2LayerFileMeta
    private let reader: Inkling2AlignedShardReader
    private let activation: @Sendable (MLXArray, MLXArray) -> MLXArray
    private let softcaps: MLXArray
    private weak var owner: Inkling2ExpertStreamer?

    private let lock = NSLock()
    private let buffers: Inkling2SlotBufferSet

    // GPU reader completion barrier: must be evaluated before ANY slot buffer overwrite.
    private var lastReader: MLXArray?
    private var closed = false

    // Flat slot state machine (-1 == invalid/empty) with zero heap allocations on hot path.
    private var slotToExpert: [Int]
    private var lastUsedTick: [UInt64]
    private var tickCounter: UInt64 = 0

    // Flat lookup tables: perPassActive indexes by pass * numExperts + expert; shared indexes by expert.
    private var expertToSlotTable: [Int]
    // Shared SLRU tiering flags + count
    private var isProtectedSlot: [Bool]
    private var protectedCount: Int = 0
    private let protectedCap: Int
    // Epoch-stamped scratch tables for O(1) pinning and duplicate detection inside lock
    private var pinEpoch: UInt64 = 0
    private var expertPinnedEpoch: [UInt64]
    private var slotPinnedEpoch: [UInt64]
    private var expertMissEpoch: [UInt64]
    private var expertMissSlot: [Int]
    // Per-pass LRU partitioning (used only when policy == .perPassLRU && effectivePassCap >= topK && passes > 1)
    private let perPassActive: Bool
    private let passCap: Int

    init(
        layer: Int,
        slotCapacity: Int,
        passes: Int,
        isLoopCore: Bool,
        textConfig: Inkling2TextConfiguration,
        meta: Inkling2LayerFileMeta,
        reader: Inkling2AlignedShardReader,
        policy: Inkling2EvictionPolicy,
        activation: @escaping @Sendable (MLXArray, MLXArray) -> MLXArray,
        softcaps: MLXArray,
        owner: Inkling2ExpertStreamer
    ) throws {
        let effectiveSlots = max(textConfig.expertsPerToken, min(slotCapacity, textConfig.experts))
        self.layer = layer
        self.numExperts = textConfig.experts
        self.topK = textConfig.expertsPerToken
        self.passes = max(1, passes)
        self.isLoopCore = isLoopCore
        self.meta = meta
        self.reader = reader
        self.activation = activation
        self.softcaps = softcaps
        self.owner = owner

        let corePasses = isLoopCore ? max(1, passes) : 1
        let computedPassCap = effectiveSlots / corePasses
        let usePerPass = (policy == .perPassLRU && isLoopCore && corePasses > 1 && computedPassCap >= textConfig.expertsPerToken)
        self.perPassActive = usePerPass
        self.passCap = usePerPass ? computedPassCap : effectiveSlots
        self.slotCapacity = usePerPass ? (computedPassCap * corePasses) : effectiveSlots
        self.policy = usePerPass ? .perPassLRU : (policy == .perPassLRU ? .sharedLRU : policy)
        self.protectedCap = self.slotCapacity / 2

        let s = self.slotCapacity
        self.buffers = try Inkling2SlotBufferSet(slotCount: s, meta: meta)

        self.slotToExpert = Array(repeating: -1, count: s)
        self.lastUsedTick = Array(repeating: 0, count: s)
        self.isProtectedSlot = Array(repeating: false, count: s)
        let tableCount = (usePerPass ? corePasses : 1) * textConfig.experts
        self.expertToSlotTable = Array(repeating: -1, count: tableCount)
        self.expertPinnedEpoch = Array(repeating: 0, count: textConfig.experts)
        self.slotPinnedEpoch = Array(repeating: 0, count: s)
        self.expertMissEpoch = Array(repeating: 0, count: textConfig.experts)
        self.expertMissSlot = Array(repeating: -1, count: textConfig.experts)

        let gu = meta.gateUp
        let dn = meta.down
        let perSlotTotal =
            gu.weight.perExpertBytes + gu.scales.perExpertBytes + (gu.biases?.perExpertBytes ?? 0)
            + dn.weight.perExpertBytes + dn.scales.perExpertBytes + (dn.biases?.perExpertBytes ?? 0)
        self.residentBytes = UInt64(s) * perSlotTotal
    }

    /// Verifies that writes through the cached raw Metal shared buffer pointers are
    /// immediately visible to MLX evaluation without copying.
    func verifyZeroCopyAliasing() throws {
        try lock.withLock {
            let sentinel: UInt32 = 0xA55A_5AA5
            buffers.gateUpWeightPtr.storeBytes(of: sentinel, as: UInt32.self)
            let observed = buffers.gateUpWeight[0, 0, 0].item(UInt32.self)
            buffers.gateUpWeightPtr.storeBytes(of: UInt32(0), as: UInt32.self)
            let restored = buffers.gateUpWeight[0, 0, 0].item(UInt32.self)
            guard observed == sentinel, restored == 0 else {
                throw Inkling2StreamingError.zeroCopyAliasingFailed
            }
            if let gsPtr = buffers.gateUpGlobalScalePtr, let gsArr = buffers.gateUpGlobalScale {
                let original = gsArr[0].item(Float.self)
                let probe: Float = 7.25
                gsPtr[0] = probe
                let seen = gsArr[0].item(Float.self)
                gsPtr[0] = original
                let reverted = gsArr[0].item(Float.self)
                guard seen == probe, reverted == original else {
                    throw Inkling2StreamingError.zeroCopyAliasingFailed
                }
            }
        }
    }

    func recordError(_ error: Error) {
        owner?.recordError(error)
    }

    func close() {
        let readerToSync: MLXArray? = lock.withLock {
            closed = true
            let r = lastReader
            lastReader = nil
            slotToExpert.indices.forEach { slotToExpert[$0] = -1 }
            expertToSlotTable.indices.forEach { expertToSlotTable[$0] = -1 }
            isProtectedSlot.indices.forEach { isProtectedSlot[$0] = false }
            protectedCount = 0
            return r
        }
        if let readerToSync {
            eval(readerToSync)
        }
    }

    // MARK: - Decode Forward

    func decode(
        _ x: MLXArray,
        indices: MLXArray,
        weights: MLXArray,
        residual: MLXArray,
        state: MLXArray,
        convolutionWeight: MLXArray,
        pass: Int
    ) throws -> (MLXArray, MLXArray) {
        owner?.releasePrefillStagingIfAllocated()

        let priorReader = lock.withLock { () -> MLXArray? in
            let r = lastReader
            lastReader = nil
            return r
        }
        // GPU reader synchronization barrier: wait for BOTH prior layer readers AND router indices.
        if let priorReader {
            eval(priorReader, indices)
        } else {
            eval(indices)
        }

        let indicesU32 = indices.dtype == .uint32 ? indices : indices.asType(.uint32)
        let rawIndices = indicesU32.asData(access: .noCopy)
        let expectedCount = indicesU32.size
        let (slotIDs, guTensors, dnTensors): ([UInt32], Inkling2ExpertTensors, Inkling2ExpertTensors) = try {
            if rawIndices.data.count >= expectedCount * MemoryLayout<UInt32>.stride {
                return try rawIndices.data.withUnsafeBytes { rawBuf in
                    let typed = rawBuf.bindMemory(to: UInt32.self)
                    return try prepareSlots(
                        for: UnsafeBufferPointer(rebasing: typed.prefix(expectedCount)),
                        pass: pass)
                }
            }
            let fallback = indicesU32.asArray(UInt32.self)
            return try prepareSlots(for: fallback, pass: pass)
        }()
        let slotIndices = slotIDs.withUnsafeBufferPointer { ptr in
            MLXArray(ptr, indices.shape)
        }

        let dims = meta.down.outputDims
        let hidden = meta.gateUp.outputDims / 2
        let (outHidden, outState): (MLXArray, MLXArray)
        if slotIndices.size < 64, slotIndices.dim(-1) <= 32, hidden % 4 == 0, dims % 2 == 0,
           guTensors.supportsFusedDecode(inputDims: dims),
           dnTensors.supportsFusedDecode(inputDims: hidden)
        {
            let activations = inkling2ExpertGateUp(
                x, experts: guTensors, indices: slotIndices, softcaps: softcaps)
            (outHidden, outState) = inkling2ExpertDown(
                activations, experts: dnTensors, indices: slotIndices, weights: weights,
                residual: residual, state: state, convolutionWeight: convolutionWeight)
        } else {
            let expandedX = MLX.expandedDimensions(x, axes: [-2, -3])
            let gateUp = gatherQuantizedMM(
                expandedX, guTensors.weight, scales: guTensors.scales, biases: guTensors.biases,
                rhsIndices: slotIndices, transpose: true,
                groupSize: guTensors.format.groupSize, bits: guTensors.format.bits,
                mode: guTensors.format.mode, globalScale: guTensors.globalScale,
                sortedIndices: false)
            let activated = activation(
                gateUp[.ellipsis, .stride(from: 0, by: 2)],
                gateUp[.ellipsis, .stride(from: 1, by: 2)])
            let down = gatherQuantizedMM(
                activated, dnTensors.weight, scales: dnTensors.scales, biases: dnTensors.biases,
                rhsIndices: slotIndices, transpose: true,
                groupSize: dnTensors.format.groupSize, bits: dnTensors.format.bits,
                mode: dnTensors.format.mode, globalScale: dnTensors.globalScale,
                sortedIndices: false)
                .reshaped(slotIndices.shape + [dims])
            (outHidden, outState) = inkling2ExpertEpilogue(
                outputs: down, weights: weights, residual: residual, state: state,
                convolutionWeight: convolutionWeight)
        }

        lock.withLock {
            lastReader = outHidden
        }
        return (outHidden, outState)
    }

    // MARK: - Prefill Forward (Fast Path A Persistent vs Fast Path B Staged + Warm Seed)

    func prefill(
        _ x: MLXArray,
        indices: MLXArray,
        weights: MLXArray,
        pass: Int
    ) throws -> MLXArray {
        let priorReader = lock.withLock { () -> MLXArray? in
            let r = lastReader
            lastReader = nil
            return r
        }
        // Wait for prior layer readers and router indices before touching slots.
        if let priorReader {
            eval(priorReader, indices)
        } else {
            eval(indices)
        }

        let shape = x.shape
        let b = shape[0]
        let l = shape[1]
        let dims = shape[2]
        let rowsCount = b * l
        let kCount = indices.dim(-1)
        let totalPairs = rowsCount * kCount
        guard totalPairs > 0 else { return MLXArray.zeros(like: x) }

        let indicesU32 = indices.dtype == .uint32 ? indices : indices.asType(.uint32)
        let flatIndices = indicesU32.asArray(UInt32.self)
        guard flatIndices.count == totalPairs else {
            throw Inkling2StreamingError.invalidConfiguration(
                "Router indices count mismatch: \(flatIndices.count) != \(totalPairs)")
        }

        // O(totalPairs) flat CSR histogram (zero Dictionary / tuple-array heap allocations).
        var expertCounts = [Int](repeating: 0, count: numExperts)
        var isInLastRow = [Bool](repeating: false, count: numExperts)
        let lastRowStart = (rowsCount - 1) * kCount
        for p in 0 ..< totalPairs {
            let e = Int(flatIndices[p])
            guard e >= 0, e < numExperts else {
                throw Inkling2StreamingError.invalidConfiguration(
                    "Invalid expert ID \(e) at layer \(layer) (numExperts=\(numExperts))")
            }
            expertCounts[e] += 1
            if p >= lastRowStart {
                isInLastRow[e] = true
            }
        }

        var activeExperts: [Int] = []
        activeExperts.reserveCapacity(numExperts)
        var expertOffsets = [Int](repeating: 0, count: numExperts + 1)
        var running = 0
        for e in 0 ..< numExperts {
            expertOffsets[e] = running
            let c = expertCounts[e]
            if c > 0 {
                activeExperts.append(e)
                running += c
            }
        }
        expertOffsets[numExperts] = running

        var pairRows = [Int32](repeating: 0, count: totalPairs)
        var pairIndices = [Int32](repeating: 0, count: totalPairs)
        var writeCursor = expertOffsets
        for p in 0 ..< totalPairs {
            let e = Int(flatIndices[p])
            let dst = writeCursor[e]
            writeCursor[e] = dst + 1
            pairRows[dst] = Int32(p / kCount)
            pairIndices[dst] = Int32(p)
        }

        let xFlat = x.reshaped(rowsCount, dims)
        let residentBatchLimit = perPassActive ? passCap : slotCapacity

        // Fast Path A: active experts fit inside this layer's persistent slots without eviction.
        if activeExperts.count <= residentBatchLimit {
            let requestedU32 = activeExperts.map(UInt32.init)
            let (slotMapU32, guTensors, dnTensors) = try prepareSlots(for: requestedU32, pass: pass)

            var slotOrder: [(slot: UInt32, expert: Int)] = []
            slotOrder.reserveCapacity(activeExperts.count)
            for i in 0 ..< activeExperts.count {
                slotOrder.append((slot: slotMapU32[i], expert: activeExperts[i]))
            }
            slotOrder.sort { $0.slot < $1.slot }

            var rowIDs: [Int32] = []
            var slotIDs: [UInt32] = []
            var pairIDs: [Int32] = []
            rowIDs.reserveCapacity(totalPairs)
            slotIDs.reserveCapacity(totalPairs)
            pairIDs.reserveCapacity(totalPairs)

            for entry in slotOrder {
                appendBatchPairs(
                    expert: entry.expert,
                    slot: entry.slot,
                    expertOffsets: expertOffsets,
                    pairRows: pairRows,
                    pairIndices: pairIndices,
                    rowIDs: &rowIDs,
                    slotIDs: &slotIDs,
                    pairIDs: &pairIDs)
            }

            let downOut = runMoEBatch(
                xFlat: xFlat,
                rowIDs: rowIDs,
                slotIDs: slotIDs,
                guTensors: guTensors,
                dnTensors: dnTensors,
                dims: dims)

            var inverseOrder = [Int32](repeating: 0, count: totalPairs)
            for j in 0 ..< pairIDs.count {
                inverseOrder[Int(pairIDs[j])] = Int32(j)
            }
            let orderedOutputs = downOut[MLXArray(inverseOrder)].reshaped(b, l, kCount, dims)
            let routed = reduceWeightedOutputs(orderedOutputs, weights: weights, targetDType: x.dtype)
            lock.withLock {
                lastReader = routed
            }
            return routed
        }

        // Fast Path B: stage active experts into the streamer's reusable staging buffer, eliminating
        // small-batch eviction thrashing while enabling contiguous file coalescing.
        guard let owner else {
            throw Inkling2StreamingError.ioError(path: "closed-streamer", errno: EBADF)
        }
        let staging = try owner.acquireStagingBuffer(
            for: meta,
            numExperts: numExperts,
            slotCapacity: slotCapacity,
            activeExpertCount: activeExperts.count)

        let effectivePass = min(pass, max(0, passes - 1))
        var residentPairs: [(expert: Int, persistentSlot: Int)] = []
        var missingExperts: [Int] = []
        var residentSlotLookup = [Int](repeating: -1, count: numExperts)

        let lockError = lock.withLock { () -> Error? in
            if closed {
                return Inkling2StreamingError.ioError(path: "closed-layer-cache", errno: EBADF)
            }
            for e in activeExperts {
                tickCounter &+= 1
                if let existingSlot = lookupSlotLocked(expert: e, pass: effectivePass) {
                    touchHitLocked(slot: existingSlot)
                    residentPairs.append((expert: e, persistentSlot: existingSlot))
                    residentSlotLookup[e] = existingSlot
                } else {
                    missingExperts.append(e)
                }
            }
            return nil
        }
        if let lockError {
            throw lockError
        }

        // Put resident experts first (always <= slotCapacity <= stagingSlotCount -> Batch 0) so
        // persistent slots can be warm-seeded safely on Batch 0, then sorted missing experts coalesce.
        var orderedExperts: [Int] = []
        orderedExperts.reserveCapacity(residentPairs.count + missingExperts.count)
        for r in residentPairs { orderedExperts.append(r.expert) }
        for e in missingExperts { orderedExperts.append(e) }

        let stageCap = staging.buffers.slotCount
        if orderedExperts.count <= stageCap {
            // Single-batch staging path (no scatter-outputs tensor, no inter-batch eval).
            var readTasks: [Inkling2AlignedShardReader.ReadTask] = []
            readTasks.reserveCapacity(missingExperts.count * 6)

            for (st, e) in orderedExperts.enumerated() {
                let resSlot = residentSlotLookup[e]
                if resSlot >= 0 {
                    staging.buffers.copyExpert(
                        from: buffers,
                        sourceSlot: resSlot,
                        destinationSlot: st,
                        meta: meta)
                } else {
                    staging.buffers.appendReadTasks(
                        forExpert: e,
                        intoSlot: st,
                        meta: meta,
                        tasks: &readTasks)
                }
            }

            var rawBytesRead: UInt64 = 0
            var alignedBytesRead: UInt64 = 0
            var ioNs: UInt64 = 0
            if !readTasks.isEmpty {
                let t0 = DispatchTime.now().uptimeNanoseconds
                let ioTotals = try reader.readSlicesConcurrently(readTasks)
                let t1 = DispatchTime.now().uptimeNanoseconds
                rawBytesRead = ioTotals.rawBytes
                alignedBytesRead = ioTotals.alignedBytes
                ioNs = t1 &- t0
            }

            staging.buffers.updateGlobalScales(experts: orderedExperts, meta: meta)
            let (guTensors, dnTensors) = staging.buffers.makeTensors(meta: meta)

            // Warm-seed persistent slots via unified-RAM memcpy while staged weights are ready.
            warmSeedPersistentSlots(
                from: staging,
                batchExperts: orderedExperts,
                expertCounts: expertCounts,
                isInLastRow: isInLastRow,
                pass: pass)

            var rowIDs: [Int32] = []
            var slotIDs: [UInt32] = []
            var pairIDs: [Int32] = []
            rowIDs.reserveCapacity(totalPairs)
            slotIDs.reserveCapacity(totalPairs)
            pairIDs.reserveCapacity(totalPairs)

            for (st, e) in orderedExperts.enumerated() {
                appendBatchPairs(
                    expert: e,
                    slot: UInt32(st),
                    expertOffsets: expertOffsets,
                    pairRows: pairRows,
                    pairIndices: pairIndices,
                    rowIDs: &rowIDs,
                    slotIDs: &slotIDs,
                    pairIDs: &pairIDs)
            }

            let downOut = runMoEBatch(
                xFlat: xFlat,
                rowIDs: rowIDs,
                slotIDs: slotIDs,
                guTensors: guTensors,
                dnTensors: dnTensors,
                dims: dims)

            var inverseOrder = [Int32](repeating: 0, count: totalPairs)
            for j in 0 ..< pairIDs.count {
                inverseOrder[Int(pairIDs[j])] = Int32(j)
            }
            let orderedOutputs = downOut[MLXArray(inverseOrder)].reshaped(b, l, kCount, dims)
            let routed = reduceWeightedOutputs(orderedOutputs, weights: weights, targetDType: x.dtype)

            owner.recordStats(
                requests: orderedExperts.count,
                hits: residentPairs.count,
                misses: missingExperts.count,
                rawBytes: rawBytesRead,
                alignedBytes: alignedBytesRead,
                ioNanoseconds: ioNs)
            owner.setLastStagingReader(routed)
            lock.withLock {
                lastReader = routed
            }
            return routed
        }

        // Multi-batch staging fallback (low-RAM Macs where a single layer exceeds staging RAM cap).
        let outputs = MLXArray.zeros([totalPairs, dims], dtype: x.dtype)
        var offset = 0
        var batchIndex = 0

        while offset < orderedExperts.count {
            if batchIndex > 0 {
                eval(outputs)
            }
            let end = min(orderedExperts.count, offset + stageCap)
            let batchExperts = Array(orderedExperts[offset ..< end])
            offset = end
            batchIndex += 1

            var batchHits = 0
            var batchMisses = 0
            var readTasks: [Inkling2AlignedShardReader.ReadTask] = []
            readTasks.reserveCapacity(batchExperts.count * 6)

            for (st, e) in batchExperts.enumerated() {
                let resSlot = residentSlotLookup[e]
                if resSlot >= 0 {
                    batchHits += 1
                    staging.buffers.copyExpert(
                        from: buffers,
                        sourceSlot: resSlot,
                        destinationSlot: st,
                        meta: meta)
                    residentSlotLookup[e] = -1
                } else {
                    batchMisses += 1
                    staging.buffers.appendReadTasks(
                        forExpert: e,
                        intoSlot: st,
                        meta: meta,
                        tasks: &readTasks)
                }
            }

            var rawBytesRead: UInt64 = 0
            var alignedBytesRead: UInt64 = 0
            var ioNs: UInt64 = 0
            if !readTasks.isEmpty {
                let t0 = DispatchTime.now().uptimeNanoseconds
                let ioTotals = try reader.readSlicesConcurrently(readTasks)
                let t1 = DispatchTime.now().uptimeNanoseconds
                rawBytesRead = ioTotals.rawBytes
                alignedBytesRead = ioTotals.alignedBytes
                ioNs = t1 &- t0
            }

            staging.buffers.updateGlobalScales(experts: batchExperts, meta: meta)
            let (guTensors, dnTensors) = staging.buffers.makeTensors(meta: meta)

            var rowIDs: [Int32] = []
            var slotIDs: [UInt32] = []
            var pairIDs: [Int32] = []
            for (st, e) in batchExperts.enumerated() {
                appendBatchPairs(
                    expert: e,
                    slot: UInt32(st),
                    expertOffsets: expertOffsets,
                    pairRows: pairRows,
                    pairIndices: pairIndices,
                    rowIDs: &rowIDs,
                    slotIDs: &slotIDs,
                    pairIDs: &pairIDs)
            }

            let downOut = runMoEBatch(
                xFlat: xFlat,
                rowIDs: rowIDs,
                slotIDs: slotIDs,
                guTensors: guTensors,
                dnTensors: dnTensors,
                dims: dims)
            outputs[MLXArray(pairIDs)] = downOut.asType(x.dtype)

            warmSeedPersistentSlots(
                from: staging,
                batchExperts: batchExperts,
                expertCounts: expertCounts,
                isInLastRow: isInLastRow,
                pass: pass)

            owner.recordStats(
                requests: batchExperts.count,
                hits: batchHits,
                misses: batchMisses,
                rawBytes: rawBytesRead,
                alignedBytes: alignedBytesRead,
                ioNanoseconds: ioNs)
        }

        let routed = reduceWeightedOutputs(
            outputs.reshaped(b, l, kCount, dims),
            weights: weights,
            targetDType: x.dtype)
        owner.setLastStagingReader(routed)
        lock.withLock {
            lastReader = routed
        }
        return routed
    }

    private func appendBatchPairs(
        expert: Int,
        slot: UInt32,
        expertOffsets: [Int],
        pairRows: [Int32],
        pairIndices: [Int32],
        rowIDs: inout [Int32],
        slotIDs: inout [UInt32],
        pairIDs: inout [Int32]
    ) {
        let start = expertOffsets[expert]
        let end = expertOffsets[expert + 1]
        for idx in start ..< end {
            rowIDs.append(pairRows[idx])
            slotIDs.append(slot)
            pairIDs.append(pairIndices[idx])
        }
    }

    private func runMoEBatch(
        xFlat: MLXArray,
        rowIDs: [Int32],
        slotIDs: [UInt32],
        guTensors: Inkling2ExpertTensors,
        dnTensors: Inkling2ExpertTensors,
        dims: Int
    ) -> MLXArray {
        // [M, 1, D] matches 1-to-1 with 1D rhsSlots [M] (no quadratic [M, M] broadcast).
        let gatheredX = xFlat[MLXArray(rowIDs)].expandedDimensions(axis: -2)
        let rhsSlots = MLXArray(slotIDs)

        let gateUp = gatherQuantizedMM(
            gatheredX, guTensors.weight, scales: guTensors.scales, biases: guTensors.biases,
            rhsIndices: rhsSlots, transpose: true,
            groupSize: guTensors.format.groupSize, bits: guTensors.format.bits,
            mode: guTensors.format.mode, globalScale: guTensors.globalScale,
            sortedIndices: true)
        let activated = activation(
            gateUp[.ellipsis, .stride(from: 0, by: 2)],
            gateUp[.ellipsis, .stride(from: 1, by: 2)])
        return gatherQuantizedMM(
            activated, dnTensors.weight, scales: dnTensors.scales, biases: dnTensors.biases,
            rhsIndices: rhsSlots, transpose: true,
            groupSize: dnTensors.format.groupSize, bits: dnTensors.format.bits,
            mode: dnTensors.format.mode, globalScale: dnTensors.globalScale,
            sortedIndices: true)
            .reshaped(rowIDs.count, dims)
    }

    private func reduceWeightedOutputs(
        _ outputs: MLXArray,
        weights: MLXArray,
        targetDType: DType
    ) -> MLXArray {
        // Sum rank-ordered k = 0 ..< K outputs in float32, matching Inkling2.weightedExperts.
        (outputs.asType(.float32) * weights[.ellipsis, .newAxis])
            .sum(axis: -2)
            .asType(targetDType)
    }

    private func warmSeedPersistentSlots(
        from staging: Inkling2PrefillStagingBuffer,
        batchExperts: [Int],
        expertCounts: [Int],
        isInLastRow: [Bool],
        pass: Int
    ) {
        // Preserve pass-0 warm experts across multi-pass shared core caches so decode starts hot.
        guard !isLoopCore || pass == 0 || perPassActive else { return }

        let targetCount = min(perPassActive ? passCap : slotCapacity, batchExperts.count)
        guard targetCount > 0 else { return }

        var ranked: [(expert: Int, stagingSlot: Int)] = []
        ranked.reserveCapacity(batchExperts.count)
        for (st, e) in batchExperts.enumerated() {
            ranked.append((expert: e, stagingSlot: st))
        }
        ranked.sort { a, b in
            let al = isInLastRow[a.expert] ? 1 : 0
            let bl = isInLastRow[b.expert] ? 1 : 0
            if al != bl { return al > bl }
            let ac = expertCounts[a.expert]
            let bc = expertCounts[b.expert]
            if ac != bc { return ac > bc }
            return a.expert < b.expert
        }
        let candidates = ranked.prefix(targetCount)

        struct PendingWarmCopy: Sendable {
            let expert: Int
            let stagingSlot: Int
            let persistentSlot: Int
        }

        let effectivePass = min(pass, max(0, passes - 1))
        let copyItems: [PendingWarmCopy] = lock.withLock {
            guard !closed else { return [] }
            let eph = nextPinEpochLocked()
            for c in candidates {
                expertPinnedEpoch[c.expert] = eph
            }

            var pending: [PendingWarmCopy] = []
            pending.reserveCapacity(candidates.count)
            for c in candidates {
                tickCounter &+= 1
                if let existing = lookupSlotLocked(expert: c.expert, pass: effectivePass) {
                    touchHitLocked(slot: existing)
                    slotPinnedEpoch[existing] = eph
                    continue
                }
                guard let victim = chooseVictimSlotLocked(pass: effectivePass, pinEpoch: eph) else {
                    break
                }
                invalidateSlotLocked(victim, pass: effectivePass)
                slotPinnedEpoch[victim] = eph
                pending.append(
                    PendingWarmCopy(
                        expert: c.expert,
                        stagingSlot: c.stagingSlot,
                        persistentSlot: victim))
            }
            return pending
        }

        guard !copyItems.isEmpty else { return }

        let dstBuffers = buffers
        let srcBuffers = staging.buffers
        let layerMeta = meta
        if copyItems.count >= 2 {
            let workers = min(4, copyItems.count)
            DispatchQueue.concurrentPerform(iterations: workers) { workerIdx in
                var idx = workerIdx
                while idx < copyItems.count {
                    let item = copyItems[idx]
                    dstBuffers.copyExpert(
                        from: srcBuffers,
                        sourceSlot: item.stagingSlot,
                        destinationSlot: item.persistentSlot,
                        meta: layerMeta)
                    idx += workers
                }
            }
        } else {
            let item = copyItems[0]
            dstBuffers.copyExpert(
                from: srcBuffers,
                sourceSlot: item.stagingSlot,
                destinationSlot: item.persistentSlot,
                meta: layerMeta)
        }

        lock.withLock {
            guard !closed else { return }
            for item in copyItems {
                tickCounter &+= 1
                commitLoadedSlotLocked(item.persistentSlot, expert: item.expert, pass: effectivePass)
                dstBuffers.updateGlobalScale(slot: item.persistentSlot, expert: item.expert, meta: layerMeta)
            }
        }
    }

    // MARK: - Slot Assignment & Pinned Eviction

    private func prepareSlots(
        for requestedExperts: [UInt32],
        pass: Int
    ) throws -> ([UInt32], Inkling2ExpertTensors, Inkling2ExpertTensors) {
        try requestedExperts.withUnsafeBufferPointer { buf in
            try prepareSlots(for: buf, pass: pass)
        }
    }

    private func prepareSlots(
        for requestedExperts: UnsafeBufferPointer<UInt32>,
        pass: Int
    ) throws -> ([UInt32], Inkling2ExpertTensors, Inkling2ExpertTensors) {
        struct PendingLoad {
            let expert: Int
            let slot: Int
        }

        var assignedSlots = [UInt32](repeating: 0, count: requestedExperts.count)
        var missesToLoad: [PendingLoad] = []

        var hitsCount = 0
        var missesCount = 0
        var earlyResultTensors: (Inkling2ExpertTensors, Inkling2ExpertTensors)?

        let earlyError = lock.withLock { () -> Error? in
            if closed {
                return Inkling2StreamingError.ioError(path: "closed-layer-cache", errno: EBADF)
            }
            let effectivePass = min(pass, max(0, passes - 1))
            let eph = nextPinEpochLocked()
            for rawExpert in requestedExperts {
                let expert = Int(rawExpert)
                guard expert >= 0, expert < numExperts else {
                    return Inkling2StreamingError.invalidConfiguration(
                        "Invalid expert ID \(expert) at layer \(layer) (numExperts=\(numExperts))")
                }
                expertPinnedEpoch[expert] = eph
            }

            for idx in 0 ..< requestedExperts.count {
                let expert = Int(requestedExperts[idx])
                tickCounter &+= 1
                if let existingSlot = lookupSlotLocked(expert: expert, pass: effectivePass) {
                    assignedSlots[idx] = UInt32(existingSlot)
                    touchHitLocked(slot: existingSlot)
                    hitsCount += 1
                    continue
                }
                if expertMissEpoch[expert] == eph {
                    // Duplicate expert requested within the same call counts as a hit on the staged fetch.
                    assignedSlots[idx] = UInt32(expertMissSlot[expert])
                    hitsCount += 1
                    continue
                }

                missesCount += 1
                guard let victimSlot = chooseVictimSlotLocked(pass: effectivePass, pinEpoch: eph) else {
                    return Inkling2StreamingError.invalidConfiguration(
                        "No evictable slot at layer \(layer) (capacity=\(slotCapacity))")
                }

                // Invalidate old resident entry BEFORE SSD I/O so an error never leaves stale weights marked valid.
                invalidateSlotLocked(victimSlot, pass: effectivePass)
                slotPinnedEpoch[victimSlot] = eph
                expertMissEpoch[expert] = eph
                expertMissSlot[expert] = victimSlot
                assignedSlots[idx] = UInt32(victimSlot)
                missesToLoad.append(PendingLoad(expert: expert, slot: victimSlot))
            }
            if missesToLoad.isEmpty {
                earlyResultTensors = buffers.makeTensors(meta: meta)
            }
            return nil
        }

        if let earlyError {
            throw earlyError
        }

        if let (guTensors, dnTensors) = earlyResultTensors {
            owner?.recordStats(
                requests: requestedExperts.count,
                hits: hitsCount,
                misses: 0,
                rawBytes: 0,
                alignedBytes: 0,
                ioNanoseconds: 0)
            return (assignedSlots, guTensors, dnTensors)
        }

        var readTasks: [Inkling2AlignedShardReader.ReadTask] = []
        readTasks.reserveCapacity(missesToLoad.count * 6)

        for item in missesToLoad {
            buffers.appendReadTasks(
                forExpert: item.expert,
                intoSlot: item.slot,
                meta: meta,
                tasks: &readTasks)
        }

        let t0 = DispatchTime.now().uptimeNanoseconds
        let ioTotals = try reader.readSlicesConcurrently(readTasks)
        let t1 = DispatchTime.now().uptimeNanoseconds
        let rawBytesRead = ioTotals.rawBytes
        let alignedBytesRead = ioTotals.alignedBytes
        let ioNs = t1 &- t0

        // Mark all newly loaded slots valid ONLY after all tensors finished reading.
        let (guTensors, dnTensors) = lock.withLock {
            let effectivePass = min(pass, max(0, passes - 1))
            for item in missesToLoad {
                tickCounter &+= 1
                commitLoadedSlotLocked(item.slot, expert: item.expert, pass: effectivePass)
                buffers.updateGlobalScale(slot: item.slot, expert: item.expert, meta: meta)
            }
            return buffers.makeTensors(meta: meta)
        }

        owner?.recordStats(
            requests: requestedExperts.count,
            hits: hitsCount,
            misses: missesCount,
            rawBytes: rawBytesRead,
            alignedBytes: alignedBytesRead,
            ioNanoseconds: ioNs)

        return (assignedSlots, guTensors, dnTensors)
    }

    private func slotRange(forPass pass: Int) -> Range<Int> {
        guard perPassActive else { return 0 ..< slotCapacity }
        let start = pass * passCap
        return start ..< (start + passCap)
    }

    private func tableBase(forPass pass: Int) -> Int {
        perPassActive ? (pass * numExperts) : 0
    }

    private func nextPinEpochLocked() -> UInt64 {
        pinEpoch &+= 1
        if pinEpoch == 0 {
            expertPinnedEpoch.indices.forEach { expertPinnedEpoch[$0] = 0 }
            slotPinnedEpoch.indices.forEach { slotPinnedEpoch[$0] = 0 }
            expertMissEpoch.indices.forEach { expertMissEpoch[$0] = 0 }
            pinEpoch = 1
        }
        return pinEpoch
    }

    private func lookupSlotLocked(expert: Int, pass: Int) -> Int? {
        let slot = expertToSlotTable[tableBase(forPass: pass) + expert]
        return slot >= 0 ? slot : nil
    }

    private func touchHitLocked(slot: Int) {
        lastUsedTick[slot] = tickCounter
        if policy == .sharedSLRU, !isProtectedSlot[slot], protectedCap > 0 {
            if protectedCount >= protectedCap {
                var demoteSlot = -1
                var demoteTick = UInt64.max
                for s in 0 ..< slotCapacity where isProtectedSlot[s] {
                    if lastUsedTick[s] < demoteTick {
                        demoteTick = lastUsedTick[s]
                        demoteSlot = s
                    }
                }
                if demoteSlot >= 0 {
                    isProtectedSlot[demoteSlot] = false
                    protectedCount -= 1
                }
            }
            isProtectedSlot[slot] = true
            protectedCount += 1
        }
    }

    private func chooseVictimSlotLocked(pass: Int, pinEpoch eph: UInt64) -> Int? {
        let range = slotRange(forPass: pass)
        var bestProbSlot = -1
        var bestProbTick = UInt64.max
        var bestAnySlot = -1
        var bestAnyTick = UInt64.max

        for slot in range {
            if slotPinnedEpoch[slot] == eph { continue }
            let oldExpert = slotToExpert[slot]
            if oldExpert == -1 {
                return slot
            }
            if oldExpert >= 0 && expertPinnedEpoch[oldExpert] == eph { continue }
            let tick = lastUsedTick[slot]
            if !isProtectedSlot[slot], tick < bestProbTick {
                bestProbTick = tick
                bestProbSlot = slot
            }
            if tick < bestAnyTick {
                bestAnyTick = tick
                bestAnySlot = slot
            }
        }

        if policy == .sharedSLRU, bestProbSlot >= 0 {
            return bestProbSlot
        }
        return bestAnySlot >= 0 ? bestAnySlot : nil
    }

    private func invalidateSlotLocked(_ slot: Int, pass: Int) {
        let oldExpert = slotToExpert[slot]
        if oldExpert >= 0 {
            expertToSlotTable[tableBase(forPass: pass) + oldExpert] = -1
        }
        slotToExpert[slot] = -1
        if isProtectedSlot[slot] {
            isProtectedSlot[slot] = false
            protectedCount -= 1
        }
    }

    private func commitLoadedSlotLocked(_ slot: Int, expert: Int, pass: Int) {
        slotToExpert[slot] = expert
        lastUsedTick[slot] = tickCounter
        expertToSlotTable[tableBase(forPass: pass) + expert] = slot
        if isProtectedSlot[slot] {
            isProtectedSlot[slot] = false
            protectedCount -= 1
        }
    }
}

// MARK: - Top-Level Streamer Coordinator

final class Inkling2ExpertStreamer: @unchecked Sendable {
    let config: Inkling2StreamingConfig
    private(set) var layers: [Inkling2LayerExpertCache] = []
    private let reader: Inkling2AlignedShardReader

    private let stagingLock = NSLock()
    private var stagingBuffer: Inkling2PrefillStagingBuffer?
    private var lastStagingReader: MLXArray?
    private var hasStagingAllocation = false

    private let statsLock = NSLock()
    private var totalRequests = 0
    private var cacheHits = 0
    private var cacheMisses = 0
    private var bytesRead: UInt64 = 0
    private var alignedBytesRead: UInt64 = 0
    private var ioNanoseconds: UInt64 = 0
    private var recordedError: Error?

    private(set) var residentSlotBytes: UInt64 = 0

    init(
        textConfig: Inkling2TextConfiguration,
        streamingConfig: Inkling2StreamingConfig,
        activation: @escaping @Sendable (MLXArray, MLXArray) -> MLXArray
    ) throws {
        self.config = streamingConfig
        let index = try Inkling2SafetensorsExpertIndex.parse(
            modelDirectory: streamingConfig.modelDirectory,
            textConfig: textConfig)
        let reader = try Inkling2AlignedShardReader(
            shardURLs: index.shardURLs,
            maxAlignedSliceBytes: index.maxAlignedSliceBytes,
            workerCount: streamingConfig.maxConcurrentReads,
            bypassPageCache: streamingConfig.bypassPageCache)
        self.reader = reader

        let slotBudgets = Inkling2StreamingConfig.slotsPerLayer(
            textConfig: textConfig,
            budgetSlots: streamingConfig.expertCacheSlots,
            loopWeighted: streamingConfig.loopWeightedSlots)
        let coreRange = textConfig.loop?.core
        let passes = textConfig.loop?.passes ?? 1
        let softcaps = MLXArray([textConfig.gateSoftcap, textConfig.upSoftcap])
        eval(softcaps)

        var builtLayers: [Inkling2LayerExpertCache] = []
        builtLayers.reserveCapacity(textConfig.hiddenLayers)
        var totalResident: UInt64 = 0

        for l in 0 ..< textConfig.hiddenLayers {
            let cache = try Inkling2LayerExpertCache(
                layer: l,
                slotCapacity: slotBudgets[l],
                passes: passes,
                isLoopCore: coreRange?.contains(l) ?? false,
                textConfig: textConfig,
                meta: index.layers[l],
                reader: reader,
                policy: streamingConfig.evictionPolicy,
                activation: activation,
                softcaps: softcaps,
                owner: self)
            totalResident += cache.residentBytes
            builtLayers.append(cache)
        }

        self.layers = builtLayers
        self.residentSlotBytes = totalResident

        if let firstLayer = builtLayers.first {
            try firstLayer.verifyZeroCopyAliasing()
        }
    }

    deinit {
        close()
    }

    func layerCache(_ layer: Int) -> Inkling2LayerExpertCache {
        layers[layer]
    }

    fileprivate func acquireStagingBuffer(
        for meta: Inkling2LayerFileMeta,
        numExperts: Int,
        slotCapacity: Int,
        activeExpertCount: Int
    ) throws -> Inkling2PrefillStagingBuffer {
        let readerToWait = stagingLock.withLock { () -> MLXArray? in
            let r = lastStagingReader
            lastStagingReader = nil
            return r
        }
        if let readerToWait {
            eval(readerToWait)
        }

        let desiredGeometry = Inkling2PrefillStagingGeometry(meta: meta)

        if let existing = stagingLock.withLock({ () -> Inkling2PrefillStagingBuffer? in
            guard let buf = stagingBuffer, buf.geometry == desiredGeometry else {
                stagingBuffer = nil
                hasStagingAllocation = false
                return nil
            }
            return buf
        }) {
            return existing
        }

        let perExpertBytes =
            meta.gateUp.weight.perExpertBytes
            + meta.gateUp.scales.perExpertBytes
            + (meta.gateUp.biases?.perExpertBytes ?? 0)
            + meta.down.weight.perExpertBytes
            + meta.down.scales.perExpertBytes
            + (meta.down.biases?.perExpertBytes ?? 0)
        let totalRAM = ProcessInfo.processInfo.physicalMemory
        let maxStagingBytes: UInt64 = if totalRAM >= 24 * 1024 * 1024 * 1024 {
            1_200_000_000
        } else if totalRAM >= 16 * 1024 * 1024 * 1024 {
            600_000_000
        } else {
            256_000_000
        }
        let ramBounded = Int(maxStagingBytes / max(1, perExpertBytes))
        let paddedActive = min(numExperts, ((activeExpertCount + 15) / 16) * 16 + 16)
        let ramCap = max(slotCapacity, ramBounded)
        let desiredSlots = min(numExperts, min(ramCap, max(slotCapacity, paddedActive)))

        let created = try Inkling2PrefillStagingBuffer(
            geometry: desiredGeometry,
            slotCount: desiredSlots,
            meta: meta)
        stagingLock.withLock {
            stagingBuffer = created
            hasStagingAllocation = true
        }
        return created
    }

    func setLastStagingReader(_ reader: MLXArray) {
        stagingLock.withLock {
            lastStagingReader = reader
        }
    }

    func releasePrefillStagingIfAllocated() {
        let needed = stagingLock.withLock { hasStagingAllocation || lastStagingReader != nil }
        if needed {
            releasePrefillStaging()
        }
    }

    func releasePrefillStaging() {
        let (readerToWait, released) = stagingLock.withLock { () -> (MLXArray?, Inkling2PrefillStagingBuffer?) in
            let r = lastStagingReader
            lastStagingReader = nil
            let buf = stagingBuffer
            stagingBuffer = nil
            hasStagingAllocation = false
            return (r, buf)
        }
        withExtendedLifetime(released) {
            if let readerToWait {
                eval(readerToWait)
            }
        }
    }

    func recordStats(
        requests: Int,
        hits: Int,
        misses: Int,
        rawBytes: UInt64,
        alignedBytes: UInt64,
        ioNanoseconds: UInt64
    ) {
        statsLock.withLock {
            totalRequests += requests
            cacheHits += hits
            cacheMisses += misses
            bytesRead += rawBytes
            alignedBytesRead += alignedBytes
            self.ioNanoseconds &+= ioNanoseconds
        }
    }

    var stats: Inkling2StreamingStats {
        statsLock.withLock {
            Inkling2StreamingStats(
                totalRequests: totalRequests,
                cacheHits: cacheHits,
                cacheMisses: cacheMisses,
                bytesRead: bytesRead,
                alignedBytesRead: alignedBytesRead,
                ioNanoseconds: ioNanoseconds,
                residentSlotBytes: residentSlotBytes)
        }
    }

    func recordError(_ error: Error) {
        statsLock.withLock {
            if recordedError == nil {
                recordedError = error
            }
        }
    }

    func consumeError() -> Error? {
        statsLock.withLock {
            let err = recordedError
            recordedError = nil
            return err
        }
    }

    func close() {
        releasePrefillStaging()
        for layerCache in layers {
            layerCache.close()
        }
        reader.close()
    }

    #if DEBUG
    func debugInjectIOFailure(onShardSubstring substring: String?) {
        reader.debugInjectFailureOnShardSubstring = substring
    }
    #endif
}

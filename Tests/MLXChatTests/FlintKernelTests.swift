import Foundation
import Metal
import MLX
import Testing
import TurboFieldfare
@testable import MLXChat

/// Contract: Flint's 8-bit embedding lookup matches MLX affine dequantization,
/// and decode attention with a learned sink matches the reference softmax.
@Suite(.serialized) struct FlintKernelTests {
    private static func buffer<T>(_ device: MTLDevice, _ values: [T]) throws -> MTLBuffer {
        try #require(values.withUnsafeBytes {
            device.makeBuffer(bytes: $0.baseAddress!, length: max($0.count, 4),
                              options: .storageModeShared)
        })
    }

    private static func run(_ context: MetalContext,
                            _ encode: (MTLComputeCommandEncoder) -> Void) throws {
        let command = try #require(context.queue.makeCommandBuffer())
        let encoder = try #require(command.makeComputeCommandEncoder())
        encode(encoder)
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        #expect(command.status == .completed)
    }

    @Test func q8EmbeddingLookupMatchesMLXDequantizedRow() throws {
        let rows = 4
        let dim = 256
        let values = (0..<rows * dim).map { sin(Float($0) * 0.017 + 0.2) }
        let weight = MLXArray(values, [rows, dim]).asType(.bfloat16)
        let (packed, scales, biases) = MLX.quantized(weight, groupSize: 64, bits: 8)
        let dense = MLX.dequantized(packed, scales: scales, biases: biases!, groupSize: 64, bits: 8)
            .asType(.float32)
            .asArray(Float.self)
        let context = try MetalContext()
        let pipeline = try context.pipeline("flint_embed_lookup_q8", privateModule: "flint")
        let device = context.device
        let out = try #require(device.makeBuffer(length: dim * 2, options: .storageModeShared))
        let weightBytes = packed.view(dtype: .uint8).asArray(UInt8.self)
        let scaleBits = scales.view(dtype: .uint16).asArray(UInt16.self)
        let biasBits = biases!.view(dtype: .uint16).asArray(UInt16.self)
        let weightBuffer = try Self.buffer(device, weightBytes)
        let scaleBuffer = try Self.buffer(device, scaleBits)
        let biasBuffer = try Self.buffer(device, biasBits)
        try Self.run(context) { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(weightBuffer, offset: 0, index: 0)
            encoder.setBuffer(scaleBuffer, offset: 0, index: 1)
            encoder.setBuffer(biasBuffer, offset: 0, index: 2)
            encoder.setBuffer(out, offset: 0, index: 3)
            var token: UInt32 = 2
            var width = UInt32(dim)
            encoder.setBytes(&token, length: 4, index: 4)
            encoder.setBytes(&width, length: 4, index: 5)
            encoder.dispatchThreads(MTLSize(width: dim, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        }
        let got = out.contents().assumingMemoryBound(to: Float16.self)
        for index in 0..<dim {
            let expected = dense[2 * dim + index]
            #expect(abs(Float(got[index]) - expected) <= 0.02 * max(1, abs(expected)))
        }
    }

    /// Split-KV attention with sinks against a CPU softmax: a short cache,
    /// and a long one that spans many chunks with a window start.
    /// `ring` stores position p in slot `p % ring`, as Flint's sliding layers do.
    @Test(arguments: [(5, 1, 0), (2_000, 1_487, 0), (2_000, 1_487, 600)])
    func attentionSinkMatchesReferenceSoftmax(length: Int, start: Int, ring: Int) throws {
        let queryHeads = 8
        let kvHeads = 2
        let dim = 128
        let group = queryHeads / kvHeads
        func wave(_ count: Int, _ phase: Float, _ rate: Float) -> [Float] {
            (0..<count).map { sin(Float($0) * rate + phase) }
        }
        let queries = wave(queryHeads * dim, 0.1, 0.05).map { Float16($0) }
        let keys = wave(length * kvHeads * dim, 0.4, 0.031).map { Float16($0) }
        let values = wave(length * kvHeads * dim, 0.8, 0.017).map { Float16($0) }
        let sinks = [Float(0.2), -0.5, 1.5, -1.0, 3.0, 0, -2.0, 0.7].map { value in
            Float(bitPattern: (value.bitPattern >> 16) << 16)
        }
        let sinkBits = sinks.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) }
        let context = try MetalContext()
        let attention = try FlintDecodeAttention(context: context, numQueryHeads: queryHeads)
        let device = context.device
        let out = try #require(device.makeBuffer(
            length: queryHeads * dim * 2, options: .storageModeShared))
        let scale = Float(1 / sqrt(128.0))
        // A ring holds only the latest position of each residue.
        func stored(_ logical: [Float16]) -> [Float16] {
            guard ring > 0 else { return logical }
            var slots = [Float16](repeating: .nan, count: ring * kvHeads * dim)
            let row = kvHeads * dim
            for position in 0..<length {
                let slot = position % ring
                slots[slot * row..<(slot + 1) * row] = logical[position * row..<(position + 1) * row]
            }
            return slots
        }
        let command = try #require(context.queue.makeCommandBuffer())
        attention.encode(commandBuffer: command,
                         q: try Self.buffer(device, queries),
                         k: try Self.buffer(device, stored(keys)),
                         v: try Self.buffer(device, stored(values)),
                         sinks: try Self.buffer(device, sinkBits), sinksOffset: 0,
                         out: out, numKVHeads: kvHeads, seqLen: length, kvStart: start,
                         scale: scale, ringCapacity: ring)
        command.commit()
        command.waitUntilCompleted()
        #expect(command.status == .completed)
        let got = out.contents().assumingMemoryBound(to: Float16.self)
        for head in 0..<queryHeads {
            let kvHead = head / group
            var scores: [Float] = []
            for position in start..<length {
                var dot: Float = 0
                for index in 0..<dim {
                    let q = Float(queries[head * dim + index])
                    let k = Float(keys[(position * kvHeads + kvHead) * dim + index])
                    dot += q * k
                }
                scores.append(dot * scale)
            }
            scores.append(sinks[head])
            let maxScore = scores.max() ?? 0
            let weights = scores.map { exp($0 - maxScore) }
            let normalizer = weights.reduce(0, +)
            for index in 0..<dim {
                var expected: Float = 0
                for (offset, position) in (start..<length).enumerated() {
                    let value = Float(values[(position * kvHeads + kvHead) * dim + index])
                    expected += weights[offset] / normalizer * value
                }
                let actual = Float(got[head * dim + index])
                #expect(abs(actual - expected) < 1e-3, "head \(head) dim \(index)")
            }
        }
    }

    /// The fused Q/K epilogue against per-head RMSNorm then NeoX rotary
    /// encoding on the CPU, with rotation on (sliding layers) and off (full).
    @Test(arguments: [true, false])
    func qkNormRopeMatchesSeparateNormThenRope(rotate: Bool) throws {
        let queryHeads = 4, kvHeads = 2, dim = 128, half = dim / 2
        let position: UInt32 = 37
        let theta: Float = 10_000, eps: Float = 1e-6
        func bf16(_ value: Float) -> UInt16 { UInt16(truncatingIfNeeded: value.bitPattern >> 16) }
        func float(_ bits: UInt16) -> Float { Float(bitPattern: UInt32(bits) << 16) }
        let q = (0..<queryHeads * dim).map { Float16(sin(Float($0) * 0.07) * 3) }
        let k = (0..<kvHeads * dim).map { Float16(cos(Float($0) * 0.05) * 2) }
        let qWeight = (0..<dim).map { bf16(1 + 0.3 * sin(Float($0) * 0.2)) }
        let kWeight = (0..<dim).map { bf16(0.8 + 0.2 * cos(Float($0) * 0.1)) }

        let context = try MetalContext()
        let pipeline = try context.pipeline("flint_qk_norm_rope", privateModule: "flint")
        let qBuffer = try Self.buffer(context.device, q)
        let kBuffer = try Self.buffer(context.device, k)
        let qwBuffer = try Self.buffer(context.device, qWeight)
        let kwBuffer = try Self.buffer(context.device, kWeight)
        try Self.run(context) { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(qBuffer, offset: 0, index: 0)
            encoder.setBuffer(kBuffer, offset: 0, index: 1)
            encoder.setBuffer(qwBuffer, offset: 0, index: 2)
            encoder.setBuffer(kwBuffer, offset: 0, index: 3)
            var heads = UInt32(queryHeads), at = position, base = theta
            var rotation: UInt32 = rotate ? 1 : 0, epsilon = eps
            encoder.setBytes(&heads, length: 4, index: 4)
            encoder.setBytes(&at, length: 4, index: 5)
            encoder.setBytes(&base, length: 4, index: 6)
            encoder.setBytes(&rotation, length: 4, index: 7)
            encoder.setBytes(&epsilon, length: 4, index: 8)
            encoder.dispatchThreadgroups(MTLSize(width: queryHeads + kvHeads, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: half, height: 1, depth: 1))
        }

        func expected(_ x: [Float16], _ weight: [UInt16]) -> [Float] {
            let values = x.map { Float($0) }
            let inv = 1 / (values.map { $0 * $0 }.reduce(0, +) / Float(dim) + eps).squareRoot()
            let normed = values.indices.map { Float(Float16(values[$0] * inv * float(weight[$0]))) }
            guard rotate else { return normed }
            var out = normed
            for pair in 0..<half {
                let angle = Float(position) * pow(theta, -Float(2 * pair) / Float(dim))
                out[pair] = normed[pair] * cos(angle) - normed[pair + half] * sin(angle)
                out[pair + half] = normed[pair] * sin(angle) + normed[pair + half] * cos(angle)
            }
            return out
        }
        for (buffer, source, weight, heads) in [(qBuffer, q, qWeight, queryHeads),
                                                (kBuffer, k, kWeight, kvHeads)] {
            let got = buffer.contents().assumingMemoryBound(to: Float16.self)
            for head in 0..<heads {
                let want = expected(Array(source[head * dim..<(head + 1) * dim]), weight)
                for index in 0..<dim {
                    let actual = Float(got[head * dim + index])
                    #expect(abs(actual - want[index]) <= 4e-3 * max(1, abs(want[index])),
                            "head \(head) dim \(index)")
                }
            }
        }
    }

    @Test func routerUsesBiasOnlyForSelection() {
        let hidden = [Float](repeating: 1, count: 2)
        // Rows are dot products 1, 2, 3, 0 against [1, 1] when the row sums are those.
        let weight: [Float] = [
            1, 0,
            0, 2,
            3, 0,
            0, 0,
        ]
        let bias: [Float] = [10, 0, 0, 0]
        let choice = FlintRouter.select(hidden: hidden, weight: weight, bias: bias,
                                        experts: 4, hiddenSize: 2, topK: 2)
        #expect(choice.indices == [0, 2])
        let scores = weight.enumerated().reduce(into: [Float](repeating: 0, count: 4)) { partial, item in
            if item.offset % 2 == 0 {
                let row = item.offset / 2
                let dot = weight[row * 2] + weight[row * 2 + 1]
                partial[row] = 1 / (1 + exp(-dot))
            }
        }
        let total = scores[0] + scores[2]
        #expect(abs(choice.weights[0] - scores[0] / total) < 1e-5)
        #expect(abs(choice.weights[1] - scores[2] / total) < 1e-5)
    }

    @Test func decodeRouterMatchesReferenceSelectionExactly() {
        var generator = SystemRandomNumberGenerator()
        let experts = 256, width = 64, topK = 6
        var scores = [Float](repeating: 0, count: experts)
        var indices = [Int](repeating: -1, count: topK)
        var weights = [Float](repeating: 0, count: topK)
        for trial in 0..<200 {
            let hidden = (0..<width).map { _ in Float.random(in: -1...1, using: &generator) }
            var weight = (0..<experts * width).map { _ in Float.random(in: -0.2...0.2, using: &generator) }
            var bias = (0..<experts).map { _ in Float.random(in: -0.05...0.05, using: &generator) }
            if trial % 3 == 0 {
                // Identical rows and biases tie on the biased score; lower index wins.
                for row in [9, 40, 41, 200] {
                    for column in 0..<width { weight[row * width + column] = weight[7 * width + column] }
                    bias[row] = bias[7] + 1
                }
            }
            if trial % 5 == 0 { bias = [Float](repeating: 0, count: experts) }
            let reference = FlintRouter.select(hidden: hidden, weight: weight, bias: bias,
                                               experts: experts, hiddenSize: width, topK: topK)
            FlintRouter.select(hidden: hidden, weight: weight, bias: bias,
                               experts: experts, hiddenSize: width, topK: topK,
                               scores: &scores, indices: &indices, weights: &weights)
            #expect(indices == reference.indices, "trial \(trial)")
            #expect(weights == reference.weights, "trial \(trial)")
        }
    }

    @Test func manifestReaderValidatesFlintAndLFM2Shapes() throws {
        let sha = String(repeating: "a", count: 64)
        func writeManifest(
            arch: ArchConfig,
            packedLayers: Int,
            embeddingBits: Int,
            routerBits: Int
        ) throws -> URL {
            let dir = FileManager.default.temporaryDirectory
                .appendingPathComponent("gturbo-manifest-test-\(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            var files: [String: [String: Any]] = [
                "model_weights.bin": ["size": 16_384, "sha256": sha],
                "packed_experts/layout.json": ["size": 1_024, "sha256": sha],
            ]
            for layer in 0..<packedLayers {
                let name = String(format: "packed_experts/layer_%02d.bin", layer)
                files[name] = ["size": 16_384, "sha256": sha]
            }
            func slot(_ bits: Int) -> [String: Any] {
                [
                    "weightBits": bits,
                    "scheme": "affine",
                    "scaleType": "BF16",
                    "biasType": "BF16",
                    "groupSize": 64,
                ]
            }
            let archDict: [String: Any] = [
                "hiddenSize": arch.hiddenSize,
                "ffnIntermediate": arch.intermediateSize,
                "moeIntermediateSize": arch.moeIntermediateSize,
                "numHeads": arch.numHeads,
                "numKVHeads": arch.numKVHeads,
                "numFullKVHeads": arch.numFullKVHeads,
                "headDim": arch.headDim,
                "fullHeadDim": arch.fullHeadDim,
                "vocabSize": arch.vocabSize,
                "slidingWindow": arch.slidingWindow,
                "finalLogitSoftcap": arch.finalLogitSoftcap,
                "ropeTheta": arch.ropeTheta,
                "fullRopeTheta": arch.fullRopeTheta,
                "partialRotaryFactor": arch.partialRotaryFactor,
                "numLayers": arch.numLayers,
                "numExperts": arch.numExperts,
                "topKExperts": arch.topKExperts,
                "tieWordEmbeddings": arch.tieWordEmbeddings,
                "attentionKEqV": arch.attentionKEqV,
                "hiddenActivation": arch.hiddenActivation,
                "fullAttentionLayerMask": arch.fullAttentionLayerMask.map(Int.init),
                "family": arch.family.rawValue,
                "attnOutputGate": arch.attnOutputGate,
                "attentionScale": arch.attentionScale,
                "embeddingScaledBySqrtHidden": arch.embeddingScaledBySqrtHidden,
                "routerScaled": arch.routerScaled,
                "ffnSandwichNorms": arch.ffnSandwichNorms,
                "sharedExpertGated": arch.sharedExpertGated,
                "ropeNeoxSubdim": arch.ropeNeoxSubdim,
                "linearNumKHeads": arch.linearAttention.numKHeads,
                "linearNumVHeads": arch.linearAttention.numVHeads,
                "linearKeyHeadDim": arch.linearAttention.keyHeadDim,
                "linearValueHeadDim": arch.linearAttention.valueHeadDim,
                "linearConvKernelSize": arch.linearAttention.convKernelSize,
            ]
            let manifestObj: [String: Any] = [
                "magic": "GTURBO",
                "versionMajor": 1,
                "versionMinor": 0,
                "flags": [
                    "streamingPresent": true,
                    "turboQuantKV": false,
                    "aneSharedExpert": false,
                ],
                "modelID": arch.family.rawValue,
                "sourceSnapshotHash": sha,
                "arch": archDict,
                "quant": [
                    "embedding": slot(embeddingBits),
                    "attention": slot(4),
                    "router": slot(routerBits),
                    "sharedExpert": slot(4),
                    "routedExpert": slot(4),
                ],
                "files": files,
                "expertsPerLayer": arch.numExperts,
                "numLayers": packedLayers,
                "expertStride": 16_384,
            ]
            let data = try JSONSerialization.data(withJSONObject: manifestObj, options: [.sortedKeys])
            try data.write(to: dir.appendingPathComponent("manifest.json"))
            return dir
        }

        let flintDir = try writeManifest(
            arch: .flint_40B_A2_5B,
            packedLayers: 47,
            embeddingBits: 8,
            routerBits: 16)
        let flintManifest = try ManifestReader.load(
            directoryURL: flintDir,
            expecting: .flint_40B_A2_5B)
        #expect(flintManifest.arch.numLayers == 49)
        #expect(flintManifest.numLayers == 47)
        #expect(flintManifest.arch.slidingWindow == 513)

        let lfmDir = try writeManifest(
            arch: .lfm25_24B_A2B,
            packedLayers: 38,
            embeddingBits: 6,
            routerBits: 32)
        let lfmManifest = try ManifestReader.load(
            directoryURL: lfmDir,
            expecting: .lfm25_24B_A2B)
        #expect(lfmManifest.arch.numLayers == 40)
        #expect(lfmManifest.numLayers == 38)
        #expect(lfmManifest.arch.slidingWindow == 0)
        #expect(lfmManifest.arch.fullAttentionLayerMask.contains(3))
    }

    @Test func flintFP16InfinityClampBitPatternMatchesGreatestFiniteMagnitude() {
        func clampBits(_ bits: UInt16) -> UInt16 {
            (bits & 0x7FFF) == 0x7C00 ? ((bits & 0x8000) | 0x7BFF) : bits
        }
        #expect(Float16(bitPattern: clampBits(Float16.infinity.bitPattern))
                == Float16.greatestFiniteMagnitude)
        #expect(Float16(bitPattern: clampBits((-Float16.infinity).bitPattern))
                == -Float16.greatestFiniteMagnitude)
        #expect(Float16(bitPattern: clampBits(Float16(42.5).bitPattern)) == Float16(42.5))
        #expect(Float16(bitPattern: clampBits(Float16(-0.125).bitPattern)) == Float16(-0.125))
        #expect(Float16(bitPattern: clampBits(Float16.nan.bitPattern)).isNaN)
    }

    @Test func gpuRouterMatchesReferenceSelection() throws {
        let experts = 256
        let hiddenSize = 2048
        let topK = 6

        func toBF16Bits(_ value: Float) -> UInt16 {
            UInt16(truncatingIfNeeded: value.bitPattern >> 16)
        }
        func fromBF16Bits(_ bits: UInt16) -> Float {
            Float(bitPattern: UInt32(bits) << 16)
        }

        let hiddenFP16 = (0..<hiddenSize).map { Float16(sin(Float($0) * 0.013 - 0.4) * 0.75) }
        let hiddenFP32 = hiddenFP16.map { Float($0) }
        let weightBF16 = (0..<experts * hiddenSize).map {
            toBF16Bits(cos(Float($0) * 0.007 + 0.15) * 0.15)
        }
        let weightFP32 = weightBF16.map(fromBF16Bits)
        let biasBF16 = (0..<experts).map {
            toBF16Bits(sin(Float($0) * 0.11) * 0.04)
        }
        let biasFP32 = biasBF16.map(fromBF16Bits)

        let reference = FlintRouter.select(
            hidden: hiddenFP32,
            weight: weightFP32,
            bias: biasFP32,
            experts: experts,
            hiddenSize: hiddenSize,
            topK: topK)

        let context = try MetalContext()
        let gemv = try context.pipeline("flint_router_gemv_bf16", privateModule: "flint")
        let device = context.device
        let hiddenBuf = try Self.buffer(device, hiddenFP16)
        let weightBuf = try Self.buffer(device, weightBF16)
        let scoresBuf = try #require(device.makeBuffer(length: experts * 4, options: .storageModeShared))
        try Self.run(context) { encoder in
            var hiddenU32 = UInt32(hiddenSize)
            var expertsU32 = UInt32(experts)
            encoder.setComputePipelineState(gemv)
            encoder.setBuffer(weightBuf, offset: 0, index: 0)
            encoder.setBuffer(hiddenBuf, offset: 0, index: 1)
            encoder.setBuffer(scoresBuf, offset: 0, index: 2)
            encoder.setBytes(&expertsU32, length: 4, index: 3)
            encoder.setBytes(&hiddenU32, length: 4, index: 4)
            encoder.dispatchThreadgroups(
                MTLSize(width: (experts + 7) / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }

        // The decode path: GPU sigmoid scores, CPU top-k.
        var indices = [Int](repeating: -1, count: topK)
        var weights = [Float](repeating: 0, count: topK)
        FlintRouter.select(scores: scoresBuf.contents().assumingMemoryBound(to: Float.self),
                           bias: biasFP32, experts: experts, topK: topK,
                           indices: &indices, weights: &weights)
        #expect(indices == reference.indices)
        for slot in 0..<topK {
            #expect(abs(weights[slot] - reference.weights[slot]) < 1e-5, "slot \(slot)")
        }
    }

    @Test(.timeLimit(.minutes(10)))
    func flintSSDSmokeGeneratesFromRealCheckpoint() async throws {
        guard ProcessInfo.processInfo.environment["MLXCHAT_RUN_FLINT_SSD_SMOKE"] == "1" else { return }
        let directory = StreamedMoEInstall.flint.managedDirectory
        try #require(FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("manifest.json").path),
            "MLXCHAT_RUN_FLINT_SSD_SMOKE=1 but checkpoint missing at \(directory.path)")

        let tokenizer = try await GFTokenizer.load(forModelDirectory: directory)
        #expect(tokenizer.dialect == .flint)
        #expect(tokenizer.thinkStartID == 150_006)
        #expect(tokenizer.thinkEndID == 150_007)
        #expect(tokenizer.decode([150_006, 150_007]) == "<think></think>")

        let catalogModel = try #require(
            ModelCatalog.model(for: StreamedMoEInstall.flint.modelID))
        let engine = StreamedMoEEngine(
            model: catalogModel,
            modelDirectory: directory,
            runtimeSettings: StreamedMoERuntimeSettings(
                contextTokens: 4_096,
                expertCacheSlots: 16))
        try await engine.load { _ in }
        defer { engine.unload() }

        var thinking = ""
        var content = ""
        let stream = engine.stream(
            history: [EngineMessage(role: .user, text: "How many r's in Strawberry?")],
            systemPrompt: "You are a helpful assistant.",
            temperature: 0.7,
            conversationID: UUID())
        for try await event in stream {
            switch event {
            case .thinkingDelta(let delta):
                thinking += delta
            case .delta(let delta):
                content += delta
            case .prefill, .thinkingReclassified, .completed:
                break
            }
            if thinking.count + content.count >= 64 {
                break
            }
        }

        #expect(!thinking.isEmpty || !content.isEmpty)
    }
}

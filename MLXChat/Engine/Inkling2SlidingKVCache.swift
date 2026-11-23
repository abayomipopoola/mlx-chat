// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Local sliding-window KV cache

/// Sliding-window ring cache for Inkling 2 that exposes `ringWriteIndex` and
/// `preservedPrefixTokens` in O(1) without relying on `package`-internal
/// properties of `MLXLMCommon.RotatingKVCache`.
public final class Inkling2SlidingKVCache: KVCache, CustomDebugStringConvertible {
    public var offset: Int = 0

    private var keep: Int
    private var keys: MLXArray?
    private var values: MLXArray?
    private var maxCacheSize: Int
    private var step: Int
    private var idx: Int = 0
    public private(set) var rewindCapacity: Int
    private var wrappedFlag: Bool? = false

    private var storageCapacity: Int { maxCacheSize + rewindCapacity }

    private var wrapped: Bool {
        wrappedFlag ?? (idx < (keys?.dim(2) ?? 0) && offset > idx)
    }

    public var maxSize: Int? { maxCacheSize }

    public var evictedTokenCount: Int {
        let retained = wrapped ? (keys?.dim(2) ?? 0) : idx
        return Swift.max(0, offset - retained)
    }

    /// Number of leading pinned tokens that never rotate out of the window.
    public var preservedPrefixTokens: Int { keep }

    /// The slot after the newest key when the last `update(keys:values:)` returned
    /// the whole ring out of temporal order, or `nil` when keys are oldest-first.
    public var ringWriteIndex: Int? {
        guard rewindCapacity == 0, wrapped, let keys, idx < keys.dim(2) else {
            return nil
        }
        return idx
    }

    public init(maxSize: Int, keep: Int = 0, step: Int = 256, rewindCapacity: Int = 0) {
        precondition(rewindCapacity >= 0 && maxSize <= Int.max - rewindCapacity)
        self.maxCacheSize = maxSize
        self.keep = keep
        self.step = step
        self.rewindCapacity = rewindCapacity
    }

    public func innerState() -> [MLXArray] {
        [keys, values].compactMap { $0 }
    }

    private func trimBuffer(trimSize: Int, _ array: MLXArray, append: MLXArray? = nil) -> MLXArray {
        var toCat: [MLXArray]
        if trimSize > 0 {
            toCat = [
                array[.ellipsis, ..<keep, 0...],
                array[.ellipsis, (trimSize + keep)..., 0...],
            ]
        } else {
            toCat = [array]
        }
        if let append {
            toCat.append(append)
        }
        return concatenated(toCat, axis: 2)
    }

    private func temporalOrder(_ array: MLXArray) -> MLXArray {
        if idx == array.dim(2) {
            return array
        } else if wrapped {
            return concatenated(
                [
                    array[.ellipsis, ..<keep, 0...],
                    array[.ellipsis, idx..., 0...],
                    array[.ellipsis, keep ..< idx, 0...],
                ], axis: 2)
        } else {
            return array[.ellipsis, ..<idx, 0...]
        }
    }

    private func retainedView(tail: Int) -> (MLXArray, MLXArray)? {
        guard let keys = self.keys, let values = self.values else { return nil }

        let available = wrapped ? keys.dim(2) : idx
        let prefix = Swift.min(keep, available)
        let requested = Swift.min(Swift.max(tail, 0), available)
        let bound = Swift.max(requested, prefix)

        if !wrapped || idx == available {
            if prefix == 0 || bound == available {
                return (
                    keys[.ellipsis, (available - bound) ..< available, 0...],
                    values[.ellipsis, (available - bound) ..< available, 0...]
                )
            }
        } else if prefix == 0 && bound <= idx {
            return (
                keys[.ellipsis, (idx - bound) ..< idx, 0...],
                values[.ellipsis, (idx - bound) ..< idx, 0...]
            )
        }

        let recent = Swift.min(bound - prefix, idx - prefix)
        let older = bound - prefix - recent
        let ranges = [0 ..< prefix, (available - older) ..< available, (idx - recent) ..< idx]
            .filter { !$0.isEmpty }
        func view(_ array: MLXArray) -> MLXArray {
            let parts = ranges.map { array[.ellipsis, $0, 0...] }
            return parts.count == 1 ? parts[0] : concatenated(parts, axis: 2)
        }
        return (view(keys), view(values))
    }

    public func currentKeyValues() -> (keys: MLXArray, values: MLXArray)? {
        retainedView(tail: rewindCapacity > 0 ? maxCacheSize : .max)
    }

    private func updateConcat(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        if self.keys == nil {
            self.keys = keys
            self.values = values
        } else {
            self.keys = temporalOrder(self.keys!)
            self.values = temporalOrder(self.values!)
            idx = self.keys!.dim(2)

            let trimSize = idx - storageCapacity + 1
            self.keys = trimBuffer(trimSize: trimSize, self.keys!, append: keys)
            self.values = trimBuffer(trimSize: trimSize, self.values!, append: values)
        }

        offset += keys.dim(2)
        idx = self.keys!.dim(2)
        wrappedFlag = false
        return (self.keys!, self.values!)
    }

    private func updateInPlace(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let b = keys.dim(0)
        let nKVHeads = keys.dim(1)
        let s = keys.dim(2)
        let kHeadDim = keys.dim(3)
        let vHeadDim = values.dim(3)

        let filled = self.keys?.dim(2) ?? 0
        if self.keys == nil || (!wrapped && idx >= filled && filled < storageCapacity) {
            let newSize = min(step, storageCapacity - filled)
            let newK = MLXArray.zeros([b, nKVHeads, newSize, kHeadDim], dtype: keys.dtype)
            let newV = MLXArray.zeros([b, nKVHeads, newSize, vHeadDim], dtype: values.dtype)

            if let currentKeys = self.keys, let currentValues = self.values {
                self.keys = concatenated([currentKeys, newK], axis: 2)
                self.values = concatenated([currentValues, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
        }

        let trimSize = self.keys!.dim(2) - storageCapacity
        if trimSize > 0 {
            self.keys = trimBuffer(trimSize: trimSize, self.keys!)
            self.values = trimBuffer(trimSize: trimSize, self.values!)
            idx = storageCapacity
        }

        if idx == storageCapacity {
            idx = keep
            wrappedFlag = true
        }

        self.keys![.ellipsis, idx ..< (idx + s), 0...] = keys
        self.values![.ellipsis, idx ..< (idx + s), 0...] = values
        offset += s
        idx += s

        if !wrapped, idx < self.keys!.dim(2) {
            return (
                self.keys![.ellipsis, ..<idx, 0...],
                self.values![.ellipsis, ..<idx, 0...]
            )
        }
        return (self.keys!, self.values!)
    }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        wrappedFlag = wrapped
        let count = keys.dim(2)
        if rewindCapacity > 0, count == 0 {
            return retainedView(tail: maxCacheSize) ?? (keys, values)
        }
        let live = wrapped ? (self.keys?.dim(2) ?? 0) : idx
        let result =
            count == 1
            ? updateInPlace(keys: keys, values: values)
            : updateConcat(keys: keys, values: values)
        if rewindCapacity > 0 {
            return retainedView(tail: Swift.min(live, maxCacheSize - 1) + count)!
        }
        return result
    }

    public var state: [MLXArray] {
        get {
            guard let keys = self.keys, let values = self.values else { return [] }
            if !wrapped, idx < keys.dim(2) {
                return [
                    keys[.ellipsis, ..<idx, 0...],
                    values[.ellipsis, ..<idx, 0...],
                ]
            }
            return [keys, values]
        }
        set {
            guard newValue.count == 2 else {
                fatalError("Inkling2SlidingKVCache state must have exactly 2 arrays")
            }
            self.keys = newValue[0]
            self.values = newValue[1]
        }
    }

    public var metaState: [String] {
        get {
            var metadata = [
                String(keep), String(maxCacheSize), String(step), String(offset), String(idx),
                "modelNative", String(wrapped),
            ]
            if rewindCapacity > 0 { metadata.append(String(rewindCapacity)) }
            return metadata
        }
        set {
            guard (5 ... 8).contains(newValue.count),
                let keepVal = Int(newValue[0]),
                let maxSizeVal = Int(newValue[1]),
                let stepVal = Int(newValue[2]),
                let offsetVal = Int(newValue[3]),
                let idxVal = Int(newValue[4])
            else {
                fatalError("Invalid Inkling2SlidingKVCache metaState")
            }
            self.keep = keepVal
            self.maxCacheSize = maxSizeVal
            self.step = stepVal
            self.offset = offsetVal
            self.idx = idxVal
            if newValue.count >= 7, let w = Bool(newValue[6]) {
                self.wrappedFlag = w
            } else {
                self.wrappedFlag = nil
            }
            if newValue.count == 8, let reserve = Int(newValue[7]), reserve >= 0 {
                self.rewindCapacity = reserve
            } else {
                self.rewindCapacity = 0
            }
        }
    }

    public var isTrimmable: Bool { isTrimmable(after: 0) }

    public var maxTrimCount: Int {
        let live = wrapped ? (keys?.dim(2) ?? 0) : idx
        guard live <= offset else { return 0 }
        return live == offset ? offset : Swift.max(0, live - maxCacheSize)
    }

    public func isTrimmable(after positions: Int) -> Bool {
        let live = wrapped ? (keys?.dim(2) ?? 0) : idx
        guard keys == nil || live == offset else { return false }
        return positions >= 0 && offset >= 0 && offset < storageCapacity
            && positions < storageCapacity - offset
    }

    @discardableResult
    public func trim(_ n: Int) -> Int {
        guard n > 0, let keys, let values else { return 0 }
        wrappedFlag = wrapped
        let live = wrapped ? keys.dim(2) : idx
        let minimum = offset > live ? Swift.min(keep, live) : 0
        let trimmed = Swift.min(n, live - minimum)
        guard trimmed > 0 else { return 0 }
        let bound = live - trimmed

        if wrapped || keys.dim(2) > storageCapacity {
            self.keys = temporalOrder(keys)[.ellipsis, ..<bound, 0...]
            self.values = temporalOrder(values)[.ellipsis, ..<bound, 0...]
        }
        idx = bound
        offset -= trimmed
        wrappedFlag = false
        return trimmed
    }

    public func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n > 1 {
            let actualWindowSize = windowSize ?? maxCacheSize
            let liveRows = wrapped ? maxCacheSize : idx
            let cappedOffset = min(maxCacheSize - 1, liveRows)
            if cappedOffset + n > actualWindowSize || returnArray {
                return .array(
                    createCausalMask(n: n, offset: cappedOffset, windowSize: actualWindowSize))
            }
            return .causal
        }
        guard let windowSize else { return .none }
        if rewindCapacity > 0 {
            let live = wrapped ? storageCapacity : idx
            let count = Swift.min(maxCacheSize - 1, live) + 1
            guard count > windowSize else { return .none }
            return .array(MLXArray(0 ..< count) .>= (count - windowSize))
        }
        if offset >= windowSize, maxCacheSize > windowSize {
            var currentIdx = idx
            if currentIdx >= maxCacheSize { currentIdx = 0 }
            let maskSize = (!wrapped && idx < maxCacheSize) ? idx + 1 : maxCacheSize
            let mask = MLXArray(0 ..< Int32(maskSize)) .>= Int32(maskSize - windowSize)
            return .array(roll(mask, shift: currentIdx + 1))
        }
        return .none
    }

    public func copy() -> any KVCache {
        let new = Inkling2SlidingKVCache(
            maxSize: maxCacheSize, keep: keep, step: step, rewindCapacity: rewindCapacity)
        let s = self.state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        new.metaState = self.metaState
        return new
    }

    public func prepare(lengths: [Int]?) {}
    public func prepare(lengths: MLXArray?) {}
    public func finalize() {}

    public var debugDescription: String {
        "Inkling2SlidingKVCache offset: \(offset), maxSize: \(maxCacheSize), keep: \(keep), idx: \(idx), wrapped: \(wrapped)"
    }
}

/// Reads `(ringIndex, keep)` for either the zero-allocation `Inkling2SlidingKVCache`
/// or a stock `MLXLMCommon.RotatingKVCache` (via its public `metaState`).
func inkling2RingState(for cache: KVCache?, keyLength: Int) -> (ringIndex: Int?, keep: Int) {
    if let sliding = cache as? Inkling2SlidingKVCache {
        return (sliding.ringWriteIndex, sliding.preservedPrefixTokens)
    }
    guard let rotating = cache as? RotatingKVCache else {
        return (nil, 0)
    }
    let meta = rotating.metaState
    guard meta.count >= 5, let keep = Int(meta[0]), let idx = Int(meta[4]) else {
        return (nil, 0)
    }
    let wrapped: Bool
    if meta.count >= 7, let flag = Bool(meta[6]) {
        wrapped = flag
    } else {
        wrapped = idx < keyLength && rotating.offset > idx
    }
    let ringIndex: Int? = (rotating.rewindCapacity == 0 && wrapped && idx < keyLength) ? idx : nil
    return (ringIndex, keep)
}

// MARK: - Local fused quantized linear projection cache

/// Local copy of `MLXLMCommon`'s package-internal `FusedQuantizedLinearProjectionCache`.
final class Inkling2FusedQuantizedLinearProjectionCache {
    private enum State {
        case unprepared
        case preparing
        case ready
        case ineligible
    }

    private var state = State.unprepared
    private(set) var fused: QuantizedLinear?

    func invalidate() {
        guard state != .preparing else { return }
        fused = nil
        state = .unprepared
    }

    @discardableResult
    func prepare(
        enabled: Bool,
        linears: [Linear],
        installSourceModules: ([Linear]) throws -> Void
    ) throws -> Bool {
        guard enabled else { return false }

        switch state {
        case .ready:
            return fused != nil
        case .preparing, .ineligible:
            return false
        case .unprepared:
            break
        }

        state = .preparing
        guard let projection = Self.fuse(linears),
            projection.sourceViews.count == linears.count
        else {
            state = .ineligible
            return false
        }

        do {
            try installSourceModules(projection.sourceViews)
        } catch let installationError {
            _ = try? installSourceModules(linears)
            fused = nil
            state = .ineligible
            throw installationError
        }

        fused = projection.fused
        state = .ready
        return true
    }

    private static func fuse(
        _ linears: [Linear]
    ) -> (fused: QuantizedLinear, sourceViews: [QuantizedLinear])? {
        guard linears.count > 1 else { return nil }

        let projections = linears.compactMap { $0 as? QuantizedLinear }
        guard projections.count == linears.count,
            zip(linears, projections).allSatisfy({ linear, projection in
                ObjectIdentifier(type(of: linear)) == ObjectIdentifier(QuantizedLinear.self)
                    && linear === projection
            }),
            let first = projections.first,
            first.bias == nil,
            first.globalScale == nil,
            first.weight.ndim == 2,
            first.weight.dim(0) == first.shape.0,
            first.scales.ndim == 2,
            first.scales.dim(0) == first.shape.0,
            first.biases == nil || first.biases?.shape == first.scales.shape
        else {
            return nil
        }

        let hasBiases = first.biases != nil
        guard
            projections.allSatisfy({ projection in
                projection.bias == nil
                    && projection.globalScale == nil
                    && projection.bits == first.bits
                    && projection.groupSize == first.groupSize
                    && projection.mode == first.mode
                    && projection.shape.1 == first.shape.1
                    && projection.weight.ndim == 2
                    && projection.weight.dim(0) == projection.shape.0
                    && projection.weight.dim(1) == first.weight.dim(1)
                    && projection.weight.dtype == first.weight.dtype
                    && projection.scales.ndim == 2
                    && projection.scales.dim(0) == projection.shape.0
                    && projection.scales.dim(1) == first.scales.dim(1)
                    && projection.scales.dtype == first.scales.dtype
                    && (projection.biases != nil) == hasBiases
                    && (projection.biases == nil || projection.biases?.shape == projection.scales.shape)
                    && projection.biases?.dtype == first.biases?.dtype
            })
        else {
            return nil
        }

        let fusedWeight = concatenated(projections.map(\.weight), axis: 0)
        let fusedScales = concatenated(projections.map(\.scales), axis: 0)
        let fusedBiases = hasBiases ? concatenated(projections.compactMap(\.biases), axis: 0) : nil

        eval(fusedWeight, fusedScales)
        if let fusedBiases { eval(fusedBiases) }

        let fused = QuantizedLinear(
            weight: fusedWeight, bias: nil, scales: fusedScales, biases: fusedBiases,
            groupSize: first.groupSize, bits: first.bits, mode: first.mode)
        fused.freeze()

        var start = 0
        let views = projections.map { projection in
            let end = start + projection.shape.0
            defer { start = end }
            let rows = start ..< end
            let view = QuantizedLinear(
                weight: fusedWeight[rows], bias: nil, scales: fusedScales[rows],
                biases: fusedBiases.map { $0[rows] }, groupSize: first.groupSize,
                bits: first.bits, mode: first.mode)
            view.freeze()
            return view
        }
        return (fused, views)
    }
}

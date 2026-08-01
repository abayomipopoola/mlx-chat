import Foundation
import SwiftUI

/// Central registry of UserDefaults keys and their defaults.
enum Keys {
    static let selectedModelID = "selectedModelID"
    static let personalizationEnabled = "personalization.enabled"
    static let personalizationInstructions = "personalization.instructions"
    static let personalizationTemperature = "personalization.temperature"
    static let promptPreset = "prompt.preset"
    static let autoUnloadMinutes = "autoUnloadMinutes"
    static let appearance = "appearance"
    static let sidebarCollapsed = "sidebarCollapsed"
    static let thinkingEnabled = "thinking.enabled"
    static let downloadedModels = "downloadedModels"
    static let customModels = "customModels"
    static let streamedModelBookmarks = "streamedModelBookmarks"
    static let managedStreamedModelEnabled = "streamedModel.managed.enabled"
    static let managedStreamedModelPreferences = "streamedModel.managed.preferences"
    static let streamedModelContextTokens = "streamedModel.runtime.contextTokens"
    static let streamedModelExpertCacheSlots = "streamedModel.runtime.expertCacheSlots"

    enum Defaults {
        /// Apple Intelligence: the only engine guaranteed present on a fresh
        /// install. A local model becomes the selection only when the
        /// user picks it, and that choice persists across launches.
        static let selectedModelID = appleIntelligenceEngineID
        static let personalizationEnabled = true
        static let personalizationInstructions = ""
        static let personalizationTemperature = 0.4
        static let autoUnloadMinutes = 10
        /// "system" | "light" | "dark"
        static let appearance = "system"
        static let sidebarCollapsed = false
        static let thinkingEnabled = true
        static let streamedModelContextTokens = 4_096
        static let streamedModelExpertCacheSlots = 16
    }
}

/// Engine id for the built-in Apple Intelligence engine.
let appleIntelligenceEngineID = "apple-intelligence"

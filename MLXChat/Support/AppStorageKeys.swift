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

    enum Defaults {
        /// Apple Intelligence: the only engine guaranteed present on a fresh
        /// install. A downloaded model becomes the selection only when the
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
    }
}

/// Engine id for the built-in Apple Intelligence engine.
let appleIntelligenceEngineID = "apple-intelligence"

import SwiftUI

/// Brief full-window brand splash at launch, cross-fading into the main layout
/// (mirrors MLX Studio's root ZStack of ContentView + SplashView).
struct SplashView: View {
    var body: some View {
        ZStack {
            Color.detailBackground
                .ignoresSafeArea()
            VStack(spacing: 18) {
                Image("SplashLogo")
                    .resizable()
                    .scaledToFit()
                    .frame(width: 130, height: 130)
                    .shadow(color: .black.opacity(0.25), radius: 14, y: 6)
                Text("MLX Chat")
                    .font(.system(size: 28, weight: .semibold))
            }
        }
    }
}

import Testing
@testable import MLXChat

@Suite struct WindowNavigationTests {
    @Test func pushClearsForwardAndEnablesBack() {
        let navigation = WindowNavigation()
        navigation.push(.settings)
        #expect(navigation.path == [.settings])
        #expect(navigation.canGoBack)
        #expect(!navigation.canGoForward)

        navigation.push(.models)
        navigation.goBack()
        #expect(navigation.current == .settings)
        #expect(navigation.canGoForward)

        navigation.push(.models)
        #expect(navigation.forwardStack.isEmpty)
        #expect(!navigation.canGoForward)
    }

    @Test func duplicatePushIsIgnored() {
        let navigation = WindowNavigation()
        navigation.push(.settings)
        navigation.push(.settings)
        #expect(navigation.path == [.settings])
    }

    @Test func familyIsARouteOnTheSameStack() {
        let navigation = WindowNavigation()
        navigation.push(.settings)
        navigation.push(.models)
        navigation.push(.modelFamily("gemma"))
        #expect(navigation.familyID == "gemma")
        #expect(navigation.isOnModels)

        navigation.goBack()
        #expect(navigation.current == .models)
        #expect(navigation.familyID == nil)
        #expect(navigation.canGoForward)

        navigation.goBack()
        #expect(navigation.current == .settings)
        #expect(!navigation.isOnModels)
    }

    @Test func goHomeClearsTheStack() {
        let navigation = WindowNavigation()
        navigation.push(.settings)
        navigation.push(.models)
        navigation.goBack()
        navigation.goHome()
        #expect(navigation.path.isEmpty)
        #expect(navigation.forwardStack.isEmpty)
        #expect(!navigation.canGoBack)
        #expect(!navigation.canGoForward)
    }

    @Test func goBackOnEmptyStackIsANoOp() {
        let navigation = WindowNavigation()
        navigation.goBack()
        navigation.goForward()
        #expect(navigation.path.isEmpty)
        #expect(navigation.forwardStack.isEmpty)
    }
}

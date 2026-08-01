# Releasing MLX Chat (with Sparkle auto-update)

MLX Chat auto-updates via [Sparkle](https://sparkle-project.org), fed by a
GitHub Releases **appcast**. The app's feed is fixed to:

```
https://github.com/abayomipopoola/mlx-chat/releases/latest/download/appcast.xml
```

`…/releases/latest/download/<asset>` always resolves to the newest published
release's asset, so each release just needs the DMG **and** `appcast.xml` attached.

The Sparkle CLI tools live in the resolved SPM artifact:

```sh
SPARKLE_BIN=".build/dd/SourcePackages/artifacts/sparkle/Sparkle/bin"
```

> ⚠️ **First working Sparkle build:** installs through 1.0.1 cannot complete an
> automatic update (1.0.0 has no updater; 1.0.1 lacks the sandbox installer-launcher
> key). Those users must install 1.0.2 once by hand. Releases after 1.0.2 update in place.

---

## One-time setup: signing keys

Sparkle verifies updates with an EdDSA key pair. The private key is stored in
your login Keychain; the public key is embedded in the app.

```sh
"$SPARKLE_BIN/generate_keys"
```

Copy the printed public key into `project.yml` under the target's
`info.properties.SUPublicEDKey` (replacing `PASTE_SPARKLE_ED_PUBLIC_KEY_HERE`),
then regenerate:

```sh
xcodegen generate
```

Back up the private key once: `"$SPARKLE_BIN/generate_keys" -x sparkle_private_key.pem`
and store it somewhere safe (a lost key means you can't sign future updates).

---

## Per release

1. **Bump both versions** in `project.yml`: set `MARKETING_VERSION` to the user-facing
   release and increment `CURRENT_PROJECT_VERSION` monotonically for Sparkle, then run
   `xcodegen generate`. Sparkle uses `CFBundleVersion`; changing only the marketing
   version will not offer an update.

2. **Build, notarize, and staple the DMG** as usual (`scripts/release.sh`). You
   should end up with `dist/MLX-Chat-<version>.dmg`.

3. **Generate the signed appcast.** Point `generate_appcast` at a folder that
   contains the DMG; it signs each update with your private key and writes
   `appcast.xml`. Use `--download-url-prefix` so the enclosure URLs point at the
   GitHub release assets:

   ```sh
   "$SPARKLE_BIN/generate_appcast" dist/ \
     --download-url-prefix "https://github.com/abayomipopoola/mlx-chat/releases/download/v<version>/"
   ```

   This produces `dist/appcast.xml`. Add release notes by putting a
   `MLX-Chat-<version>.html` next to the DMG before running it (Sparkle shows it
   in the update dialog).

4. **Publish the GitHub Release.** Tag it `v<version>`, then attach **both**
   `MLX-Chat-<version>.dmg` and `appcast.xml`:

   ```sh
   gh release create v<version> \
     dist/MLX-Chat-<version>.dmg \
     dist/appcast.xml \
     --title "MLX Chat <version>" --notes "…"
   ```

That's it. Running apps poll `latest/download/appcast.xml`, see the new version,
and show the in-app banner + "Update Now" (or the user picks **MLX Chat → Check
for Updates…**). Sparkle downloads the DMG, verifies the signature, swaps the app
in place, and relaunches.

---

## How the app side is wired

- `MLXChat/App/UpdateController.swift` — wraps Sparkle. Probes silently on launch
  (`checkForUpdateInformation`) to drive the banner; `checkForUpdates()` runs the
  full install flow. Stays dormant until `SUPublicEDKey` is a real key.
- `SUFeedURL`, `SUPublicEDKey`, and `SUEnableInstallerLauncherService: true` —
  `project.yml` → `info.properties` (the generated `MLXChat/Support/Info.plist`). The
  feed is also set in code via `feedURLString(for:)` as a fallback.
- Sandbox: `SUEnableInstallerLauncherService` enables Sparkle's installer XPC, while
  `MLXChat.entitlements` grants its `…-spks` / `…-spki` mach-lookup names. Both are
  required; without the Info.plist key installation fails with authorization error
  `-60005` after a successful download.
- Signing: the embedded `Sparkle.framework` (and its XPC services) must be signed
  and notarized along with the app — the Release build + notarization pipeline
  covers this because it signs nested code.

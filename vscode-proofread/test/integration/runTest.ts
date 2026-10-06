import * as fs from "fs";
import * as path from "path";
import * as os from "os";
import { downloadAndUnzipVSCode, runTests } from "@vscode/test-electron";

(async () => {
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), "proofread-vscode-"));
  try {
    let exe = await downloadAndUnzipVSCode();
    // Recent macOS builds name the binary "Code"; @vscode/test-electron 2.5.x still points at "Electron".
    const alt = path.join(path.dirname(exe), "Code");
    if (!fs.existsSync(exe) && fs.existsSync(alt)) exe = alt;
    await runTests({
      vscodeExecutablePath: exe,
      extensionDevelopmentPath: path.resolve(__dirname, "..", "..", ".."),
      extensionTestsPath: path.resolve(__dirname, "suite"),
      launchArgs: [`--user-data-dir=${profile}`, "--disable-extensions", "--skip-welcome", "--skip-release-notes"],
    });
  } catch (e) { console.error("integration test failed", e); process.exit(1); }
  finally { fs.rmSync(profile, { recursive: true, force: true }); }
})();

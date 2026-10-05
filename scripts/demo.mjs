#!/usr/bin/env node
import {spawnSync} from 'node:child_process';
import {mkdirSync, copyFileSync, existsSync, writeFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import path from 'node:path';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = new Set(process.argv.slice(2));
const supported = new Set(['--doctor', '--prepare-only']);
for (const arg of args) if (!supported.has(arg)) {
  console.error(`Unknown option ${arg}. Use --doctor or --prepare-only.`); process.exit(1);
}
function run(command, argv, cwd = root) {
  const result = spawnSync(command, argv, {cwd, stdio: 'inherit', shell: process.platform === 'win32'});
  if (result.error || result.status !== 0) {
    console.error(`Could not run ${command}. ${result.error?.message ?? 'See the output above.'}`);
    process.exit(result.status || 1);
  }
}
function available(command) {
  const result = spawnSync(command, ['--version'], {stdio: 'ignore', shell: process.platform === 'win32'});
  return !result.error && result.status === 0;
}
if (Number(process.versions.node.split('.')[0]) < 22) {
  console.error('TabLint setup needs Node.js 22 or newer: https://nodejs.org/'); process.exit(1);
}
const npm = available('npm'); const code = available('code');
if (args.has('--doctor')) {
  console.log(`Node.js: ${process.versions.node}
npm: ${npm ? 'ready' : 'missing'}
VS Code CLI: ${code ? 'ready' : 'missing — install VS Code and add code to PATH'}`);
  process.exit(npm && code ? 0 : 1);
}
if (!npm || (!code && !args.has('--prepare-only'))) {
  console.error('Install Node.js 22+ and VS Code. On macOS, use “Shell Command: Install code command in PATH” in VS Code. Run npm run doctor to check again.'); process.exit(1);
}
console.log('Building TabLint for VS Code…');
run('npm', ['ci'], path.join(root, 'vscode-proofread'));
mkdirSync(path.join(root, 'dist'), {recursive: true});
run('npm', ['run', 'package'], path.join(root, 'vscode-proofread'));
const workspace = path.join(root, 'demo', '.workspace');
mkdirSync(workspace, {recursive: true});
for (const name of ['auto_mpg_dirty.csv', 'auto_mpg_dirty.proofread.json']) {
  const target = path.join(workspace, name);
  if (!existsSync(target)) copyFileSync(path.join(root, 'demo', 'video', name), target);
}
const workspaceFile = path.join(workspace, 'TabLint-demo.code-workspace');
writeFileSync(workspaceFile, JSON.stringify({folders: [{path: '../..'}], settings: {'workbench.colorTheme': 'Default Light Modern'}}, null, 2) + '\n');
console.log('Demo ready: saved TabPFN report, no Python, model download or GPU needed.');
if (args.has('--prepare-only')) {
  console.log(`Extension: ${path.join(root, 'dist', 'tablint.vsix')}
Workspace: ${workspaceFile}`);
} else {
  run('code', ['--install-extension', path.join(root, 'dist', 'tablint.vsix'), '--force']);
  run('code', ['--new-window', workspaceFile, '--goto', `${path.join(workspace, 'auto_mpg_dirty.csv')}:183:48`]);
  console.log('Hover 4354.0 on the Civic row. Quick Fix: Ctrl+. / Cmd+. Then run “TabLint: Open decision ledger”.');
}

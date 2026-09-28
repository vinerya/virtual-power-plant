// Copy Monaco's AMD build into public/ so the settings editor is served from
// our own origin instead of cdn.jsdelivr.net (the @monaco-editor/react
// default). Keeps the operator console working offline / behind strict CSP
// and in air-gapped control rooms. Runs before `dev` and `build`.
import { cpSync, existsSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const pkgDir = dirname(require.resolve("monaco-editor/package.json"));
const version = JSON.parse(readFileSync(join(pkgDir, "package.json"), "utf8")).version;

const dest = join(root, "public", "monaco");
const stamp = join(dest, ".version");

if (existsSync(stamp) && readFileSync(stamp, "utf8") === version) {
  process.exit(0);
}

rmSync(dest, { recursive: true, force: true });
cpSync(join(pkgDir, "min", "vs"), join(dest, "vs"), { recursive: true });
writeFileSync(stamp, version);
console.log(`[copy-monaco] monaco-editor ${version} -> public/monaco/vs`);

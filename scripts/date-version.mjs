#!/usr/bin/env node

/**
 * Set every workspace to a date-based version: `YYYY.M.D-N`.
 *
 * This fork tracks upstream by pulling `upstream/main` when we want it, not by
 * following upstream's release cadence. An inherited `0.7.1` says nothing about
 * which upstream state we are on; a date says exactly when we last took one,
 * and `-N` separates two builds on the same day.
 *
 * It is valid semver, so npm, the caret ranges `sync-versions.js` writes, and
 * `packages/coding-agent`'s own update check all keep working. `-N` is a
 * prerelease tag, which means `2026.8.10-1` sorts before a bare `2026.8.10`;
 * we never publish a bare one, and within the scheme the ordering is right:
 * `2026.8.10-1` < `2026.8.10-2` < `2026.8.11-1`.
 *
 * Usage:
 *   node scripts/date-version.mjs              # today, next free -N
 *   node scripts/date-version.mjs 2026.8.10-3  # exactly this
 */

import { execFileSync } from "node:child_process";
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));

/** Local date, not UTC: the version names the day the person made the build. */
function today() {
	const now = new Date();
	return `${now.getFullYear()}.${now.getMonth() + 1}.${now.getDate()}`;
}

function existingTags() {
	try {
		return execFileSync("git", ["tag", "--list", "v*"], { cwd: root, encoding: "utf8" }).split("\n");
	} catch {
		// No repository, or no tags yet. Starting at -1 is correct either way.
		return [];
	}
}

function nextBuild(date) {
	const prefix = `v${date}-`;
	const used = existingTags()
		.map((tag) => tag.trim())
		.filter((tag) => tag.startsWith(prefix))
		.map((tag) => Number(tag.slice(prefix.length)))
		.filter((build) => Number.isInteger(build) && build > 0);
	return used.length === 0 ? 1 : Math.max(...used) + 1;
}

const DATE_VERSION = /^\d{4}\.\d{1,2}\.\d{1,2}-\d+$/;

const requested = process.argv[2];
if (requested && !DATE_VERSION.test(requested)) {
	console.error(`Not a date version: ${requested}. Expected YYYY.M.D-N, e.g. 2026.8.10-1.`);
	process.exit(1);
}
const version = requested ?? `${today()}-${nextBuild(today())}`;

/**
 * Written directly rather than through `npm version -ws`, which re-resolves
 * every workspace dependency against the registry and fails: our own
 * `@earendil-works/*` versions do not exist there, and after this they never
 * will, because a date version is deliberately not an upstream release.
 */
function setVersion(path) {
	const pkg = JSON.parse(readFileSync(path, "utf8"));
	if (pkg.version === undefined) return false;
	pkg.version = version;
	// The root is not a workspace, so `sync-versions.js` never sees it, and it
	// depends on a workspace package.
	for (const field of ["dependencies", "devDependencies", "peerDependencies"]) {
		for (const name of Object.keys(pkg[field] ?? {})) {
			if (name.startsWith("@earendil-works/")) pkg[field][name] = `^${version}`;
		}
	}
	writeFileSync(path, `${JSON.stringify(pkg, null, "\t")}\n`);
	return true;
}

console.log(`Setting all packages to ${version}`);
const targets = [
	join(root, "package.json"),
	...readdirSync(join(root, "packages"), { withFileTypes: true })
		.filter((entry) => entry.isDirectory())
		.map((entry) => join(root, "packages", entry.name, "package.json")),
];
// Example extensions under packages/coding-agent/examples/extensions/* are
// listed as npm workspaces, but they are private samples with their own
// versions. `sync-versions.js` only inspects direct children of packages/, so
// it never locksteps them; leave their package.json alone.
for (const path of new Set(targets)) {
	if (setVersion(path)) console.log(`  ${path.slice(root.length + 1)}`);
}

execFileSync("node", [join(root, "scripts", "sync-versions.js")], { cwd: root, stdio: "inherit" });

console.log(`
Done. Next:

  npm install                 # refresh the lockfile
  npm run build               # required: the fork is run from dist/
  git commit -am "Release ${version}"
  git tag v${version}

The tag is not created here. Tagging is a git decision, and this script is
also run to correct a version that was never released.`);

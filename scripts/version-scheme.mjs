#!/usr/bin/env node

/**
 * The fork's version scheme, and the guard that keeps upstream's semver bumps
 * away from it.
 *
 * `npm version patch` on `2026.8.10-1` returns the bare `2026.8.10`: it reads
 * `-1` as a prerelease tag and drops it on the way to the next patch. That is a
 * version FORK.md says is never to be built, and the mistake does not announce
 * itself afterwards, because `2026.8.10` sorts *after* every `-N` build of that
 * day. The next `npm run version:date` on the same date would then hand back a
 * version that npm considers older than what is already published.
 *
 * `version:patch`, `version:minor`, `version:major` and `scripts/release.mjs`
 * are upstream's and still work on upstream's semver, so they are guarded
 * rather than deleted: taking a merge back from upstream leaves them intact,
 * and the guard is what makes them a no-op here.
 *
 * `version:set` is deliberately unguarded. It takes an explicit version, which
 * is how you would deliberately leave the date scheme.
 *
 * Usage as a guard:
 *   node scripts/version-scheme.mjs version:patch
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));

/** `YYYY.M.D-N`. See `scripts/date-version.mjs` for why the fork uses it. */
export const DATE_VERSION = /^\d{4}\.\d{1,2}\.\d{1,2}-\d+$/;

/** All workspaces are locksteped, so the root speaks for every package. */
export function currentVersion() {
	return JSON.parse(readFileSync(join(root, "package.json"), "utf8")).version;
}

/** Exit non-zero if a semver bump would be applied to a date version. */
export function assertNotDateVersion(command) {
	const version = currentVersion();
	if (!DATE_VERSION.test(version)) return;
	const bare = version.replace(/-\d+$/, "");
	console.error(`Refusing to run ${command}: this fork is on the date version ${version}.`);
	console.error(`\`npm version\` would turn that into the bare ${bare}, which is never built here and`);
	console.error(`sorts after every ${bare}-N build, so the mistake would not show up in the ordering.`);
	console.error("Use `npm run version:date` instead. See FORK.md.");
	process.exit(1);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
	assertNotDateVersion(process.argv[2] ?? "this command");
}

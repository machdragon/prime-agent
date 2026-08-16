#!/usr/bin/env node

/**
 * Fail when the fork's dependency ranges have drifted from the upstream release
 * it is based on.
 *
 * The obvious check -- every workspace agrees on a range -- is the wrong one.
 * Upstream is not internally consistent: `v0.7.2` carries `@types/node` at
 * `^22.10.5` in the root and `^24.3.0` in every package, and `chalk` at
 * `^5.6.2` in `packages/ai` and `^5.5.0` in `packages/coding-agent` and
 * `packages/tui`. Enforcing agreement would fail on upstream's own tree and
 * push the fork into diverging from upstream in order to pass, which is exactly
 * the thing that costs us at the next merge.
 *
 * The property that matters is the opposite one: a fork-side range that differs
 * from upstream's is either a conflict or a silent mis-resolution at the next
 * upstream release. The `v0.7.2` merge produced two, and neither surfaced.
 * `typebox` was held at `^1.1.24` in `packages/agent` while upstream moved to
 * `^1.3.9`, and `typescript` was held at `^5.9.2` in the root while upstream
 * moved to `^7.0.2`. Git auto-merged both without a conflict, `npm run check`
 * passed, and they were only found by reading the tree afterwards.
 *
 * The baseline is `upstream.json`. Bump it in the commit that merges a new
 * upstream release, and this reports every range that did not come along.
 *
 * Deliberate divergence is legitimate but has to be written down: add the
 * dependency to `dependencyExceptions` in `upstream.json` with the reason.
 *
 * Only ranges for dependencies present on both sides are compared. Dependencies
 * the fork adds or drops are its own business and show up in the diff anyway.
 */

import { execFileSync } from "node:child_process";
import { readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const FIELDS = ["dependencies", "devDependencies", "peerDependencies"];

const baseline = JSON.parse(readFileSync(join(root, "upstream.json"), "utf8"));

function git(args) {
	return execFileSync("git", args, { cwd: root, encoding: "utf8" });
}

/** The blob at `path` in the baseline commit, or null if it did not exist. */
function upstreamManifest(path) {
	try {
		return JSON.parse(git(["show", `${baseline.commit}:${path}`]));
	} catch {
		return null;
	}
}

try {
	git(["cat-file", "-e", `${baseline.commit}^{commit}`]);
} catch {
	// A shallow or partial clone will not have the object. That is not a fork
	// problem, and failing here would only teach people to skip the check.
	console.log(`Skipped: upstream commit ${baseline.commit} is not in this clone.`);
	process.exit(0);
}

try {
	git(["merge-base", "--is-ancestor", baseline.commit, "HEAD"]);
} catch {
	console.error(`upstream.json names ${baseline.ref} (${baseline.commit.slice(0, 9)}), which is not an ancestor of HEAD.`);
	console.error("Either that upstream release has not been merged yet, or the baseline was bumped without merging it.");
	process.exit(1);
}

// Example extensions under packages/coding-agent/examples/extensions/* are npm
// workspaces but private samples, and date-version.mjs leaves them alone for the
// same reason. Match that: only the root and the direct children of packages/.
const manifests = [
	"package.json",
	...readdirSync(join(root, "packages"), { withFileTypes: true })
		.filter((entry) => entry.isDirectory())
		.map((entry) => `packages/${entry.name}/package.json`),
];

const drift = [];

for (const path of manifests) {
	const upstream = upstreamManifest(path);
	if (upstream === null) continue; // Fork-only package.

	const fork = JSON.parse(readFileSync(join(root, path), "utf8"));
	for (const field of FIELDS) {
		for (const [dep, range] of Object.entries(fork[field] ?? {})) {
			// The fork's own packages carry the fork's version by design.
			if (dep.startsWith("@earendil-works/")) continue;
			if (dep in baseline.dependencyExceptions) continue;
			const theirs = upstream[field]?.[dep];
			if (theirs !== undefined && theirs !== range) {
				drift.push({ path, field, dep, ours: range, theirs });
			}
		}
	}
}

if (drift.length === 0) {
	const exceptions = Object.keys(baseline.dependencyExceptions);
	const note = exceptions.length === 0 ? "" : ` (${exceptions.length} recorded exception(s): ${exceptions.join(", ")})`;
	console.log(`Fork dependency ranges match upstream ${baseline.ref}${note}.`);
	process.exit(0);
}

console.error(`Fork dependency ranges have drifted from upstream ${baseline.ref}:\n`);
for (const { path, field, dep, ours, theirs } of drift) {
	console.error(`  ${path} ${field}`);
	console.error(`    ${dep}: fork ${ours}, upstream ${theirs}`);
}
console.error(`
Each of these is a conflict or a silent mis-resolution at the next upstream
merge. Either align the range with upstream, or record it in
upstream.json "dependencyExceptions" with the reason it has to differ.`);
process.exit(1);

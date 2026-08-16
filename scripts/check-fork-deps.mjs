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
 * Both directions count. A range the fork holds back is drift, and so is a
 * dependency upstream added that a merge dropped from a fork manifest: same
 * silently-auto-merged manifest change, same failure to notice.
 *
 * The baseline is `upstream.json`. Bump it in the commit that merges a new
 * upstream release, and this reports everything that did not come along.
 *
 * Deliberate divergence is legitimate but has to be written down, per manifest:
 *
 *     "dependencyExceptions": {
 *       "packages/tui/package.json": { "typescript": "why it has to differ" }
 *     }
 *
 * Scoped to the manifest on purpose. Keyed by dependency name alone, one
 * approved difference would switch the check off for that dependency in every
 * other workspace too -- and single-workspace drift is precisely what this
 * exists to catch.
 */

import { execFileSync } from "node:child_process";
import { readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const FIELDS = ["dependencies", "devDependencies", "peerDependencies"];

const baseline = JSON.parse(readFileSync(join(root, "upstream.json"), "utf8"));
const exceptionsFor = (path) => baseline.dependencyExceptions?.[path] ?? {};

/** stderr is swallowed: a missing object is an expected answer here, not noise. */
function git(args) {
	return execFileSync("git", args, { cwd: root, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] });
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
	// A tarball or partial clone will not have the object, and failing there
	// would only teach people to skip the check. CI is the opposite case: it is
	// where this has to run, so a missing baseline there means the checkout is
	// too shallow, not that the check does not apply.
	const message = `upstream commit ${baseline.commit} is not in this clone`;
	if (process.env.CI) {
		console.error(`${message}.`);
		console.error("CI must check out enough history to reach it: set `fetch-depth: 0` on actions/checkout.");
		process.exit(1);
	}
	console.log(`Skipped: ${message}.`);
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

for (const path of Object.keys(baseline.dependencyExceptions ?? {})) {
	if (manifests.includes(path)) continue;
	console.error(`upstream.json records exceptions for ${path}, which is not a manifest this check reads.`);
	console.error(`Expected one of: ${manifests.join(", ")}`);
	process.exit(1);
}

const drift = [];
const dropped = [];

for (const path of manifests) {
	const upstream = upstreamManifest(path);
	if (upstream === null) continue; // Fork-only package.

	const fork = JSON.parse(readFileSync(join(root, path), "utf8"));
	const excepted = exceptionsFor(path);

	for (const field of FIELDS) {
		const ours = fork[field] ?? {};
		const theirs = upstream[field] ?? {};

		for (const [dep, range] of Object.entries(ours)) {
			// The fork's own packages carry the fork's version by design.
			if (dep.startsWith("@earendil-works/") || dep in excepted) continue;
			if (theirs[dep] !== undefined && theirs[dep] !== range) {
				drift.push({ path, field, dep, ours: range, theirs: theirs[dep] });
			}
		}

		for (const [dep, range] of Object.entries(theirs)) {
			if (dep.startsWith("@earendil-works/") || dep in excepted) continue;
			if (!(dep in ours)) dropped.push({ path, field, dep, theirs: range });
		}
	}
}

if (drift.length === 0 && dropped.length === 0) {
	const recorded = Object.entries(baseline.dependencyExceptions ?? {}).flatMap(([path, deps]) =>
		Object.entries(deps).map(([dep, reason]) => `    ${path} ${dep}: ${reason}`),
	);
	console.log(`Fork dependency ranges match upstream ${baseline.ref}.`);
	if (recorded.length > 0) console.log(`  Recorded exceptions:\n${recorded.join("\n")}`);
	process.exit(0);
}

console.error(`Fork dependencies have drifted from upstream ${baseline.ref}:\n`);
for (const { path, field, dep, ours, theirs } of drift) {
	console.error(`  ${path} ${field}`);
	console.error(`    ${dep}: fork ${ours}, upstream ${theirs}`);
}
for (const { path, field, dep, theirs } of dropped) {
	console.error(`  ${path} ${field}`);
	console.error(`    ${dep}: absent from the fork, upstream ${theirs}`);
}
console.error(`
Each of these is a conflict or a silent mis-resolution at the next upstream
merge. Either match upstream, or record it in upstream.json
"dependencyExceptions" under this manifest's path, with the reason.`);
process.exit(1);

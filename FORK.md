# Running this fork

This is `machdragon/prime-agent`, a fork of `PrimeIntellect-ai/prime-agent`. It
is the version that actually runs on this machine, not a checkout kept beside an
installed release.

## Versioning

Releases are `vYYYY.M.D-N`: `v2026.8.10-1` is the first build made on 10 August
2026. Upstream's `0.7.1` says nothing about which upstream state this fork is
on, and the fork does not follow upstream's cadence: it takes `upstream/main`
when we decide to. A date says exactly when we last took one, and `-N`
separates two builds on the same day.

```bash
npm run version:date              # today, next -N among existing v* tags
npm run version:date 2026.8.10-3  # exactly this
```

`version:date` refreshes the lockfile in place rather than deleting it. Deleting
it re-resolves every caret range to whatever is newest, which is a dependency
upgrade smuggled into a version bump: cutting `2026.8.15-1` that way moved
`@mistralai/mistralai` from 2.2.1 to 2.6.1, whose OpenTelemetry imports the
bundler cannot resolve because they come from an optional peer nothing installs,
and the build failed on a change that had nothing to do with the version. Upgrade
dependencies deliberately, in their own commit.

`-N` comes from git tags (`vYYYY.M.D-*`), not from the package.json already on
disk. The script does not create the tag; that is a separate git step after the
commit. Two runs on the same day without tagging both produce the same `-N`,
which is how you correct a version that was never released. Tag after the
release commit when you want the next build number.

It is valid semver, so npm, the caret ranges `sync-versions.js` writes, and the
update check all keep working. `-N` is a prerelease tag, so `2026.8.10-1` sorts
before a bare `2026.8.10`; we never build a bare one, and within the scheme the
ordering is right: `2026.8.10-1` < `2026.8.10-2` < `2026.8.11-1`.

Versions are written directly rather than through `npm version -ws`, which
re-resolves workspace dependencies against the registry and fails: these
`@earendil-works/*` versions are not published there and never will be.

## Taking upstream changes

Whenever you want them, not on upstream's schedule. Take a release tag rather
than `upstream/main`: it is a known-good point, and it is what `upstream.json`
records.

```bash
git fetch upstream --tags
git log --oneline HEAD..v0.7.3     # what you would be taking
git checkout -b feat/upstream-v0.7.3
git merge v0.7.3
```

### Where this fork may differ from upstream

Every file this fork edits that upstream also edits is a conflict at the next
merge. A dependency range this fork holds back is worse than a conflict, because
git resolves it silently and nothing says it happened. The v0.7.2 merge did that
twice: `typebox` stayed at `^1.1.24` in `packages/agent` and `typescript` at
`^5.9.2` in the root while upstream had moved both. Neither produced a conflict,
`npm run check` passed, and they were only found by reading the tree afterwards.

So the divergence is kept narrow and deliberate:

- **Dependency ranges: none.** This fork carries upstream's ranges exactly.
  `npm run check:fork-deps` compares every range against the release named in
  `upstream.json` and fails on any difference. Where a divergence is genuinely
  needed, record it under `dependencyExceptions` there with the reason.
- **Versions: always, and mechanically.** Every package carries the date
  version, so `package.json` conflicts on every upstream release. The fork's
  side always wins.
- **New behaviour: in new files.** A fork feature living in a file upstream does
  not have never conflicts. `packages/coding-agent/skills/goal-graph/` is the
  model to copy.
- **Changelogs: fork entries stay under `[Unreleased]`.** Upstream's arrive
  under their own released headings below, so the two do not collide.

### Resolving the merge

1. **`package.json`, one per workspace, every time.** Keep the date version and
   the `@earendil-works/*` ranges. Take upstream's side for everything else,
   including every third-party dependency range. That last part is the one that
   gets missed.

2. **`package-lock.json`.** Do not resolve it by hand, and do not trust a
   three-way merge of it: the result describes a tree npm never resolved.
   `.gitattributes` marks it `-merge`, so git reports the conflict and leaves
   our copy instead of inventing a resolution. Delete it; step 4 regenerates it.

3. **`CHANGELOG.md`.** Fork entries under `[Unreleased]`, upstream's under its
   released heading.

4. **Re-baseline and reinstall.** Set `ref` and `commit` in `upstream.json` to
   the release just merged, then:

   ```bash
   npm install
   npm run check:fork-deps     # every range that did not come across
   ```

   Align each range it reports, or record it as an exception, and reinstall.

5. **Verify before installing.** Run the build, checks, and tests below. The
   build matters because the fork runs from `dist/`. The tests matter because
   `npm run check` only typechecks, and a merge of any size can typecheck
   cleanly and still be broken. The goal-graph skill has Python tests that
   `npm test` does not reach:

   ```bash
   cd packages/coding-agent/skills/goal-graph && uv run pytest
   ```

6. **Cut a release.** Taking upstream is exactly the event the date is meant to
   record, so re-run `npm run version:date` for a fresh one, then rebuild,
   commit, and tag.

## Building and installing

The fork must be built: `packages/ai` exports subpaths such as `./mcp` from
`dist/`, so without a build the tracked providers extension fails to load and
every custom provider disappears.

```bash
npm install
npm run build
npm run check
npm test
```

`packages/coding-agent/dist/bundle/cli.js` is the entry point, with a shebang
and the executable bit set, and it resolves its own chunks through its real
path, so a symlink to it works.

Install it as `prime-agent` on PATH:

```bash
ln -sf ~/Projects/prime-agent/packages/coding-agent/dist/bundle/cli.js \
       ~/.local/bin/prime-agent
prime-agent --version         # expect the date version
```

`~/.local/bin` comes before `/opt/homebrew/bin` on PATH, so the symlink wins
immediately and you can verify it before removing anything. Once it is working:

```bash
npm uninstall -g prime-agent  # the released 0.7.1, if still installed
```

The release was **not** a Homebrew formula. `brew list prime-agent` finds
nothing; it was `npm install -g prime-agent` run with Homebrew's npm, which is
why it lived under `/opt/homebrew/lib/node_modules/`. `brew uninstall` will not
remove it.

`~/.local/bin` already holds `prime-general` and `prime-personal`, which exec
`prime-agent` from PATH, so both profiles follow the symlink with no further
change.

A symlink rather than `npm link` or a global install: the fork is rebuilt often,
and a symlink means `npm run build` is the whole deployment step. The cost is
that a broken build is live immediately, which is why `npm run check` and the
tests come before it.

The npm global install is no longer the update path, and the in-app update check
has nothing to upgrade to.

## Running from source

`./prime-agent.sh` runs the TypeScript directly through tsx, which is slower to
start but skips the build. `./prime-agent.sh --dist` runs the built bundle. Both
need `npm install` first, and running from source still needs `npm run build`
once for the `packages/ai` subpath exports.

## What this fork changes

- `packages/coding-agent/skills/goal-graph` — the bundled goal-graph skill:
  decomposable graph of work, dispatch to RLM children, and attribution of a
  result to the model that actually produced it.
- Date-based versioning (`scripts/date-version.mjs`).
- `upstream.json` and `scripts/check-fork-deps.mjs` — the upstream release this
  fork is based on, and the check that keeps dependency ranges from drifting
  away from it between merges.

Configuration, providers, routing policy, and the router extension live in
`~/.dotfiles/prime-agent/`, not here.

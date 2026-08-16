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

`version:patch`, `version:minor`, `version:major` and `npm run release:*` are
upstream's, and they refuse to run while the fork is on a date version. `npm
version patch` reads the `-1` as a prerelease tag and drops it, so `2026.8.10-1`
would become the bare `2026.8.10` -- a version we never build, and one that
sorts *after* every `-N` build of that day, so the mistake would not show up in
the ordering afterwards. They are guarded rather than deleted so that a merge
from upstream leaves them intact. `version:set` is not guarded: it takes an
explicit version, which is how you would deliberately leave the date scheme.

## Taking upstream changes

Whenever you want them, not on upstream's schedule:

```bash
git fetch upstream
git log --oneline HEAD..upstream/main     # what you would be taking
git checkout -b feat/upstream-YYYY-MM-DD
git merge upstream/main
```

The merge will conflict on `package.json` versions, because upstream bumps its
own and this fork is on a date. **Keep the date version** and re-run
`npm run version:date` for a fresh one, since taking upstream is exactly the
event the date is meant to record.

Then rebuild and re-run the checks below before installing.

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

Configuration, providers, routing policy, and the router extension live in
`~/.dotfiles/prime-agent/`, not here.

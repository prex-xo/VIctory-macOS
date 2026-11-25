# Mac test fleet

The fleet runs VIctory's test suites on many different Macs and collects the
results in this repository's issues. It answers a question the project keeps
asking: does a given commit pass on M1, M2, M3, M4 and M5 Macs, and on each
macOS version? It does this without anyone sending a log by hand.

The fleet only ever runs the project's own tests on synthetic data. It never
uses a game. Read [fleet-legal.md](fleet-legal.md) before joining.

## Join with your AI agent

Paste this into an AI coding agent that runs on your Mac, such as Claude Code,
Codex or Cursor:

```text
Read https://raw.githubusercontent.com/pragyaangaur/VIctory-macOS/main/docs/fleet-join.md and follow it to help test VIctory on my Mac. Ask me before installing or posting anything.
```

The agent asks you one question. It explains what it will install, what it
will post under your GitHub account, and the fleet rules, and asks whether you
accept all of it. After your yes it does the setup. You only have to:

1. Sign in to Apple and click Download when the agent opens the Game Porting Toolkit page. Apple's licence has to be accepted by you.
2. Sign in to GitHub in your browser, if the GitHub CLI isn't signed in already.
3. Install Homebrew yourself if it's missing, because its installer needs your password.

## Join by hand

```bash
tools/fleet/fleet.sh doctor      # what is missing
tools/fleet/fleet.sh accept      # read and accept the rules (type "I agree")
tools/fleet/fleet.sh gptk        # open Apple's page and wait for the toolkit download
tools/fleet/fleet.sh next        # open tasks for your chip
tools/fleet/fleet.sh claim N
tools/fleet/fleet.sh run N       # setup.sh, cargo test, then the Wine tests
tools/fleet/fleet.sh submit N
```

`run` checks out the task's commit in a separate git worktree under
`~/Library/Caches/victory-fleet/work`, so your own checkout is not touched.
Logs are kept in `~/Library/Caches/victory-fleet/logs/N` and are never posted.

## How the queue works

The queue is this repository's issues, driven by GitHub Actions
(`.github/workflows/fleet-*.yml` running `tools/fleet/queue.sh`). There is no
server.

**Tasks.** A task is an issue opened by a maintainer with the labels
`fleet:task`, `fleet:open` and a chip label such as `fleet:m3`. A
`<!-- fleet-task -->` block in the body pins the commit and the chip family.
The queue and the worker ignore any task that was not opened by someone listed
in `tools/fleet/fleet.json`.

**Leases.** Workers act through comments. Each command must be the whole
comment.

| Comment | Effect |
| --- | --- |
| `/claim` | Leases an open task to you: assigns it and labels it `fleet:leased`. You can hold one lease at a time. |
| `/beat` | Keeps your lease alive. `fleet.sh run` posts one every 50 minutes. |
| `/release REASON` | Hands the task back. REASON is `setup_failed`, `wrong_hardware`, `no_time` or `other`. |
| `/result` and one json block | Submits the result. Only the lease holder can submit it. |

A lease that goes four hours without a `/claim` or `/beat` is released by the
hourly `fleet-leases` workflow.

**Results.** A result must match this schema exactly. The queue deletes
anything that does not match and says why.

```json
{
  "schema": 1,
  "kind": "test-matrix",
  "upstream_commit": "<the task's 40-character commit>",
  "hw_model": "Mac15,12",
  "chip": "Apple M3 Pro",
  "macos": "15.4.1",
  "gptk": "3.0",
  "setup": "ok",
  "native": { "passed": 300, "failed": 0, "ignored": 2 },
  "wine": { "passed": 343, "failed": 0, "ignored": 0 },
  "failed_tests": ["wine/victory_gpu::tests::example"]
}
```

- `native` and `wine` are `null` when the suite did not run, or its command
  exited unsuccessfully without any counted test failures (for example, a later
  test binary failed to build or crashed). A `null` suite makes the result
  failing. Counted test failures retain their counts.
- `failed_tests` holds up to 100 names, each prefixed with `native/` or `wine/`.

An accepted result closes the task with `fleet:done`, plus `fleet:passing` or
`fleet:failing`, and the queue posts a summary table.

**Screening.** In fleet issues, the queue deletes links to file-sharing sites
and torrents. It hides text that looks like game material or circumvention,
and image attachments, until a maintainer has reviewed it. It labels such
issues `fleet:review`. See [fleet-legal.md](fleet-legal.md).

## For maintainers

```bash
tools/fleet/maintainer.sh labels              # once: create the labels
tools/fleet/maintainer.sh tasks main          # one task per chip for the commit main points to
tools/fleet/maintainer.sh tasks <sha> m1 m2   # or for chosen chips
```

- The list of people allowed to open tasks is `maintainers` in `tools/fleet/fleet.json`.
- `tools/fleet/maintainer.sh report > report.md` writes a Markdown report: coverage per commit, tests that fail only on some chips, and every result. It re-checks each result against the schema.
- To screen every issue in the repository, not just fleet issues, set the repository variable `FLEET_MODERATE_ALL` to `true`.
- Turn on private vulnerability reporting so that rights holders have a private contact route.

## Is it helping?

The fleet answers one question: does each commit pass on every kind of Mac? It doesn't measure progress inside a game. Run `maintainer.sh report` after each round of tasks and watch:

| Signal | What good looks like |
| --- | --- |
| Coverage | Every chip family has a result for each tasked commit, ideally on more than one macOS version. |
| Time to result | Hours, not days, from opening tasks to the last result. |
| Chip-specific failures | The report's "fail on some chips" list. Each entry is a hardware or OS difference found before a user hit it. Turn each one into an issue or a fix. |
| Trend | Failing results per commit go down over time, and the chip-specific list empties as fixes land. |
| Fleet health | Few expired leases and few rejected results. Many of either means the join instructions or the worker need work. |

The fleet pays off when entries in the chip-specific list turn into fixes. Link those fixes to the task issue so the connection is visible.

## Development

```bash
tools/fleet/test.sh          # parsing, schema and screening tests, no network
shellcheck tools/fleet/*.sh
```

The scripts use bash and jq only, so they run on macOS's bash 3.2 and on the
Ubuntu runners. They never evaluate comment text.

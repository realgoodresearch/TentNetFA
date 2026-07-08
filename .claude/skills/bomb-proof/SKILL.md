---
name: bomb-proof
description: Iterate a pull request with the repository's thermo-nuclear code review workflow until the review agent approves. Use after agentically creating a PR into main in a Claude Code session, when asked to bomb-proof a PR, or to drive automated review iteration to approval.
---

# Bomb-Proof PR Review Loop

Drive an open pull request through this repository's manually-triggered Claude
review workflow (`.github/workflows/claude-code-review.yml`) until the review
agent is satisfied: tag the reviewer, await its verdict, fix its findings,
push, and re-trigger — repeating until approved or a round cap is hit.

## Prerequisites

- An open PR **targeting `main`** (the review workflow silently skips other
  base branches — do not loop on a PR with a different base).
- The PR branch checked out locally with push access, so findings can be fixed.
- `gh` CLI (or equivalent GitHub MCP tools) authenticated for this repository.
- The repository's `ANTHROPIC_API_KEY` secret configured (the workflow fails
  without it — if round 1 produces a workflow failure instead of a review,
  stop and report rather than re-triggering).

## The Loop

Run at most **3 rounds** by default. Only exceed that if the user explicitly
asks — every round costs real tokens.

### 1. Trigger the reviewer

Record the current time, then tag the review agent with a top-level PR comment:

```bash
TRIGGER_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)
gh pr comment "$PR_NUMBER" --body '@claude review'
```

Never post a trigger comment while a previous review run is still in progress
(check with `gh run list --workflow claude-code-review.yml --limit 1`).
One trigger per round, nothing more.

### 2. Await the verdict

The workflow ends its summary comment with a line containing
`Thermo-Nuclear Verdict:` followed by `APPROVED` or `CHANGES REQUIRED`.
Wait for a comment **newer than the trigger** containing that marker.

If the session has PR-activity subscription or scheduled check-in tools
available (e.g. remote Claude Code sessions), prefer those over sleep-polling.
Otherwise poll, roughly once a minute for up to ~20 minutes:

```bash
VERDICT=""
for i in $(seq 1 20); do
  sleep 60
  VERDICT=$(gh pr view "$PR_NUMBER" --json comments --jq \
    "[.comments[] | select(.createdAt > \"$TRIGGER_TIME\") | .body
      | select(contains(\"Thermo-Nuclear Verdict\"))] | last // empty")
  [ -n "$VERDICT" ] && break
done
```

If no verdict arrives, check `gh run list --workflow claude-code-review.yml`
for a failed or skipped run, report what happened, and stop — a timeout is
not a cue to re-trigger.

### 3. Evaluate

- **APPROVED** → done. Report the result to the user and exit the loop.
- **CHANGES REQUIRED** → read the full summary comment *and* all inline
  review comments posted since the trigger, then proceed to step 4.

### 4. Fix the findings

Address the findings faithfully, in the reviewer's priority order (structural
issues first, nits last). The reviewer applies the
`thermo-nuclear-code-quality-review` skill, so expect demands for structural
simplification, not cosmetic tweaks — actually restructure; do not paper over
findings with comments or minimal edits, and do not game the verdict.

If a finding is genuinely wrong or out of scope for the PR, reply to that
comment explaining why instead of implementing it — but be honest: the bar
for disagreeing with the reviewer is high, and repeated disagreement is a
signal to escalate to the user, not to keep looping.

Verify the code still works (tests, linters, whatever the repo provides),
commit with a clear message, and push to the PR branch.

### 5. Repeat

Go back to step 1 for the next round.

## Exit Conditions

- **Approved** — report success, including how many rounds it took.
- **Round cap reached** — stop and hand the user a summary of the findings
  that remain unresolved and why. Do not silently keep spending tokens.
- **Workflow failure or timeout** — diagnose via `gh run list` /
  `gh run view`, report, and stop.
- **Non-converging review** (same findings re-raised after honest fix
  attempts, or contradictory demands between rounds) — stop and escalate to
  the user with both positions laid out.

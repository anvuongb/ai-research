# AGENTS.md

This repository is the root of an AI research workspace. It is **topic-organized**:
every research topic gets its own dedicated subdirectory, and all persistent work
for a topic lives inside that subdirectory.

This file is the operating contract for any agent (human-directed or autonomous)
working in this repository. Follow it exactly. When this file and an ad-hoc
instruction conflict, ask the user.

---

## 1. Core principles

1. **One topic = one top-level directory.** A topic is a coherent research focus
   (e.g. "mechanistic interpretability", "RL post-training", "retrieval-augmented
   generation"). Every artifact about that topic belongs inside its directory.
2. **Reading is broad; writing is narrow.** Agents may *read* anything in the
   repository. Agents may *write* only inside the directory of the topic
   currently being discussed.
3. **When in doubt, stop and ask.** If the active topic cannot be determined from
   context with certainty, do not guess and do not write. Ask the user to confirm
   before creating or modifying anything. See §4.
4. **Do not cross topic boundaries when writing.** Never edit files in a topic
   directory other than the active one, even if a fix seems obvious. Flag it to
   the user instead.

---

## 2. Repository layout

```
ai-research/
├── AGENTS.md                 # this file (root config)
└── <topic-name>/             # one directory per research topic
    ├── README.md
    ├── notes/
    ├── data/
    ├── sources/
    ├── experiments/
    ├── papers/
    └── assets/
```

### Topic directory naming

- Names are **lowercase-hyphenated**: `mechanistic-interpretability`,
  `rl-post-training`, `retrieval-augmented-generation`.
- ASCII only, no spaces, no underscores, no trailing punctuation.
- Keep names short but specific enough to disambiguate.
- One directory per topic; if two names overlap in meaning, ask the user whether
  to merge or keep them separate.

### Standard topic scaffold

When a new topic is created, initialize it with this structure:

| Path             | Purpose                                                              |
| ---------------- | -------------------------------------------------------------------- |
| `README.md`      | Scope, core questions, current status, dated changelog.              |
| `notes/`         | Findings, summaries, decision logs, meeting/reading notes.           |
| `data/`          | Datasets, processed artifacts, downloaded or generated data.         |
| `sources/`       | Paper references, links, citations, bibliographies, extracted text.  |
| `experiments/`   | Code, configs, run outputs, metrics for this topic.                  |
| `papers/`        | Draft write-ups, LaTeX, figures intended for publication.            |
| `assets/`        | Images, diagrams, and other non-data media.                          |

Empty directories should contain a short `.gitkeep` or a one-line `README.md`
explaining their intended contents so the structure is self-documenting.

`html/` is **not** part of the initial scaffold. It is generated on demand from a
topic's Markdown by the `md-to-html` skill (`.agents/skills/md-to-html/`), is
disposable, and must never be hand-edited. See §8 for the sync obligation.

---

## 3. Read/write permissions

### Reading (allowed everywhere)

Agents may read any file in the repository at any time, to build context,
cross-reference related topics, or avoid duplicating work.

### Writing (scoped to the active topic)

Agents may create, modify, or delete files **only inside the directory of the
topic currently being discussed**, per the rules below.

| Location                                   | Write access                                        |
| ------------------------------------------ | --------------------------------------------------- |
| `<active-topic>/**`                        | ✅ Allowed (the active topic)                        |
| `<other-topic>/**` and root-level files    | 🚫 Read-only unless the user grants explicit permission |
| `AGENTS.md`                                | 🚫 Only when the user explicitly asks to change it   |
| Repository root (new files, indexes, etc.) | 🚫 Only with explicit user permission                |

Rules:

- **Explicit permission is required** to write outside the active topic
  directory. A general instruction ("keep the workspace tidy", "update the docs")
  does *not* count — the user must name the target or approve a specific write.
- **Root-level writes** (e.g. a cross-topic index) are permitted only after the
  user explicitly authorizes that specific file.
- **Deleting** files outside the active topic is never allowed without explicit
  permission, even for cleanup.
- If a write outside the active topic would clearly help, **propose it and wait**
  rather than doing it.

---

## 4. Determining the active topic

Before writing anything, determine which topic is active. Use this priority order:

1. **Explicit statement** by the user naming the topic or its directory.
2. **The task's subject matter**, when it maps unambiguously to exactly one
   existing topic directory.
3. **An existing directory mentioned in the conversation** (a file path, a
   `cd` target, a previously discussed topic).

Ask the user to confirm — and do **not** write — when:

- The request could belong to two or more topics.
- The topic is new and no directory exists yet (confirm the intended name).
- The request is about a topic with no matching directory, but could be a
  sub-aspect of an existing one.
- The request touches multiple topics at once.
- The request is ambiguous, meta ("organize things"), or repository-wide.
- You are less than certain for any reason.

When asking, state what you inferred, the candidate directories, and what you
propose to do. Example:

> This could belong to `rl-post-training/` or `mechanistic-interpretability/`.
> Which topic should I write to — or should I create a new one?

If the user confirms a **new** topic, create the directory with the standard
scaffold (§2) before writing anything else.

---

## 5. Creating a new topic

When the user confirms a new topic:

1. Confirm the **slug** (lowercase-hyphenated) with the user if it wasn't given.
2. Create `<slug>/` with the standard scaffold.
3. Write `<slug>/README.md` capturing the topic's scope and initial questions.
4. Add a dated entry to the changelog.
5. Only then begin the actual research work for that topic.

Do not create speculative or "just in case" topic directories. Every directory
should correspond to a topic the user has confirmed they want to focus on.

---

## 6. Working inside a topic

- **`README.md` is the entry point.** Keep it current: scope, open questions,
  status, and a dated changelog of meaningful changes.
- **Put artifacts in the right subdirectory.** Don't dump papers into `data/`
  or datasets into `sources/`.
- **Prefer descriptive filenames.** Use lowercase-hyphenated or snake_case file
  names with dates where useful: `2026-09-27-initial-findings.md`.
- **Record provenance.** For downloaded data and cited sources, note origin,
  URL/DOI/arXiv ID, and retrieval date.
- **Keep experiments reproducible.** Store code, configs, and environment notes
  together, and record how each result was produced.
- **Don't duplicate across topics.** If a finding is relevant to several topics,
  keep the canonical copy in one topic and reference it from others by relative
  path (read-only cross-references are fine).

---

## 7. Research skills and subagent delegation

Two installed capability sets extend what agents can do. Both remain **subject to
the write-scope rules in §3**: using a skill or launching a child agent never
grants permission to write outside the active topic directory.

### 7.1 Research skills (Feynman)

Invoke a skill by naming it or running its slash command. Paper work uses the
`feynman alpha` CLI/tools — **not** a bare global `alpha` binary, which may be
stale. Research workflows typically emit a `.provenance.md` sidecar next to
their report and orchestrate built-in subagents (`researcher`, `verifier`,
`reviewer`, `writer`).

| Skill | Use when | Command | Output |
| --- | --- | --- | --- |
| `alpha-research` | Find/read/ask about papers; inspect a paper's code; annotate | `feynman alpha ...` | Inline results + local annotations |
| `deep-research` | Thorough multi-source investigation | `/deepresearch` | `outputs/<slug>.md` + `.provenance.md` |
| `literature-review` | Lit review / state of the art | `/lit` | `outputs/` + `.provenance.md` |
| `source-comparison` | Compare papers/tools/approaches/claims | `/compare` | comparison matrix in `outputs/` |
| `paper-code-audit` | Paper vs. public code consistency | `/audit` | audit report in `outputs/` |
| `research-review` | Tough internal critique before submission | `/review` | structured review in `outputs/` |
| `paper-writing` | Turn findings into a paper-style draft | `/draft` | draft in `papers/` |
| `ml-training-recipe` | Find an implementable training recipe | `/recipe` | ranked recipe brief in `outputs/` |
| `replication` | Plan/execute reproducing a claim (asks environment first) | `/replicate` | plan, scripts, raw results |
| `autoresearch` | Bounded hypothesis/benchmark experiment loop | `/autoresearch` | `autoresearch.md`, `.sh`, `.jsonl` |
| `pdf-explore` | Extract/cross-check across pages, figures, tables | — | extracted notes + provenance |
| `preview` | Render Markdown/LaTeX to HTML/PDF via pandoc | — | HTML/PDF preview (Markdown stays canonical) |
| `session-log` | Durable session log | `/log` | `notes/session-logs/` |
| `session-search` | Find prior sessions on disk | — | reads `~/.feynman/sessions/*.jsonl` |
| `eli5` | Plain-English explanation of a paper/idea | — | inline |
| `docker` | Run untrusted/experimental research code in isolation | — | results synced to the mounted dir |

A missing slash command is not a blocker — reproduce the workflow with the
matching skill and `subagent(...)`. Use `alpha-research` for academic papers and
web-search tools for current products/releases; combine both for mixed topics.

### 7.2 Subagent delegation (`pi-subagents`)

- **Authorization gate:** delegate only when the current request or applicable
  instructions authorize it. Complexity, risk, or tool-call count alone does
  **not** authorize delegation; otherwise the parent works directly.
- **Parent keeps authority:** user intent, routing, arbitration, decisions, and
  final acceptance stay with the parent.
- **Builtin roles:** `scout` (recon), `worker` (implementation), `reviewer`
  (fresh-context review), `researcher` (web research brief), `delegate`, and
  `oracle`/`advisor` (bounded, read-only hard-decision escalation). Project agents
  override user agents, which override builtins.
- **Launch shapes:** direct `subagent({ agent, task })` for one bounded child;
  `workflowScript` with `runs.run` / `runs.all` / `runs.lanes` for sequencing,
  fanout, retries, and aggregation; **council mode** for multi-advisor debate.
- **Isolation:** one writer per cwd/worktree; parallelize read-only research,
  review, and validation. Use worktree isolation when writers could overlap.
- **Context:** fresh context is the default for adversarial review; `fork` is a
  rare escalation that inherits parent history. Default nesting depth is 2.
- **Speed:** launch async/background by default; block (`async:false`) only when a
  same-turn artifact is required. Final reviews and publication gates stay async.
- **Council mode:** 2–4 read-only advisors, one independent pass, at most one
  cross-examination (a third pass only if explicitly requested). The parent writes
  the decision memo; advisors never mutate files.
- **Reports:** subagent scratch reports are not deliverables. Keep repo-root paths
  such as `reports/...` or `*-report.json` out of child tasks; prefer managed
  artifact paths and `output: false` unless a file is genuinely needed.

### 7.3 Reconciling skills with the write scope (§3)

- Skill output paths are **relative to the active topic directory**. `outputs/`,
  `notes/session-logs/`, and `papers/` mean `<active-topic>/outputs/`, etc.
  `outputs/` is created on demand by research workflows; it is not part of the
  initial scaffold in §2.
- Every delegated child must receive an explicit **authority/edit boundary**:
  read-only, or write only within `<active-topic>/`. Never delegate writes that
  target another topic.
- Subagent scratch artifacts stay out of the repository root (see §3 and §9).
- Council advisors and read-only reviewers never gain write authority.
- Running a skill that would write outside the active topic is a **stop-and-ask**
  condition, exactly as in §4.

---

## 8. Documentation obligations

Whenever an agent makes a meaningful change, it should:

1. Update the active topic's `README.md` changelog with a dated one-line summary.
2. Add or update a note under `notes/` explaining non-obvious findings, decisions,
   or dead ends.
3. Keep citations in `sources/` accurate and complete.
4. Regenerate the topic's HTML preview whenever any of its Markdown changes. Run
   the `md-to-html` skill
   (`python3 .agents/skills/md-to-html/scripts/convert.py <topic>/`) so the
   topic's `html/` tree stays in sync with its Markdown. `html/` is generated
   output: never hand-edit it, and treat a stale `html/` tree as a documentation
   defect to be fixed before finishing.

---

## 9. Guardrails

- **Never guess a write target.** Ambiguity is a stop condition, not a prompt to
  improvise.
- **Never modify another topic's files** to make the current task easier.
- **Never overwrite user-authored content** without preserving it or asking.
- **Never commit secrets, credentials, or private data** into the repository.
- **Never rewrite this `AGENTS.md`** unless the user explicitly requests it.
- **Report blockers.** If a task would require writing outside the active topic,
  stop and tell the user what you need and why.

---

## 10. Quick checklist before any write

- [ ] Which topic is active, and did I confirm it?
- [ ] Is the target path inside `<active-topic>/`?
- [ ] If not, do I have **explicit, specific** user permission for that exact write?
- [ ] Does the write land in the correct subdirectory?
- [ ] Will I update the `README.md` changelog and `notes/` afterward?

If any box is unchecked and unresolved, **ask the user before writing.**

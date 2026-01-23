# Workspace File Creation Policy

**Purpose**: Ensure Claude never creates intermediate reports, summaries, or analysis files in version-controlled areas.

**Last Updated**: 2026-01-19
**Status**: ENFORCED via hooks and agent instructions

---

## Core Principle

**Working papers, analyses, and intermediate summaries MUST NOT be created in version-controlled workspace.**

All such files belong in designated non-version-controlled directories:

- `.claude/reports/` - Temporary analysis reports (gitignored, auto-archived)
- `.claude/agent-outputs/` - Agent completion reports (gitignored, auto-archived)
- `.coordination/` - Coordination files (gitignored, auto-cleaned)

---

## Allowed File Creation in Workspace

### ✅ User-Facing Documentation (ONLY)

Create files in workspace ONLY when they are:

1. **User-facing documentation** that adds value to repository users:
   - README.md updates
   - CONTRIBUTING.md updates
   - Actual tutorials/guides in docs/
   - API documentation
   - Example code with explanations

2. **Source code and tests**:
   - Implementation files (src/)
   - Test files (tests/)
   - Configuration files (pyproject.toml, etc.)

3. **CI/CD and tooling**:
   - Workflow files (.github/workflows/)
   - Scripts (scripts/)
   - Git hooks

### ❌ FORBIDDEN File Patterns

**NEVER create files matching these patterns in workspace root or version-controlled areas:**

```
*_ANALYSIS*.md
*_AUDIT*.md
*_FIXES*.md
*_REPORT*.md
*_RESULTS*.md
*_SUMMARY*.md
*_ENHANCEMENTS*.md
*_VERIFICATION*.md
*COMPREHENSIVE_*.md
*ULTIMATE_*.md
*COMPLETE_*.md
*FINAL_*.md
```

**Rationale**: These patterns indicate intermediate working documents that violate SSOT principles.

---

## Where to Put Different Types of Content

| Content Type                | Correct Location                                                         | Wrong Location                      |
| --------------------------- | ------------------------------------------------------------------------ | ----------------------------------- |
| **Analysis of problem**     | Direct communication to user OR `.claude/reports/YYYY-MM-DD-analysis.md` | `ROOT/PROBLEM_ANALYSIS.md` ❌       |
| **Implementation summary**  | CHANGELOG.md entry OR direct communication                               | `ROOT/IMPLEMENTATION_SUMMARY.md` ❌ |
| **Audit findings**          | Direct communication OR `.claude/reports/YYYY-MM-DD-audit.md`            | `ROOT/AUDIT_REPORT.md` ❌           |
| **Badge documentation**     | `.github/BADGE_MAINTENANCE.md` OR docs/                                  | `ROOT/BADGE_GUIDE.md` ❌            |
| **Test verification**       | Direct communication OR CI test results                                  | `ROOT/TEST_VERIFICATION.md` ❌      |
| **Configuration changes**   | CHANGELOG.md entry                                                       | `ROOT/CONFIG_CHANGES.md` ❌         |
| **User guide**              | `docs/guides/user-guide.md` ✅                                           | `.claude/reports/` ❌               |
| **API reference**           | `docs/api/reference.md` ✅                                               | `.claude/reports/` ❌               |
| **Agent completion report** | `.claude/agent-outputs/[id]-complete.json` ✅                            | `ROOT/` ❌                          |

---

## Enforcement Mechanisms

### 1. Gitignore Rules

`.gitignore` blocks these patterns (DO NOT bypass with `git add -f`):

```gitignore
*REPORT*.md      # Line 273
*ANALYSIS*.md    # Line 276
*AUDIT*.md       # Line 277
*FIXES*.md       # Line 288
```

### 2. Pre-Tool-Use Hook

`.claude/hooks/check_report_proliferation.py`:

- Warns when attempting to write files matching forbidden patterns in `.claude/`
- Non-blocking (warning only) but logged
- Checks against `coding-standards.yaml` forbidden patterns

### 3. Agent Instructions

All agents have instructions to:

- Write completion reports to `.claude/agent-outputs/`
- Communicate findings directly to user
- NOT create intermediate analysis/summary files in workspace

### 4. Coding Standards

`.claude/coding-standards.yaml` defines:

- `report_generation.policy: minimal`
- `report_generation.forbidden_reports` - list of patterns
- `report_generation.allowed_reports` - only structured outputs in `.claude/`

---

## Agent-Specific Guidance

### technical_writer

**Allowed**:

- Update existing docs in `docs/`
- Create new tutorials in `docs/guides/`
- Update README.md for user-facing changes
- Write completion report to `.claude/agent-outputs/`

**Forbidden**:

- Create `WRITING_SUMMARY.md` in root
- Create `DOCUMENTATION_ANALYSIS.md` in root
- Create intermediate drafts in workspace (use `.claude/reports/` for drafts)

### knowledge_researcher

**Allowed**:

- Communicate research findings directly in response
- Write structured findings to `.claude/reports/YYYY-MM-DD-research.md` (gitignored)
- Update actual documentation with researched facts
- Write completion report to `.claude/agent-outputs/`

**Forbidden**:

- Create `RESEARCH_FINDINGS.md` in root
- Create `COMPREHENSIVE_ANALYSIS.md` in root
- Create `INVESTIGATION_REPORT.md` in root

### code_reviewer

**Allowed**:

- Communicate review feedback directly
- Create review summary in `.claude/reports/YYYY-MM-DD-review.md` (gitignored)
- Write completion report to `.claude/agent-outputs/`

**Forbidden**:

- Create `CODE_REVIEW_REPORT.md` in root
- Create `REVIEW_SUMMARY.md` in root
- Create `QUALITY_AUDIT.md` in root

### All Agents

**Universal Rule**: If you're creating a file to document "what you did", it's probably an intermediate file and should:

1. Go in `.claude/reports/` or `.claude/agent-outputs/`
2. OR be communicated directly to user
3. OR be added to CHANGELOG.md (for user-visible changes)

**Never** create files like:

- `IMPLEMENTATION_SUMMARY.md`
- `WORK_COMPLETED.md`
- `CHANGES_MADE.md`
- `FINAL_REPORT.md`

---

## Decision Tree for File Creation

```
Want to create a file in workspace?
│
├─ Is it user-facing documentation?
│  ├─ Yes → Does it add value to repo users?
│  │  ├─ Yes → OK to create in docs/ or update README/CONTRIBUTING
│  │  └─ No → Put in .claude/reports/ or communicate directly
│  └─ No → Continue...
│
├─ Is it source code or tests?
│  ├─ Yes → OK to create in src/ or tests/
│  └─ No → Continue...
│
├─ Is it configuration or CI/CD?
│  ├─ Yes → OK to create in appropriate location
│  └─ No → Continue...
│
└─ Is it analysis/summary/report/audit?
   ├─ Yes → Put in .claude/reports/ or .claude/agent-outputs/
   └─ Unsure → ASK USER where it should go
```

---

## Examples of Proper Behavior

### ❌ WRONG: Create intermediate summary in workspace

```
User: "Add 5 new badges to README"
Agent:
1. Updates README.md with badges ✅
2. Creates README_BADGE_ENHANCEMENTS.md ❌
3. Creates BADGE_IMPLEMENTATION_SUMMARY.md ❌
```

### ✅ CORRECT: Update user-facing docs only

```
User: "Add 5 new badges to README"
Agent:
1. Updates README.md with badges ✅
2. Updates CHANGELOG.md with entry ✅
3. Creates .github/BADGE_MAINTENANCE.md (proper doc location) ✅
4. Communicates changes directly to user ✅
```

### ❌ WRONG: Create analysis file in workspace

```
User: "Analyze CI/local configuration consistency"
Agent:
1. Analyzes configurations
2. Creates CONFIGURATION_CONSISTENCY_AUDIT.md ❌
3. Creates CI_LOCAL_GAP_ANALYSIS.md ❌
```

### ✅ CORRECT: Use designated locations

```
User: "Analyze CI/local configuration consistency"
Agent:
1. Analyzes configurations
2. Creates .claude/reports/2026-01-19-config-audit.md ✅ (gitignored)
3. Communicates findings directly ✅
4. Updates CHANGELOG.md if changes made ✅
```

---

## Handling Requests for Documentation

### User says: "Document the changes"

**Interpret as**:

- Update CHANGELOG.md with user-visible changes
- Update relevant documentation in docs/ if applicable
- Communicate summary directly to user

**NOT**:

- Create `CHANGES_DOCUMENTATION.md` in root

### User says: "Create a summary of what was done"

**Interpret as**:

- Communicate summary directly in response
- Write agent completion report to `.claude/agent-outputs/[id]-complete.json`

**NOT**:

- Create `WORK_SUMMARY.md` in root

### User says: "Analyze and document X"

**Interpret as**:

1. Analyze X
2. Write analysis to `.claude/reports/YYYY-MM-DD-X-analysis.md` (gitignored)
3. Communicate findings directly
4. If findings result in changes, update CHANGELOG.md

**NOT**:

- Create `X_ANALYSIS_REPORT.md` in root

---

## Cleanup of Existing Violations

Files matching forbidden patterns that were accidentally committed:

1. **Recent commits (not yet on main)**: Remove immediately
2. **Already on main branch**: Leave for historical context, add to gitignore
3. **Always update gitignore**: Ensure pattern is blocked

---

## Gitignore Structure

```gitignore
# Analysis, reports, and summaries (working papers)
*REPORT*.md
*ANALYSIS*.md
*AUDIT*.md
*FIXES*.md
*SUMMARY*.md
*ENHANCEMENTS*.md
*VERIFICATION*.md

# Claude working directories (auto-created)
.claude/reports/
.claude/agent-outputs/
.claude/hooks/*.log
.coordination/

# Exceptions (files that should be tracked)
!.github/**
!docs/**
!CHANGELOG.md
!CONTRIBUTING.md
!README.md
```

---

## Monitoring and Maintenance

### Weekly

- Run `scripts/cleanup.sh` to archive old reports
- Check `.claude/reports/` size doesn't exceed 100MB

### Monthly

- Review `.claude/reports/archive/` - delete files >30 days old
- Verify gitignore patterns cover all intermediate file types

### Quarterly

- Audit workspace for files matching forbidden patterns
- Update `.claude/coding-standards.yaml` if new patterns emerge

---

## Questions and Edge Cases

### Q: User explicitly asks for a file called "ANALYSIS.md" in root

**A**: Clarify intent. Ask: "Do you want this in the git repository (for other developers) or as a working paper (.claude/reports/)?" Then follow user's explicit instruction.

### Q: Creating documentation that references implementation details

**A**: OK if it's user-facing (e.g., architecture docs, design decisions). Create in `docs/`. NOT OK if it's "what I did" summary - that's CHANGELOG.md or direct communication.

### Q: Multiple agents working on same task - where to coordinate?

**A**: Use `.coordination/` with task-specific subdirectories. All files gitignored. Write handoff data to `.claude/agent-outputs/`.

### Q: Need to draft documentation before finalizing

**A**: Draft in `.claude/reports/YYYY-MM-DD-draft-[topic].md`. When finalized, move to proper location in `docs/` and delete draft.

---

## Success Metrics

- ✅ No `*_REPORT*.md`, `*_ANALYSIS*.md`, `*_AUDIT*.md` files in workspace root
- ✅ All working papers in `.claude/reports/` or `.claude/agent-outputs/`
- ✅ CHANGELOG.md is SSOT for "what changed when"
- ✅ Gitignore blocks all intermediate file patterns
- ✅ Weekly cleanup keeps `.claude/reports/` under 100MB
- ✅ Agents communicate findings directly instead of creating files

---

**Remember**: When in doubt, communicate directly or use `.claude/reports/`. Never create intermediate files in version-controlled workspace.

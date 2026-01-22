# Keyword Disambiguation in Agent Routing

## Overview

Some routing keywords intentionally overlap across multiple agents. This document explains how disambiguation works and which keywords overlap.

## Overlapping Keywords

### "document" keyword

**Agents**: `technical_writer` ONLY (as of 2026-01-22)

**Change**: The "document" keyword was removed from `knowledge_researcher` to eliminate routing ambiguity.

**Disambiguation** (NO LONGER NEEDED):

- "document this code" → `technical_writer`
- "document research findings" → `technical_writer` (will create documentation from research)
- "create documentation" → `technical_writer`
- "research and document" → Route to `knowledge_researcher` first (research keyword), then handoff to `technical_writer`

**Migration**: No overlap remains. "document" now exclusively routes to `technical_writer`.

### "write" keyword

**Agents**: `code_assistant`, `technical_writer`

**Disambiguation**:

- "write code" → `code_assistant`
- "write a function" → `code_assistant`
- "write documentation" → `technical_writer`
- "write a guide" → `technical_writer`
- "write a tutorial" → `technical_writer`
- Ambiguous cases: orchestrator checks for code-related keywords (function, class, script)

### "review" keyword

**Agents**: `code_reviewer`

**Disambiguation**:

- "review this code" → `code_reviewer`
- "code review" → `code_reviewer`
- "review PR" → `code_reviewer`
- No overlap - unique to code_reviewer

### "research" vs "document"

**Agents**: `knowledge_researcher` vs `technical_writer`

**Disambiguation**:

- Research keywords (research, investigate, validate, verify, fact-check, sources) → `knowledge_researcher`
- Writing keywords (write, tutorial, guide, how-to, instruction, walkthrough) → `technical_writer`
- When both are present: `knowledge_researcher` first, then handoff to `technical_writer`

### "commit" vs "create"

**Agents**: `git_commit_manager` vs `code_assistant`

**Disambiguation**:

- "commit changes" → `git_commit_manager`
- "create a commit" → `git_commit_manager`
- "create a function" → `code_assistant`
- Git keywords (git, commit, push, staged) take precedence

## Routing Algorithm

The orchestrator uses this algorithm to handle keyword overlaps:

1. **Extract keywords** from user input (lowercase, stemmed)
2. **Match keywords to agents** via frontmatter `routing_keywords`
3. **Calculate match score** for each agent:
   - +2 points for exact keyword match
   - +1 point for context-relevant match
   - +3 points for command keyword (e.g., "git" for git_commit_manager)
4. **Disambiguation logic**:
   - If clear winner (score > 2x second place): route to that agent
   - If ambiguous (scores within 50%): check context
   - If still ambiguous: route to orchestrator for user clarification
5. **Context checking** for ambiguous cases:
   - Look for disambiguating keywords in surrounding text
   - Check for file extensions (.py, .md, .json)
   - Analyze sentence structure and intent

## Examples

### Example 1: Clear Winner

**User Input**: "Write a function to parse JSON"

**Keyword Matches**:

- `code_assistant`: write (+2), function (+2) = 4
- `technical_writer`: write (+2) = 2

**Result**: Route to `code_assistant` (score 2x higher)

### Example 2: Context Disambiguation

**User Input**: "Write documentation for the API"

**Keyword Matches**:

- `code_assistant`: write (+2) = 2
- `technical_writer`: write (+2), documentation (+2) = 4

**Result**: Route to `technical_writer` (documentation keyword disambiguates)

### Example 3: Orchestrator Intervention

**User Input**: "Create something"

**Keyword Matches**:

- `code_assistant`: create (+2) = 2
- `technical_writer`: (no match) = 0

**Result**: Score not high enough, ambiguous intent. Route to `orchestrator` for clarification.

### Example 4: Workflow Chain

**User Input**: "Research Docker networking and write a tutorial"

**Keyword Matches**:

- `knowledge_researcher`: research (+2) = 2
- `technical_writer`: write (+2), tutorial (+2) = 4

**Result**: Orchestrator creates workflow:

1. Phase 1: `knowledge_researcher` (research Docker networking)
2. Phase 2: `technical_writer` (write tutorial from research)

## Adding New Agents

When creating new agents, follow this process:

1. **Check existing keywords** for overlap:
   - Review all agent frontmatter files
   - Identify potential conflicts
2. **If overlap is intentional**:
   - Document it in this file
   - Add disambiguation logic explanation
   - Provide examples
3. **If accidental overlap**:
   - Choose more specific keywords
   - Use compound keywords (e.g., "code review" vs "review")
   - Add disambiguating context keywords
4. **Test disambiguation**:
   - Create 5-10 sample user requests
   - Verify correct routing
   - Document edge cases

## Testing Keyword Disambiguation

To test keyword routing:

```bash
# Test with orchestrator
/ai "write a function to parse JSON"  # Should route to code_assistant
/ai "write documentation for API"     # Should route to technical_writer
/ai "research Docker and write guide" # Should create workflow
```markdown

## Keyword Priority Levels

Some keywords have higher priority for specific agents:

| Keyword | Primary Agent | Priority | Reason |
|---------|---------------|----------|--------|
| git, commit, push | git_commit_manager | HIGH | Domain-specific |
| review, audit | code_reviewer | HIGH | Unique to domain |
| research, investigate | knowledge_researcher | HIGH | Core function |
| function, class, script | code_assistant | MEDIUM | Common in code requests |
| tutorial, guide | technical_writer | MEDIUM | Writing-specific |
| document, write | (multiple) | LOW | Highly ambiguous |

## Conflict Resolution Rules

When scores are tied or close:

1. **Domain-specific keywords win**: "git" beats generic "create"
2. **Compound keywords win**: "code review" beats "review"
3. **Technical context wins**: File extensions, code syntax
4. **Workflow over single agent**: If multiple agents needed, create workflow
5. **Orchestrator clarifies**: When all else fails, ask user

## See Also

- `.claude/agents/orchestrator.md` - Routing algorithm details
- `.claude/docs/routing-concepts.md` - Complete routing explanation
- `.claude/commands/agents.md` - Agent catalog with keywords

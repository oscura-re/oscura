# Fuzzy Routing with RapidFuzz

## Overview

The routing system supports fuzzy keyword matching to handle typos and variations in user requests, improving the user experience when commands contain minor spelling errors.

## How It Works

1. User submits request: "reviw this code" (typo)
2. System fuzzy-matches "reviw" against all keywords
3. "review" keyword scores 91% match (above 80% threshold)
4. Request routes to code_reviewer

## Configuration

**Fuzzy matching threshold**: 80% (configurable in `routing.py`)

This threshold means:

- 80-100%: Strong match, route to agent
- 0-79%: Weak match, ignore

## Dependencies

- **Optional**: `rapidfuzz` package
- **Fallback**: Exact substring matching if rapidfuzz not installed

The system gracefully degrades to exact matching when rapidfuzz is unavailable.

## Installation

```bash
# Optional enhancement for fuzzy matching
uv add --optional fuzzy rapidfuzz
```markdown

Or manually add to `pyproject.toml`:

```toml
[project.optional-dependencies]
fuzzy = ["rapidfuzz>=3.0.0,<4.0.0"]
```markdown

## Examples

| Request (with typo) | Matched Keyword | Agent | Match Score |
|---------------------|----------------|-------|-------------|
| "reviw this code" | "review" | code_reviewer | 91% |
| "wrte documentation" | "write" | technical_writer | 87% |
| "commit chnages" | "commit" | git_commit_manager | 88% |
| "reserch this topic" | "research" | knowledge_researcher | 89% |
| "implment feature" | "implement" | code_assistant | 85% |

## Implementation

Located in `.claude/hooks/shared/routing.py`:

```python
from .routing import fuzzy_keyword_match, rank_agents_by_relevance

# Match single request against keywords
score = fuzzy_keyword_match("reviw code", ["review", "audit"])
# Returns: 0.91

# Rank all agents by relevance
agents = {
    "code_reviewer": ["review", "audit", "check"],
    "code_assistant": ["write", "code", "implement"]
}
rankings = rank_agents_by_relevance("reviw this code", agents)
# Returns: [("code_reviewer", 0.91), ("code_assistant", 0.33)]
```markdown

## Algorithm

Uses RapidFuzz's `partial_ratio` algorithm:

1. Converts request and keywords to lowercase
2. Calculates partial string similarity ratio (0-100)
3. Filters matches below threshold (default: 80%)
4. Normalizes scores to 0.0-1.0 range
5. Averages scores across all keywords

**Partial ratio** means it finds the best matching substring, so:
- "reviw" matches "review" in "code review" → 91%
- "doc" matches "document" → 100%

## Benefits

- **Typo tolerance**: Handles common spelling errors
- **Better UX**: Users don't need perfect spelling
- **Graceful fallback**: Works without rapidfuzz installed
- **Configurable**: Adjust threshold for strictness

## Limitations

- Does not fix semantic misunderstandings (e.g., "review" vs "research")
- Threshold trade-off: Too low = false matches, too high = misses typos
- Only matches keywords, not full intent understanding

## Testing

Test with common typos:

```python
# Test file: tests/unit/test_routing.py
def test_fuzzy_matching_typos():
    assert fuzzy_keyword_match("reviw", ["review"]) > 0.8
    assert fuzzy_keyword_match("wrte", ["write"]) > 0.8
    assert fuzzy_keyword_match("implment", ["implement"]) > 0.8
```markdown

## Configuration Options

Adjust threshold in `routing.py`:

```python
# Stricter matching (fewer false positives)
score = fuzzy_keyword_match(request, keywords, threshold=90)

# Lenient matching (more typo tolerance)
score = fuzzy_keyword_match(request, keywords, threshold=70)
```markdown

## See Also

- `.claude/docs/routing-concepts.md` - Complete routing documentation
- `.claude/agents/orchestrator.md` - Orchestrator agent
- `.claude/hooks/shared/routing.py` - Implementation
- [RapidFuzz documentation](https://github.com/maxbachmann/RapidFuzz) - Algorithm details

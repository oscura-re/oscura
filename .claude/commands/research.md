# /research - Web Research & Citation

Force web research workflow with comprehensive citation and documentation.

## Purpose

Bypass intelligent routing and go directly to `knowledge_researcher` agent for in-depth web research, documentation gathering, and cited summaries.

## Usage

```bash
/research <topic>                           # Research topic
/research Docker networking                 # Technical research
/research Python async best practices       # Language features
/research compare JWT vs OAuth              # Comparisons
/research latest React 19 features          # Recent developments
```

## When to Use

✅ **Use /research when**:

- Need to learn about unfamiliar technology
- Want current best practices
- Comparing alternatives (libraries, approaches, tools)
- Need recent developments or changes
- Want cited sources for documentation
- Learning new technologies or patterns

❌ **Don't use /research when**:

- Just want to write code → Use `/code` or `/ai`
- Already know what to do → Use `/code` or `/feature`
- Need to implement something → Ask `/ai write...` or `/route code_assistant`
- Just want documentation created → Use `/doc`

## How It Works

```
/research <topic>
  ↓
Force route to knowledge_researcher (bypass orchestrator)
  ↓
knowledge_researcher searches web
  ↓
Gathers information from multiple sources
  ↓
Synthesizes findings
  ↓
Returns: Comprehensive summary with citations
```

## Examples

### Example 1: Technology Research

```bash
/research WebAssembly performance characteristics
```

Returns:

- Overview of WebAssembly
- Performance comparisons with JavaScript
- Use cases and limitations
- Code examples
- Citations from MDN, official docs, benchmarks

### Example 2: Best Practices

```bash
/research Python async/await patterns 2026
```

Returns:

- Current async patterns
- Common pitfalls
- Performance considerations
- Code examples
- Links to official docs and PEPs

### Example 3: Comparison Research

```bash
/research compare PostgreSQL vs MongoDB for time-series data
```

Returns:

- Pros/cons of each
- Performance characteristics
- Use case recommendations
- Citations from benchmarks and case studies

## Output Format

Research results are structured as:

````markdown
# Research: <Topic>

## Summary

[Executive summary of findings]

## Key Findings

1. **Finding 1**
   - Details
   - [Source](url)

2. **Finding 2**
   - Details
   - [Source](url)

## Detailed Analysis

### Aspect 1

[In-depth analysis with citations]

### Aspect 2

[In-depth analysis with citations]

## Code Examples

```language
[Relevant code examples]
```

## Recommendations

Based on research:

- Recommendation 1
- Recommendation 2

## Sources

- [Source 1 Title](url) - Description
- [Source 2 Title](url) - Description

## Next Steps

- Suggested actions based on research
````

## After Using /research

What you can do next:

1. **Implement findings** - `/code implement [what you learned]`
2. **Create documentation** - `/doc create guide on [topic]`
3. **Results documented** - Research findings saved to project docs
4. **Ask follow-up** - `/research [deeper topic]`
5. **Build feature** - `/feature implement [based on research]`

## Research Scope

The agent researches:

- ✅ Technical documentation
- ✅ Best practices and patterns
- ✅ Performance characteristics
- ✅ Recent developments (2026)
- ✅ Comparison analyses
- ✅ Code examples
- ✅ Official documentation
- ✅ Community consensus

The agent does NOT:

- ❌ Write code directly (use `/code` after research)
- ❌ Implement features (use `/feature` after research)
- ❌ Create project documentation (use `/doc` for that)

## Benefits of /research

- **Comprehensive**: Gathers from multiple authoritative sources
- **Current**: Focuses on recent information (2026)
- **Cited**: All claims backed by sources
- **Synthesized**: Not just copy-paste, but analyzed and summarized
- **Actionable**: Includes recommendations and next steps

## Integration with Workflow

Typical flow:

```
/research Docker best practices 2026
  ↓
[Read research results]
  ↓
/feature implement Docker deployment based on best practices
  ↓
[System generates requirements from research]
  ↓
Implementation with validation
```

## Search Strategy

knowledge_researcher uses:

1. **Official documentation** (first priority)
2. **Technical blogs** (authoritative sources)
3. **Stack Overflow** (community consensus)
4. **GitHub discussions** (real-world usage)
5. **Academic papers** (for algorithms/theory)
6. **Benchmarks** (for performance data)

## Comparison with Other Commands

|Command|Purpose|Output|Citations|Agent|
|---|---|---|---|---|
|`/research`|Learn/gather info|Research summary|✅ Yes|knowledge_researcher|
|`/doc`|Create docs|Documentation|❌ No|technical_writer|
|`/code`|Write code|Code|❌ No|code_assistant|
|`/ai`|General|Varies|Varies|orchestrator|

## Configuration

Research behavior in `.claude/config.yaml`:

```yaml
agents:
  knowledge_researcher:
    max_sources: 10 # Max sources to cite
    prefer_recent: true # Prioritize 2025-2026 sources
    depth: comprehensive # or "quick"
```

## Workflow

```
/research → knowledge_researcher (direct, no routing)
          → Web search (multiple queries)
          → Source analysis
          → Synthesis
          → Return summary with citations
```

## Aliases

The following aliases work identically:

- `/learn` → `/research`
- `/study` → `/research`
- `/investigate` → `/research`

## Agent

Routes to: **knowledge_researcher** (always, no routing logic)

For normal routing behavior, see `.claude/docs/routing-concepts.md`

## See Also

- `.claude/docs/routing-concepts.md` - How routing works
- `.claude/agents/knowledge_researcher.md` - Agent details
- `.claude/commands/agents.md` - All available agents

## Version

v1.1.0 (2026-01-16) - Removed vestigial "knowledge base" references
v1.0.0 (2026-01-09) - Initial creation as part of workflow command system

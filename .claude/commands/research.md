---
name: research
description: Conduct comprehensive research with authoritative sources and citations
arguments: <topic> [--depth=standard|deep]
version: 1.0.0
created: 2026-01-22
updated: 2026-01-22
status: stable
target_agent: knowledge_researcher
---

# /research - Comprehensive Research with Citations

Conduct thorough research on any topic using authoritative sources with proper citations and validation.

## Usage

````bash
/research <topic>                    # Standard depth research
/research <topic> --depth=deep       # Deep research with more sources
/research "Docker networking"        # Research specific topic
/research "Python async patterns"    # Technical research
```markdown

## Purpose

This command routes to the **knowledge_researcher** agent for:

- Learning new technologies and frameworks
- Investigating best practices and patterns
- Gathering authoritative sources with citations
- Validating technical accuracy and facts
- Cross-referencing with existing documentation
- Quality assurance before publishing content

**When to use**:
- Need to learn unfamiliar technology
- Want authoritative sources with citations
- Require fact-checking and validation
- Before making important technical decisions
- Quarterly content audits

**When NOT to use**:
- Just want documentation written → Use `/ai document` or let orchestrator route to technical_writer
- Just need code implementation → Use natural language or code_assistant
- Just need code review → Use `/review`

## Examples

### Example 1: Technology Research

```bash
/research "Rust async runtime comparison 2026"
```markdown

**Output**:
- 5-10 authoritative sources (official docs, RFCs, technical blogs)
- Comparison of Tokio, async-std, smol
- Performance benchmarks with citations
- Best practices and use cases
- Bibliography in consistent format

### Example 2: Deep Research

```bash
/research "gRPC performance optimization" --depth=deep
```markdown

**Result**:
- 10+ high-quality sources
- Technical validation (tested examples)
- Cross-references to related topics
- Comprehensive bibliography

### Example 3: Best Practices

```bash
/research "Python testing strategies 2026"
```bash

**Returns**:
- pytest vs unittest comparison
- Fixture patterns and examples
- Coverage best practices
- Citations from Python testing documentation

## Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `topic` | string | Yes | - | Research topic or question |

## Options

| Flag | Description |
|------|-------------|
| `--depth=standard` | 5-10 authoritative sources (default) |
| `--depth=deep` | 10+ sources, comprehensive analysis |

## How It Works

```bash
/research <topic>
  ↓
Route to knowledge_researcher agent
  ↓
1. Investigation: Gather 5-10 authoritative sources
  ↓
2. Validation: Fact-check and verify technical accuracy
  ↓
3. Citation: Format citations consistently
  ↓
4. Quality Check: Cross-reference and validate
  ↓
Return research findings with bibliography
```markdown

**Research Philosophy**: 80/20 rule - focus on 20% of authoritative sources that provide 80% of value.

## Research Output Format

Research results include:

1. **Executive Summary**: Key findings (2-3 paragraphs)
2. **Detailed Findings**: Organized by topic
3. **Sources**: All sources with proper citations
4. **Bibliography**: Formatted citations
5. **Quality Assessment**: Source authority and recency
6. **Technical Validation**: Tested examples (if applicable)

## Quality Standards

All research includes:

- ✅ **Authoritative sources**: Official docs, peer-reviewed, established authorities
- ✅ **Citations**: Proper attribution for all facts
- ✅ **Recency**: Prioritize recent sources (< 2 years old)
- ✅ **Technical validation**: Test code examples
- ✅ **Cross-references**: Link to existing project docs

## Error Handling

### Empty Topic

```bash
/research ""
```bash

**Response**:
```bash
Error: Topic cannot be empty
Usage: /research <topic> [--depth=standard|deep]
```bash

### Insufficient Sources

If fewer than 3 authoritative sources found:
```bash
Warning: Only 2 authoritative sources found for "obscure topic"
Recommend: Broaden search terms or accept limited results
Proceed? [y/N]:
```markdown

## Related Commands

| Command | Purpose | When to Use Instead |
|---------|---------|---------------------|
| `/research` | Conduct research | Need authoritative sources |
| `/ai document <topic>` | Create documentation | Just need docs written |
| `/agents` | List agents | Explore capabilities |
| `/route knowledge_researcher <task>` | Force routing | Manual control |

## Workflow Integration

Common patterns:

1. **Research → Document**:
   ```bash
   /research "Docker networking 2026"
   # Review findings
   /ai document Docker networking guide based on research
```bash

2. **Research → Code**:
   ```bash
   /research "Python async best practices"
   # Learn patterns
   /ai implement async queue with best practices
```markdown

3. **Research → Validate**:
   ```bash
   /research "JWT security 2026"
   # Validate existing implementation
   /review src/auth/jwt.py
```python

## Configuration

Research behavior controlled in `.claude/config.yaml`:

```yaml
orchestration:
  agents:
    knowledge_researcher:
      model: opus                    # High reasoning capability
      max_sources: 10                # Max authoritative sources
      citation_format: web           # web|academic|apa|mla
```markdown

## See Also

- `.claude/agents/knowledge_researcher.md` - Full agent capabilities
- `.claude/commands/route.md` - Manual routing control
- `.claude/commands/agents.md` - List all agents
- `.claude/docs/routing-concepts.md` - How routing works
- `CLAUDE.md` - Project workflow

## Version History

- **v1.0.0** (2026-01-22): Initial creation with routing to knowledge_researcher agent
````

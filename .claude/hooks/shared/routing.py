#!/usr/bin/env python3
"""
Fuzzy keyword matching for agent routing.

Provides fuzzy string matching to improve routing decisions when exact keyword
matches are not found. Uses RapidFuzz for high-performance fuzzy matching.

Version: 2.0.0
Created: 2026-01-22
Updated: 2026-01-22
"""

import logging
from pathlib import Path

# Try importing RapidFuzz, gracefully degrade if not available
try:
    from rapidfuzz import fuzz, process

    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    logging.warning(
        "RapidFuzz not installed - fuzzy routing disabled. Install with: uv sync --extra routing"
    )


def fuzzy_match_keywords(
    query_keywords: list[str],
    agent_keywords: list[str],
    threshold: int = 70,
) -> tuple[float, list[tuple[str, str, float]]]:
    """Match query keywords to agent keywords using fuzzy matching.

    Args:
        query_keywords: Keywords extracted from user query
        agent_keywords: Keywords from agent's routing_keywords
        threshold: Minimum fuzzy match score (0-100, default: 70)

    Returns:
        Tuple of (total_score, list of (query_kw, agent_kw, score) matches)

    Example:
        >>> fuzzy_match_keywords(
        ...     ["implement", "authentication"],
        ...     ["implement", "auth", "security"],
        ...     threshold=70
        ... )
        (185.0, [
            ("implement", "implement", 100.0),
            ("authentication", "auth", 85.0)
        ])
    """
    if not FUZZY_AVAILABLE:
        # Fall back to exact matching
        return exact_match_keywords(query_keywords, agent_keywords)

    matches: list[tuple[str, str, float]] = []
    total_score: float = 0.0

    for query_kw in query_keywords:
        # Find best match for this query keyword among agent keywords
        result = process.extractOne(
            query_kw,
            agent_keywords,
            scorer=fuzz.token_sort_ratio,  # Handles word order
            score_cutoff=threshold,
        )

        if result:
            agent_kw, score, _ = result
            matches.append((query_kw, agent_kw, score))
            total_score += score

    return total_score, matches


def exact_match_keywords(
    query_keywords: list[str], agent_keywords: list[str]
) -> tuple[float, list[tuple[str, str, float]]]:
    """Fallback exact keyword matching (no fuzzy).

    Args:
        query_keywords: Keywords from user query
        agent_keywords: Keywords from agent

    Returns:
        Tuple of (total_score, list of exact matches)
    """
    matches: list[tuple[str, str, float]] = []
    total_score: float = 0.0

    # Normalize to lowercase for comparison
    agent_keywords_lower = [kw.lower() for kw in agent_keywords]

    for query_kw in query_keywords:
        query_kw_lower = query_kw.lower()
        if query_kw_lower in agent_keywords_lower:
            matches.append((query_kw, query_kw_lower, 100.0))
            total_score += 100.0

    return total_score, matches


def rank_agents_by_relevance(
    query_keywords: list[str],
    agent_keywords_map: dict[str, list[str]],
    threshold: int = 70,
    top_n: int = 3,
) -> list[tuple[str, float, list[tuple[str, str, float]]]]:
    """Rank all agents by relevance to query.

    Args:
        query_keywords: Keywords from user query
        agent_keywords_map: Dict mapping agent names to their routing keywords
        threshold: Minimum fuzzy match score (default: 70)
        top_n: Return top N agents (default: 3)

    Returns:
        List of (agent_name, total_score, matches) sorted by score descending

    Example:
        >>> agent_map = {
        ...     "code_assistant": ["write", "code", "implement"],
        ...     "knowledge_researcher": ["research", "learn", "study"],
        ... }
        >>> rank_agents_by_relevance(
        ...     ["write", "function"],
        ...     agent_map,
        ...     threshold=70
        ... )
        [
            ("code_assistant", 100.0, [("write", "write", 100.0)]),
        ]
    """
    agent_scores: list[tuple[str, float, list[tuple[str, str, float]]]] = []

    for agent_name, agent_keywords in agent_keywords_map.items():
        score, matches = fuzzy_match_keywords(query_keywords, agent_keywords, threshold)

        if score > 0:  # Only include agents with at least one match
            agent_scores.append((agent_name, score, matches))

    # Sort by score descending
    agent_scores.sort(key=lambda x: x[1], reverse=True)

    return agent_scores[:top_n]


def extract_keywords_from_query(query: str, min_length: int = 3) -> list[str]:
    """Extract keywords from user query.

    Args:
        query: User query string
        min_length: Minimum keyword length (default: 3)

    Returns:
        List of keywords (lowercase, filtered)

    Example:
        >>> extract_keywords_from_query("Write a function to parse JSON")
        ["write", "function", "parse", "json"]
    """
    # Common stopwords to filter out
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "will",
        "with",
    }

    # Split on whitespace and punctuation, lowercase, filter
    import re

    words = re.findall(r"\w+", query.lower())
    keywords = [w for w in words if len(w) >= min_length and w not in stopwords]

    return keywords


def load_agent_keywords(agent_dir: Path) -> dict[str, list[str]]:
    """Load routing keywords from all agent files.

    Args:
        agent_dir: Path to .claude/agents/ directory

    Returns:
        Dict mapping agent names to their routing keywords

    Example:
        >>> keywords = load_agent_keywords(Path(".claude/agents"))
        >>> keywords["code_assistant"]
        ["write", "code", "implement", "build", "create"]
    """
    import yaml

    agent_keywords_map: dict[str, list[str]] = {}

    if not agent_dir.exists():
        return agent_keywords_map

    for agent_file in agent_dir.glob("*.md"):
        try:
            content = agent_file.read_text()

            # Extract frontmatter (between first two --- lines)
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter = yaml.safe_load(parts[1])

                    if frontmatter and "routing_keywords" in frontmatter:
                        agent_name = agent_file.stem
                        keywords = frontmatter["routing_keywords"]

                        if isinstance(keywords, list):
                            agent_keywords_map[agent_name] = keywords

        except Exception as e:
            logging.warning(f"Failed to load keywords from {agent_file}: {e}")

    return agent_keywords_map


def route_with_fuzzy_matching(
    query: str, agent_dir: Path, threshold: int = 70, top_n: int = 3
) -> list[tuple[str, float, list[tuple[str, str, float]]]]:
    """Complete fuzzy routing pipeline.

    Args:
        query: User query string
        agent_dir: Path to .claude/agents/ directory
        threshold: Minimum fuzzy match score (default: 70)
        top_n: Return top N agents (default: 3)

    Returns:
        List of (agent_name, total_score, matches) sorted by relevance

    Example:
        >>> results = route_with_fuzzy_matching(
        ...     "Write a function to authenticate users",
        ...     Path(".claude/agents"),
        ...     threshold=70
        ... )
        >>> results[0]
        ("code_assistant", 285.0, [
            ("write", "write", 100.0),
            ("function", "function", 100.0),
            ("authenticate", "auth", 85.0)
        ])
    """
    # Extract keywords from query
    query_keywords = extract_keywords_from_query(query)

    # Load agent keywords from filesystem
    agent_keywords_map = load_agent_keywords(agent_dir)

    # Rank agents by relevance
    ranked_agents = rank_agents_by_relevance(query_keywords, agent_keywords_map, threshold, top_n)

    return ranked_agents


def get_fuzzy_threshold_from_config(config_path: Path) -> int:
    """Load fuzzy matching threshold from config.yaml.

    Args:
        config_path: Path to .claude/config.yaml

    Returns:
        Fuzzy matching threshold (default: 70 if not found)
    """
    import yaml

    try:
        if config_path.exists():
            config = yaml.safe_load(config_path.read_text())

            # Navigate to routing.fuzzy_threshold
            if config and "routing" in config:
                routing_config = config["routing"]
                if isinstance(routing_config, dict) and "fuzzy_threshold" in routing_config:
                    threshold = routing_config["fuzzy_threshold"]
                    if isinstance(threshold, int) and 0 <= threshold <= 100:
                        return threshold

    except Exception as e:
        logging.warning(f"Failed to load fuzzy threshold from config: {e}")

    return 70  # Default threshold


# Backwards compatibility with existing code
def fuzzy_keyword_match(request: str, keywords: list[str], threshold: int = 80) -> float:
    """Calculate fuzzy match score for request against keywords.

    DEPRECATED: Use fuzzy_match_keywords() for new code.

    Args:
        request: User's request text
        keywords: List of routing keywords for an agent
        threshold: Minimum fuzzy match ratio (0-100)

    Returns:
        Match score (0.0-1.0)
    """
    query_keywords = extract_keywords_from_query(request)
    score, _ = fuzzy_match_keywords(query_keywords, keywords, threshold)

    # Normalize to 0-1 range
    max_possible = len(query_keywords) * 100
    return score / max_possible if max_possible > 0 else 0.0


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: routing.py <query>")
        print('Example: routing.py "write a function to parse JSON"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    agent_dir = Path(".claude/agents")

    if not agent_dir.exists():
        print(f"Error: Agent directory not found: {agent_dir}")
        sys.exit(1)

    print(f"Query: {query}")
    print(f"Fuzzy matching: {'enabled' if FUZZY_AVAILABLE else 'disabled'}")
    print()

    results = route_with_fuzzy_matching(query, agent_dir)

    if not results:
        print("No matching agents found")
    else:
        print("Top matching agents:")
        for i, (agent_name, score, matches) in enumerate(results, 1):
            print(f"\n{i}. {agent_name} (score: {score})")
            print("   Matches:")
            for query_kw, agent_kw, match_score in matches:
                print(f"     - '{query_kw}' → '{agent_kw}' ({match_score})")

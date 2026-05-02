"""Analysis helpers for measurement counts and classical-vs-quantum comparison."""


def most_likely_bitstring(counts: dict[str, int]) -> str:
    """Return the most frequently measured bitstring."""

    if not counts:
        raise ValueError("Measurement counts cannot be empty.")
    return max(counts, key=counts.get)


def classical_linear_search_steps(items: list[str], target_item: str) -> int | None:
    """Return one-based linear-search steps, or None when the target is absent."""

    for step, item in enumerate(items, start=1):
        if item == target_item:
            return step
    return None


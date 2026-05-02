"""Educational explanations and misconception warnings for GroverLab."""


CORE_WARNING = (
    "Grover's algorithm does not directly search raw CSV text. "
    "GroverLab maps dataset entries to quantum-searchable indices and searches "
    "those encoded indices."
)

MISCONCEPTION_WARNINGS = [
    "The oracle is problem-specific and not free.",
    "More Grover iterations are not always better.",
    "Measurement is probabilistic.",
    "Noise can reduce or destroy the quantum advantage.",
    "Non-power-of-two datasets require padded unused states.",
]


def educational_overview() -> dict[str, list[str] | str]:
    """Return educational copy shared by UI, API, reports, and docs."""

    return {
        "core_warning": CORE_WARNING,
        "misconception_warnings": MISCONCEPTION_WARNINGS,
    }


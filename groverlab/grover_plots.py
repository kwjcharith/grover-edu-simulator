"""Plotting utilities for measurement distributions and experiment outputs."""


def plot_measurement_counts(counts: dict[str, int], output_path: str | None = None):
    """Create a bar chart for measurement counts.

    Matplotlib implementation will be added with the simulation layer.
    """

    if not counts:
        raise ValueError("Measurement counts cannot be empty.")
    raise NotImplementedError("Plotting implementation is planned next.")


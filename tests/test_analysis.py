from groverlab.grover_analysis import classical_linear_search_steps, most_likely_bitstring


def test_most_likely_bitstring_returns_max_count():
    assert most_likely_bitstring({"00": 3, "11": 9, "10": 4}) == "11"


def test_classical_linear_search_steps_returns_one_based_step():
    assert classical_linear_search_steps(["a", "b", "c"], "b") == 2
    assert classical_linear_search_steps(["a", "b", "c"], "z") is None


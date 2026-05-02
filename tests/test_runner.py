from groverlab.grover_config import SimulationConfig
from groverlab.grover_runner import GroverRunRequest, prepare_run


def test_prepare_run_maps_target_to_bitstring():
    request = GroverRunRequest(
        items=["red", "green", "blue", "yellow"],
        target_item="blue",
        config=SimulationConfig(manual_iterations=2),
    )

    prepared = prepare_run(request)

    assert prepared.target_index == 2
    assert prepared.target_bitstring == "10"
    assert prepared.num_qubits == 2
    assert prepared.iterations == 2


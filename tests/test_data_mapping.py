from groverlab.grover_data import bitstring_to_index, index_dataset, index_to_bitstring


def test_index_dataset_cleans_indexes_and_pads_items():
    dataset = index_dataset([" alpha ", "", "beta", "gamma"])

    assert dataset.items == ["alpha", "beta", "gamma"]
    assert dataset.item_to_index["beta"] == 1
    assert dataset.num_qubits == 2
    assert dataset.padded_items == ["alpha", "beta", "gamma", None]


def test_index_bitstring_round_trip():
    assert index_to_bitstring(2, 3) == "010"
    assert bitstring_to_index("010") == 2


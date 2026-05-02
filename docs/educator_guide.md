# GroverLab Educator Guide

GroverLab is designed to help students understand Grover's Algorithm through an
encoded-index search simulation.

## Key Warning

Grover's algorithm does not directly search raw CSV text. The simulator maps
dataset entries to quantum-searchable indices and searches those encoded
indices.

## Misconceptions To Address

- The oracle is problem-specific and not free.
- More Grover iterations are not always better.
- Measurement is probabilistic.
- Noise can reduce or destroy the quantum advantage.
- Non-power-of-two datasets require padded unused states.
- Grover's Algorithm does not automatically know whether a text item exists in
  a classical dataset.

## Searching For An Item That Is Not In The Dataset

Grover requires a valid oracle. If the target is absent, no marked state exists.
Without a marked state, no phase inversion occurs and no useful amplitude
amplification occurs.

Use this example:

```text
Dataset: apple, banana, pineapple
Target: abc
```

Expected practical result: the simulator reports that the item is not present
and stops before quantum execution. This is the recommended classroom behavior
when teaching practical validation.

Experimental mode demonstrates the no-solution case. It runs a superposition-only
circuit with no marked state, so the histogram should look approximately uniform.
Any measured item is random and should not be interpreted as a successful search.

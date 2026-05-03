# GroverLab Research Protocol

GroverLab is intended for educational research contexts such as TOCE/SIGCSE-style
studies of how students learn quantum search concepts through interactive
simulation.

## Possible Research Questions

- Does encoded-index visualization improve student understanding of Grover's
  Algorithm?
- Do iteration sweeps help learners understand over-rotation?
- Do noise sweeps improve student understanding of quantum hardware limitations?
- Can students distinguish raw text search from encoded-state search after using
  GroverLab?
- How do students reason about the missing-target/no-solution case?

## Evaluation Design

A typical study can use a pre/post design:

1. Pre-test: assess baseline understanding of classical search, superposition,
   oracle marking, amplitude amplification, and measurement.
2. Intervention: students use GroverLab in a structured activity.
3. Post-test: repeat concept questions and add interpretation prompts based on
   generated histograms.
4. Optional delayed post-test: measure retention after one or more weeks.

Comparisons can be made between lecture-only instruction, static visualization,
and GroverLab-guided activity.

## Pre/Post-Test Structure

Suggested question categories:

- Identify what is encoded into qubits.
- Predict the target bitstring for a small dataset.
- Explain what the oracle marks.
- Interpret a measurement histogram.
- Explain why too many iterations can lower success probability.
- Explain how noise changes output.
- Explain why a missing target has no valid marked state.

Use a mix of multiple-choice questions, short answers, and histogram
interpretation tasks.

## Anonymous Interaction Logging

Research logging must be anonymous. Do not log raw dataset contents, names,
emails, student IDs, IP addresses, or free-text identifiers.

Approved log fields include:

- anonymous session ID
- timestamp
- dataset size
- target index
- number of qubits
- padded size
- unused states
- iteration count
- shots
- noise settings
- success probability
- decoded correctness
- runtime
- circuit depth
- gate counts

## Ethics Considerations

- Obtain institutional approval when required.
- Use informed consent for human-subjects studies.
- Make participation voluntary when used in a course.
- Avoid collecting identifiable data unless explicitly approved.
- Separate grades from research participation.
- Report aggregate findings rather than individual traces.

## TOCE/SIGCSE Positioning

GroverLab is positioned as an educational computing tool for teaching abstract
quantum concepts through interactive, inspectable simulations. It supports
research contributions around visualization, misconceptions, novice mental
models, and the role of simulation in quantum computing education.

Possible paper angles:

- A modular educational simulator for Grover's Algorithm.
- Teaching oracle construction and encoded search through interactive examples.
- Understanding over-rotation with iteration sweeps.
- Teaching quantum noise with controlled noisy simulations.
- Investigating student reasoning about no-solution Grover cases.

## Suggested Metrics

Quantitative metrics:

- pre/post score gains
- normalized learning gain
- success on target encoding questions
- success on oracle/diffuser explanation questions
- histogram interpretation accuracy
- time on task
- number of simulation runs
- use of iteration/noise sweeps

Simulation metrics:

- dataset size
- qubit count
- padded states
- iteration count
- shot count
- ideal success probability
- noisy success probability
- ideal-vs-noisy success gap
- circuit depth
- total gate count

Qualitative metrics:

- student explanations of the oracle
- student explanations of measurement probability
- misconceptions about raw text search
- explanations of missing-target behavior
- reflections on noise and practical limitations


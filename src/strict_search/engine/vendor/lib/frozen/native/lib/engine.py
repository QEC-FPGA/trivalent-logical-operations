"""Generate the re-indexed top-cropped (2d+1) by d S-memory candidate.

The old data row y=0 is deleted. Surviving rows are physically re-indexed
by y_new=y_old-1, so old y=1..d becomes physical y=0.5..d-0.5 and is used
for the final readout. The domain-wall seam old y=d+1 is therefore physical
y=d+0.5. Internal checks are kept unchanged wherever they do not touch
the deleted row; the old y=1 ancilla row becomes the new upper boundary.
For YX, R/H/MPP is an ideal preparation boundary and SI1000 starts only at
the first memory operation after the MPP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import stim

from generate_s_memory import load_source

DEFAULT_SOURCE = Path(__file__).resolve().parent / "source_work_topcrop"


MEASURE_FLIP_AXIS = {"M": "X_ERROR", "MZ": "X_ERROR",
                     "MX": "Z_ERROR", "MY": "X_ERROR"}
RESET_FLIP_AXIS = {"R": "X_ERROR", "RX": "Z_ERROR", "RY": "X_ERROR"}
ONE_QUBIT = {"H", "S", "S_DAG"}
TWO_QUBIT = {"CX", "CZ"}
PHYSICAL_GATES = set(MEASURE_FLIP_AXIS) | set(RESET_FLIP_AXIS) | ONE_QUBIT | TWO_QUBIT | {"MPP"}


def build_builder(distance: int, source: Path, basis: str):
    Stage1, *_ = load_source(source)
    if distance < 5 or distance % 2 != 1:
        raise ValueError("distance must be odd and at least 5")
    builder = Stage1(distance)
    assert builder.h == 2 * distance + 2  # internal old-row indexing; y=0 is deleted
    builder.right_seam_width = distance - 3

    # Keep the source gate words, but use the currently requested timing:
    # d standard -> expansion -> d wide -> MY -> d twist -> MY -> d full-wall.
    builder.data_init(basis)
    for k in range(distance):
        builder.standard_round("B" if k % 2 == 0 else "A", k + 1)
    builder.expansion()
    for k in range(distance):
        builder.wide_round("A" if k % 2 == 0 else "B", distance + 1 + k)
    builder.my_seam("right")
    for k in range(distance):
        builder.twist_round("B" if k % 2 == 0 else "A", distance + 6 + k)
    builder.my_seam("left")
    for k in range(distance):
        builder.fullwall_round("A" if k % 2 == 0 else "B", k + 1)
    builder.h_lower()
    builder.shrink()

    names = list(builder.segments)
    assert len([n for n in names if n.startswith("standard_")]) == distance
    assert len([n for n in names if n.startswith("wide_")]) == distance
    assert [n.rsplit("_", 1)[-1] for n in names if n.startswith("wide_")] == ["A" if k % 2 == 0 else "B" for k in range(distance)]
    assert len([n for n in names if n.startswith("twist_memory_")]) == distance
    assert len([n for n in names if n.startswith("fullwall_")]) == distance
    return builder


def body_after_init(builder) -> stim.Circuit:
    body = stim.Circuit()
    for name, segment in builder.segments.items():
        if name != "data_init":
            body += segment
    return body


def y_mpp_preparation(builder) -> tuple[stim.Circuit, dict]:
    """|0> above and on diagonal, |+> below; ideal MPP measures corner Y.

    The two X factors act on prepared + states and the two Z factors act on
    prepared 0 states.  In the noiseless preparation, the measured product is
    therefore the Y value of the upper-left corner data qubit.
    """
    d = builder.d
    data = [builder.data(x, y) for x in range(d) for y in range(1, d + 1)]
    plus = [builder.data(x, y) for x in range(d) for y in range(1, d + 1)
            if y - 1 > x]
    corner = builder.data(0, 1)
    x1, x2 = builder.data(0, 2), builder.data(0, 3)
    z1, z2 = builder.data(1, 1), builder.data(2, 1)
    prep = stim.Circuit()
    prep.append("R", data)
    prep.append("TICK")
    prep.append("H", plus)
    prep.append("TICK")
    terms = [("X", x1), ("X", x2), ("Y", corner), ("Z", z1), ("Z", z2)]
    prep += stim.Circuit("MPP " + "*".join(axis + str(q) for axis, q in terms))
    prep.append("TICK")
    return prep, {"operator": "XXYZZ", "terms": [{"axis": axis, "qid": q,
                                                    "coordinate": list(builder.coords()[q])}
                                                   for axis, q in terms],
                  "measurement_record": 0, "noise": "none; ideal analysis boundary"}


def used_qubits(circuit: stim.Circuit) -> set[int]:
    used = set()
    for op in circuit:
        if op.name in PHYSICAL_GATES and op.name != "MPP":
            used.update(t.value for t in op.targets_copy() if t.is_qubit_target)
    return used


def with_coordinates(circuit: stim.Circuit, builder) -> stim.Circuit:
    result = stim.Circuit()
    coords = builder.coords()
    for q in sorted(used_qubits(circuit)):
        result.append("QUBIT_COORDS", [q], coords[q])
    result += circuit
    return result


def si1000_with_ideal_mpp(circuit: stim.Circuit, p: float) -> stim.Circuit:
    """H-package SI1000 convention, including source extensions MY/CZ/S.

    MPP is an ideal zero-duration boundary operation: no MPP error and no
    added idle noise in its own moment.  All other moments use the same
    5p/2p/p/10/p rules as the previous 2d+2 package.
    """
    if not 0 < p < 0.1:
        raise ValueError("p must be in (0, 0.1)")
    active = used_qubits(circuit)
    result = stim.Circuit()
    moment: list[stim.CircuitInstruction] = []

    def flush() -> None:
        if not moment:
            return
        operated: set[int] = set()
        collapsed: set[int] = set()
        for op in moment:
            name = op.name
            targets = [t.value for t in op.targets_copy() if t.is_qubit_target]
            if name in MEASURE_FLIP_AXIS and targets:
                result.append(MEASURE_FLIP_AXIS[name], targets, 5 * p)
            result.append(op)
            operated.update(targets)
            if name in MEASURE_FLIP_AXIS or name in RESET_FLIP_AXIS:
                collapsed.update(targets)
            if name in RESET_FLIP_AXIS and targets:
                result.append(RESET_FLIP_AXIS[name], targets, 2 * p)
            elif name in ONE_QUBIT and targets:
                result.append("DEPOLARIZE1", targets, p / 10)
            elif name in TWO_QUBIT and targets:
                result.append("DEPOLARIZE2", targets, p)
        idle = sorted(active - operated)
        if idle:
            result.append("DEPOLARIZE1", idle, p / 10)
        if collapsed:
            waiting = sorted(active - collapsed)
            if waiting:
                result.append("DEPOLARIZE1", waiting, 2 * p)
        moment.clear()

    for op in circuit:
        if op.name == "TICK":
            flush()
            result.append(op)
        elif op.name in {"QUBIT_COORDS", "SHIFT_COORDS"}:
            flush()
            result.append(op)
        elif op.name == "MPP":
            flush()
            result.append(op)
        elif op.name in PHYSICAL_GATES:
            moment.append(op)
        else:
            raise ValueError(f"unsupported gate for SI1000 mapping: {op.name}")
    flush()
    return result


def physical_grouping_check(circuit: stim.Circuit) -> tuple[bool, int]:
    """Same physical gate type has at most one instruction in each TICK span."""
    seen: set[str] = set()
    moments = 0
    for op in circuit:
        if op.name == "TICK":
            seen.clear()
            moments += 1
        elif op.name in PHYSICAL_GATES:
            if op.name in seen:
                return False, moments
            seen.add(op.name)
    return True, moments


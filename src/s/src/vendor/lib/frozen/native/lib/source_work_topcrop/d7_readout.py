"""Real MZ/MX terminal readouts for the d=7 frozen-geometry candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import stim

from d7_stage1 import Stage1
from d7_stage1_flow import flow, logicals


P = 0.001


def standard_stabilizers(d: int) -> list[stim.PauliString]:
    def p(terms: list[tuple[int, int, str]]) -> stim.PauliString:
        result = stim.PauliString(d * d)
        for i, j, axis in terms:
            result[d * i + j] = axis
        return result

    checks = []
    for i in range(d - 1):
        for j in range(d - 1):
            axis = "Z" if (i + j) % 2 == 0 else "X"
            checks.append(p([(i, j, axis), (i + 1, j, axis),
                             (i, j + 1, axis), (i + 1, j + 1, axis)]))
    for i in range(0, d - 1, 2):
        checks.append(p([(i, 0, "X"), (i + 1, 0, "X")]))
    for i in range(1, d - 1, 2):
        checks.append(p([(i, d - 1, "X"), (i + 1, d - 1, "X")]))
    for j in range(1, d - 1, 2):
        checks.append(p([(0, j, "Z"), (0, j + 1, "Z")]))
    for j in range(0, d - 1, 2):
        checks.append(p([(d - 1, j, "Z"), (d - 1, j + 1, "Z")]))
    assert len(checks) == d * d - 1
    return checks


def ideal_y_preparation(builder: Stage1) -> tuple[stim.Circuit, list[stim.PauliString]]:
    d = builder.d
    stabs = standard_stabilizers(d)
    logical_y = stim.PauliString(d * d)
    global_y = logicals(builder, "input")["Y"]
    for i in range(d):
        for j in range(d):
            logical_y[d * i + j] = global_y[builder.data(i, j + 1)]
    stabs.append(logical_y)
    tableau = stim.Tableau.from_stabilizers(stabs)
    local = tableau.to_circuit("graph_state")
    out = stim.Circuit()
    for op in local:
        if op.name == "TICK":
            out.append("TICK")
        else:
            out.append(op.name, [builder.data(t.value // d, t.value % d + 1)
                                 for t in op.targets_copy()], op.gate_args_copy())
    out.append("TICK")
    assert out.num_measurements == 0
    return out, stabs


def body_after_init(builder: Stage1) -> stim.Circuit:
    out = stim.Circuit()
    for name, segment in builder.segments.items():
        if name != "data_init":
            out += segment
    return out


def physical_and_prefix(d: int, label: str) -> tuple[stim.Circuit, stim.Circuit, Stage1]:
    builder = Stage1(d)
    if label == "ZZ":
        builder.build_all("Z")
        prefix = builder.circuit.copy()
        terminal_op = "MZ"
    elif label == "YX":
        builder.build_all("X")
        prep, _ = ideal_y_preparation(builder)
        prefix = prep + body_after_init(builder)
        terminal_op = "MX"
    else:
        raise ValueError(label)
    out = prefix.copy()
    for i in range(d):
        for j in range(1, d + 1):
            out.append(terminal_op, [builder.data(i, j)])
    out.append("TICK")
    assert not any(op.name == "MPP" for op in out)
    return out, prefix, builder


def noisy_version(circuit: stim.Circuit) -> stim.Circuit:
    out = stim.Circuit()
    for op in circuit:
        name = op.name
        ts = [t.value for t in op.targets_copy()]
        if name in ("QUBIT_COORDS", "TICK"):
            out.append(op)
        elif name in ("R", "RX", "RY"):
            for q in ts:
                out.append(name, [q])
                out.append({"R": "X_ERROR", "RX": "Z_ERROR", "RY": "X_ERROR"}[name], [q], P)
        elif name in ("H", "S", "S_DAG"):
            for q in ts:
                out.append(name, [q])
                out.append("DEPOLARIZE1", [q], P)
        elif name in ("CX", "CZ"):
            for a, b in zip(ts[::2], ts[1::2]):
                out.append(name, [a, b])
                out.append("DEPOLARIZE2", [a, b], P)
        elif name in ("M", "MX", "MY", "MZ"):
            for q in ts:
                out.append(name, [q], P)
        else:
            raise NotImplementedError(name)
    return out


def indices(mask: int) -> list[int]:
    out = []
    while mask:
        bit = mask & -mask
        out.append(bit.bit_length() - 1)
        mask ^= bit
    return out


def mask_of(items) -> int:
    mask = 0
    for i in items:
        mask ^= 1 << int(i)
    return mask


def observable_indices(physical: stim.Circuit, prefix: stim.Circuit,
                       builder: Stage1, label: str) -> tuple[list[int], int]:
    target = logicals(builder, "shrink")["Z" if label == "ZZ" else "Y"]
    solution = flow(prefix, stim.PauliString(physical.num_qubits), target)
    assert solution["exists"], (label, "input-to-output flow absent")
    mask = mask_of(solution["measurements"])
    first = prefix.num_measurements
    d = builder.d
    if label == "ZZ":
        terminal_sites = {builder.data(i, 1) for i in range(d)}
    else:
        terminal_sites = {builder.data(0, j) for j in range(1, d + 1)}
    for i in range(d):
        for j in range(1, d + 1):
            if builder.data(i, j) in terminal_sites:
                mask ^= 1 << (first + d * i + (j - 1))
    obs = indices(mask)
    empty = stim.PauliString(physical.num_qubits)
    plus = physical.has_flow(stim.Flow(input=empty, output=empty, measurements=obs))
    minus = physical.has_flow(stim.Flow(input=empty, output=-empty, measurements=obs))
    assert plus ^ minus
    return obs, int(minus)


def complete_detectors(physical: stim.Circuit, obs: list[int],
                       terminal_logical_indices: list[int],
                       prefixes: list[stim.Circuit] | None = None) -> tuple[list[list[int]], int]:
    obs_mask = mask_of(obs)
    assert obs_mask
    logical_mask = mask_of(terminal_logical_indices)
    assert (obs_mask & logical_mask).bit_count() % 2 == 1
    basis: dict[int, int] = {}

    def add(mask: int) -> bool:
        reduced = mask
        while reduced:
            bit = reduced & -reduced
            if bit not in basis:
                basis[bit] = reduced
                return True
            reduced ^= basis[bit]
        return False

    candidates: dict[int, tuple[int, int]] = {}
    for stage, circuit in enumerate((prefixes or []) + [physical]):
        for generator in circuit.flow_generators():
            if any(generator.input_copy()) or any(generator.output_copy()):
                continue
            mask = mask_of(generator.measurements_copy())
            if (mask & logical_mask).bit_count() % 2:
                mask ^= obs_mask
            if mask:
                candidates.setdefault(mask, (mask.bit_count(), stage))
    rows = []
    for mask, _ in sorted(candidates.items(), key=lambda item: item[1]):
        if add(mask):
            rows.append(indices(mask))
    rank = physical.count_determined_measurements()
    assert len(rows) + 1 == rank, (len(rows), rank)
    assert add(obs_mask)
    return rows, rank


def annotate(circuit: stim.Circuit, detectors: list[list[int]], obs: list[int]) -> stim.Circuit:
    out = circuit.copy()
    n = circuit.num_measurements
    for row in detectors:
        out.append("DETECTOR", [stim.target_rec(k - n) for k in row])
    out.append("OBSERVABLE_INCLUDE", [stim.target_rec(k - n) for k in obs], 0)
    return out


def build(d: int, label: str, outdir: Path, *, with_noise: bool = True) -> dict:
    physical, prefix, builder = physical_and_prefix(d, label)
    obs, expected = observable_indices(physical, prefix, builder, label)
    first_terminal = prefix.num_measurements
    # Quotient by the encoded observable using its anticommuting dual string.
    # ZZ's dual X is the left column; YX's dual Z is the top row.
    terminal_logical_indices = ([first_terminal + j for j in range(d)] if label == "ZZ"
                                else [first_terminal + d * i for i in range(d)])
    stage_prefix = (builder.segments["data_init"].copy() if label == "ZZ"
                    else ideal_y_preparation(builder)[0])
    prefixes = []
    for name, segment in builder.segments.items():
        if name == "data_init":
            continue
        stage_prefix += segment
        prefixes.append(stage_prefix.copy())
    detectors, rank = complete_detectors(physical, obs, terminal_logical_indices, prefixes)
    analysis = annotate(physical, detectors, obs)
    outdir.mkdir(parents=True, exist_ok=True)
    physical.to_file(str(outdir / f"{label}_physical.stim"))
    analysis.to_file(str(outdir / f"{label}_analysis.stim"))
    noisy_analysis = None
    dem = None
    if with_noise:
        if label == "YX":
            prep, _ = ideal_y_preparation(builder)
            noisy = prep + noisy_version(body_after_init(builder))
            for i in range(d):
                for j in range(1, d + 1):
                    noisy.append("MX", [builder.data(i, j)], P)
            noisy.append("TICK")
        else:
            noisy = noisy_version(physical)
        noisy_analysis = annotate(noisy, detectors, obs)
        noisy_analysis.to_file(str(outdir / f"{label}_noisy.stim"))
        dem = noisy_analysis.detector_error_model(decompose_errors=False, allow_gauge_detectors=False)
        dem.to_file(str(outdir / f"{label}.dem"))
    report = {"distance": d, "label": label, "measurements": physical.num_measurements,
              "detectors": len(detectors), "deterministic_parity_rank": rank,
              "observable_record_count": len(obs), "observable_expected_bit": expected,
              "has_mpp": False, "no_gauge": dem is not None,
              "dem_error_instructions": sum(op.type == "error" for op in dem) if dem else None}
    (outdir / f"{label}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance", type=int, required=True)
    parser.add_argument("--label", choices=["ZZ", "YX"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-noise", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.distance, args.label, args.output, with_noise=not args.no_noise), indent=2), flush=True)

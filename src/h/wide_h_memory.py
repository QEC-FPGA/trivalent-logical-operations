#!/usr/bin/env python3
"""Generate and simulate wide-surface-code memory experiments.

The syndrome-extraction rounds reuse ``wide_h_generator.py``.  The memory
schedule assumes a virtual first-half round before the experiment starts, then
runs d real syndrome rounds by default:

    virtual first -> second -> first -> ... -> final data measurement

The first detector layer replaces the virtual first-half measurements by the
known initialized data parity.  The final detector layer replaces the next
virtual half-cycle measurements by the final data measurement parity.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pymatching
import stim

from wide_h_generator import MeasurementTracker, WideHGenerator, stim_num_text


CoordKey = tuple[int, int]


def ikey(x: int, y: int) -> CoordKey:
    return (2 * x, 2 * y)


def hkey(x: int, y: int) -> CoordKey:
    return (2 * x + 1, 2 * y + 1)


@dataclass(frozen=True)
class BoundarySpec:
    basis: str
    odd_refs: tuple[CoordKey, ...]
    even_refs: tuple[CoordKey, ...]
    data_refs: tuple[CoordKey, ...]
    coord_key: CoordKey


@dataclass
class MemoryBuild:
    distance: int
    basis: str
    measurement_rounds: int
    text_without_observable: str
    tracker: MeasurementTracker
    final_time: int
    before_final_data_measurements: int
    final_logical_indices: list[int]


def data_qubits(g: WideHGenerator) -> list[int]:
    corner = (0, 2 * g.d - 1)
    return sorted(q for key, q in g.half.items() if key != corner)


def _int_keys(g: WideHGenerator) -> set[CoordKey]:
    return {ikey(x, y) for x, y in g.integer}


def _is_data_key(g: WideHGenerator, x: int, y: int) -> bool:
    return (x, y) in g.half and (x, y) != (0, 2 * g.d - 1)


def _append_spec(
    specs: list[BoundarySpec],
    g: WideHGenerator,
    basis: str,
    odd_refs: Iterable[CoordKey],
    even_refs: Iterable[CoordKey],
    data_halfs: Iterable[tuple[int, int]],
    coord_key: CoordKey | None = None,
) -> None:
    ints = _int_keys(g)
    odd = tuple(odd_refs)
    even = tuple(even_refs)
    data = tuple(hkey(x, y) for x, y in data_halfs)
    corner = g.corner_half_key()
    if not all(k in ints or k == corner for k in odd + even):
        return
    if not all(_is_data_key(g, (k[0] - 1) // 2, (k[1] - 1) // 2) for k in data):
        return
    specs.append(BoundarySpec(basis, odd, even, data, coord_key or (even[0] if even else odd[0])))


def odd_to_even_boundary_specs(g: WideHGenerator) -> list[BoundarySpec]:
    """Detector templates for first/last memory boundary layers."""
    d = g.d
    specs: list[BoundarySpec] = []

    # Interior first-half checks.
    for x in range(1, d):
        for y in range(1, 2 * d):
            if (x, y) == (1, 2 * d - 1):
                continue
            if (x + y) % 2 == 0:
                basis = "X"
                even = ikey(x + 1, y)
            else:
                basis = "Z"
                even = ikey(x, y + 1)
            _append_spec(
                specs,
                g,
                basis,
                [ikey(x, y)],
                [even],
                [(x - 1, y - 1), (x, y - 1), (x - 1, y), (x, y)],
                even,
            )

    # y=0 X boundary.
    for n in range(d):
        _append_spec(
            specs,
            g,
            "X",
            [ikey(2 + 2 * n, 0), ikey(3 + 2 * n, 0)],
            [ikey(2 + 2 * n, 0), ikey(3 + 2 * n, 0)],
            [(1 + 2 * n, 0), (2 + 2 * n, 0)],
            ikey(2 + 2 * n, 0),
        )

    # x=0 Z boundary.
    for n in range(d):
        _append_spec(
            specs,
            g,
            "Z",
            [ikey(0, 1 + 2 * n), ikey(0, 2 + 2 * n)],
            [ikey(0, 1 + 2 * n), ikey(0, 2 + 2 * n)],
            [(0, 2 * n), (0, 1 + 2 * n)],
            ikey(0, 1 + 2 * n),
        )

    # Left-bottom special Z corner.
    _append_spec(
        specs,
        g,
        "Z",
        [ikey(0, 2 * d - 1), ikey(1, 2 * d), ikey(2, 2 * d)],
        [ikey(0, 2 * d - 1), g.corner_half_key()],
        [(0, 2 * d - 2), (1, 2 * d - 1)],
        ikey(0, 2 * d - 1),
    )

    # Left-bottom special X corner from the virtual second-half three-body check.
    _append_spec(
        specs,
        g,
        "X",
        [ikey(1, 2 * d - 1), g.corner_half_key()],
        [ikey(2, 2 * d - 1)],
        [(0, 2 * d - 2), (1, 2 * d - 2), (1, 2 * d - 1)],
        ikey(2, 2 * d - 1),
    )

    # y=2d Z boundary.
    for n in range(d):
        _append_spec(
            specs,
            g,
            "Z",
            [ikey(3 + 2 * n, 2 * d), ikey(4 + 2 * n, 2 * d)],
            [ikey(3 + 2 * n, 2 * d)],
            [(2 + 2 * n, 2 * d - 1), (3 + 2 * n, 2 * d - 1)],
            ikey(3 + 2 * n, 2 * d),
        )

    # Right Z boundary.
    n = 0
    while 2 + 2 * n <= d:
        _append_spec(
            specs,
            g,
            "Z",
            [ikey(d, 2 + 2 * n)],
            [ikey(d, 3 + 2 * n)],
            [(d - 1, 1 + 2 * n), (d - 1, 2 + 2 * n)],
            ikey(d, 3 + 2 * n),
        )
        n += 1

    # Right X boundary.
    for n in range(d):
        _append_spec(
            specs,
            g,
            "X",
            [ikey(d, d + 2 + 2 * n), ikey(d, d + 3 + 2 * n)],
            [ikey(d, d + 2 + 2 * n)],
            [(d - 1, d + 1 + 2 * n), (d - 1, d + 2 + 2 * n)],
            ikey(d, d + 2 + 2 * n),
        )

    return specs


def even_to_odd_boundary_specs(g: WideHGenerator) -> list[BoundarySpec]:
    """Detector templates whose data support lives on the odd half-cycle."""
    d = g.d
    specs: list[BoundarySpec] = []

    # Interior second-half checks.
    for x in range(1, d):
        for y in range(1, 2 * d):
            if (x, y) == (1, 2 * d - 1):
                continue
            if (x + y) % 2 == 1:
                basis = "X"
                odd = ikey(x + 1, y)
            else:
                basis = "Z"
                odd = ikey(x, y + 1)
            _append_spec(
                specs,
                g,
                basis,
                [odd],
                [ikey(x, y)],
                [(x - 1, y - 1), (x, y - 1), (x - 1, y), (x, y)],
                odd,
            )

    # y=0 X boundary.
    for n in range(1, d):
        x0 = 2 * n - 1
        x1 = 2 * n
        _append_spec(
            specs,
            g,
            "X",
            [ikey(x0, 0), ikey(x1, 0)],
            [ikey(x0, 0), ikey(x1, 0)],
            [(x0 - 1, 0), (x1 - 1, 0)],
            ikey(x0, 0),
        )

    # Right Z boundary, y <= d.
    for n in range(1, d + 1):
        y0 = 2 * n - 1
        y1 = 2 * n
        if 0 < y0 <= d and 0 < y1 <= d:
            _append_spec(
                specs,
                g,
                "Z",
                [ikey(d, y1)],
                [ikey(d, y0)],
                [(d - 1, y0 - 1), (d - 1, y0)],
                ikey(d, y1),
            )

    # Right X boundary, y > d.
    for n in range(1, d + 1):
        y0 = 2 * n
        y1 = 2 * n + 1
        if d < y0 <= 2 * d and d < y1 <= 2 * d:
            _append_spec(
                specs,
                g,
                "X",
                [ikey(d, y0)],
                [ikey(d, y0), ikey(d, y1)],
                [(d - 1, y0 - 1), (d - 1, y1 - 1)],
                ikey(d, y0),
            )

    # x=0 Z boundary.
    for n in range(1, d):
        y0 = 2 * n
        y1 = 2 * n + 1
        _append_spec(
            specs,
            g,
            "Z",
            [ikey(0, y0), ikey(0, y1)],
            [ikey(0, y0), ikey(0, y1)],
            [(0, y0 - 1), (0, y1 - 1)],
            ikey(0, y0),
        )

    # y=2d Z boundary.
    for n in range(1, d + 1):
        x0 = 2 * n
        x1 = 2 * n + 1
        if 1 <= x0 <= d and 1 <= x1 <= d:
            _append_spec(
                specs,
                g,
                "Z",
                [ikey(x0, 2 * d)],
                [ikey(x0, 2 * d), ikey(x1, 2 * d)],
                [(x0 - 1, 2 * d - 1), (x1 - 1, 2 * d - 1)],
                ikey(x0, 2 * d),
            )

    # Left-bottom special Z corner.
    _append_spec(
        specs,
        g,
        "Z",
        [ikey(1, 2 * d)],
        [ikey(1, 2 * d - 1), g.corner_half_key()],
        [(0, 2 * d - 1), (1, 2 * d - 1)],
        ikey(1, 2 * d),
    )

    return specs


def append_memory_detector(
    g: WideHGenerator,
    lines: list[str],
    tracker: MeasurementTracker,
    refs: Iterable[tuple[int, CoordKey]],
    coord_key: CoordKey,
    time: int,
) -> None:
    g.append_detector(lines, tracker, refs, coord_keys=[coord_key], time=time)


def build_memory_without_observable(distance: int, basis: str, *, rounds: int | None = None) -> MemoryBuild:
    if basis not in ("X", "Z"):
        raise ValueError("basis must be X or Z")
    if rounds is None:
        rounds = distance
    if rounds < 1:
        raise ValueError("rounds must be >= 1")

    g = WideHGenerator(distance, include_polygons=False)
    tracker = MeasurementTracker(g)
    lines = g.qubit_coord_lines()
    specs = odd_to_even_boundary_specs(g)
    data = data_qubits(g)

    lines.append("TICK")
    lines.append(("RX" if basis == "X" else "R") + " " + " ".join(map(str, data)))

    # Real syndrome rounds start at time 2.  Time 1 is the virtual first-half
    # round whose known value is replaced by the initialized data parity.
    last_syndrome_time = rounds + 1
    for time in range(2, last_syndrome_time + 1):
        g.append_measurement_round(lines, tracker, time=time, second_half=(time % 2 == 0))
        if time == 2:
            for spec in specs:
                if spec.basis == basis:
                    append_memory_detector(
                        g,
                        lines,
                        tracker,
                        [(2, k) for k in spec.even_refs],
                        spec.coord_key,
                        time=2,
                    )
        else:
            g.append_detectors_between(lines, tracker, previous_time=time - 1, current_time=time)

    lines.append("TICK")
    lines.append(("MX" if basis == "X" else "M") + " " + " ".join(map(str, data)))
    final_time = last_syndrome_time + 1
    before_final_data = tracker.count
    tracker.record_round(final_time, data)

    final_specs = specs if last_syndrome_time % 2 == 1 else even_to_odd_boundary_specs(g)
    for spec in final_specs:
        if spec.basis == basis:
            previous_refs = spec.odd_refs if last_syndrome_time % 2 == 1 else spec.even_refs
            append_memory_detector(
                g,
                lines,
                tracker,
                [(last_syndrome_time, k) for k in previous_refs] + [(final_time, k) for k in spec.data_refs],
                spec.coord_key,
                time=final_time,
            )

    if basis == "X":
        if rounds % 2 == 0:
            logical_ys = range(distance + 1)
        else:
            logical_ys = range(distance)
        final_logical_indices = [tracker.rounds[final_time][hkey(distance - 1, y)] for y in logical_ys]
    else:
        final_logical_indices = [tracker.rounds[final_time][hkey(x, 0)] for x in range(distance)]

    return MemoryBuild(
        distance=distance,
        basis=basis,
        measurement_rounds=rounds,
        text_without_observable="\n".join(lines) + "\n",
        tracker=tracker,
        final_time=final_time,
        before_final_data_measurements=before_final_data,
        final_logical_indices=final_logical_indices,
    )


def _solve_xor_system(samples: np.ndarray, target: np.ndarray) -> list[int]:
    """Find columns whose xor matches target over sampled gauge outcomes."""
    solution, _ = _solve_xor_system_with_nullspace(samples, target)
    return [i for i in range(samples.shape[1]) if (solution >> i) & 1]


def _solve_xor_system_with_nullspace(samples: np.ndarray, target: np.ndarray) -> tuple[int, list[int]]:
    """Find one solution and a nullspace basis for samples @ x == target."""
    basis: dict[int, tuple[int, int]] = {}
    for row_arr, rhs0 in zip(samples, target):
        row = 0
        for i, value in enumerate(row_arr):
            if value:
                row |= 1 << i
        rhs = int(rhs0)
        while row:
            pivot = row.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = (row, rhs)
                break
            old_row, old_rhs = basis[pivot]
            row ^= old_row
            rhs ^= old_rhs
        if row == 0 and rhs:
            raise ValueError("sampled equations are inconsistent; increase --observable-shots")

    cols = samples.shape[1]
    solution = 0
    for pivot in sorted(basis):
        row, rhs = basis[pivot]
        value = rhs
        lower = row & ((1 << pivot) - 1)
        while lower:
            bit = lower & -lower
            index = bit.bit_length() - 1
            value ^= (solution >> index) & 1
            lower ^= bit
        if value:
            solution |= 1 << pivot

    pivots = set(basis)
    nullspace: list[int] = []
    for free in range(cols):
        if free in pivots:
            continue
        vector = 1 << free
        for pivot in sorted(basis):
            row, _ = basis[pivot]
            lower = row & ((1 << pivot) - 1)
            value = 0
            while lower:
                bit = lower & -lower
                index = bit.bit_length() - 1
                value ^= (vector >> index) & 1
                lower ^= bit
            if value:
                vector |= 1 << pivot
        nullspace.append(vector)

    return solution, nullspace


def _measurement_count_for_line(parts: list[str]) -> int:
    op = parts[0]
    if op in ("M", "MX", "MY", "MR", "MRX", "MRY"):
        return len(parts) - 1
    return 0


def _detector_span_basis(text: str) -> dict[int, int]:
    measurement_count = 0
    basis: dict[int, int] = {}

    def add_row(row: int) -> None:
        while row:
            pivot = row.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = row
                return
            row ^= basis[pivot]

    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        op = parts[0]
        if op.startswith("DETECTOR"):
            row = 0
            for offset_text in re.findall(r"rec\[(-?\d+)\]", line):
                row ^= 1 << (measurement_count + int(offset_text))
            add_row(row)
        measurement_count += _measurement_count_for_line(parts)

    return basis


def _is_in_span(row: int, basis: dict[int, int]) -> bool:
    while row:
        pivot = row.bit_length() - 1
        if pivot not in basis:
            return False
        row ^= basis[pivot]
    return True


def measurement_record_terms(build: MemoryBuild, indices: Iterable[int]) -> list[str]:
    total_measurements = stim.Circuit(build.text_without_observable).num_measurements
    return [f"rec[{index - total_measurements}]" for index in sorted(indices)]


def measurement_record_terms_from_row(build: MemoryBuild, row: int) -> list[str]:
    total_measurements = stim.Circuit(build.text_without_observable).num_measurements
    return [f"rec[{index - total_measurements}]" for index in range(total_measurements) if (row >> index) & 1]


def default_observable_indices(build: MemoryBuild) -> list[int]:
    """Use the fixed wide-code boundary representatives.

    Z lives on the top data row y=0.5 and the adjacent RZ syndrome row y=1.
    The integer y=0 row is an X-type auxiliary row and is intentionally not
    used for Z matching.
    """
    d = build.distance
    indices: list[int] = []

    if build.basis == "Z":
        for time in range(2, build.final_time):
            if time % 2 == 0:
                xs = [0] + list(range(1, d + 1, 2))
            else:
                xs = list(range(0, d + 1, 2))
            for x in xs:
                index = build.tracker.rounds[time].get(ikey(x, 1))
                if index is not None:
                    indices.append(index)
        for x in range(d):
            indices.append(build.tracker.rounds[build.final_time][hkey(x, 0)])
    elif build.basis == "X":
        for time in range(2, build.final_time):
            if time % 2 == 0:
                ys = list(range(0, d + 1, 2))
            else:
                ys = [0] + list(range(1, d + 1, 2)) + [d + 1]
            for y in ys:
                index = build.tracker.rounds[time].get(ikey(d, y))
                if index is not None:
                    indices.append(index)
        if build.measurement_rounds % 2 == 0:
            for y in range(d + 1):
                indices.append(build.tracker.rounds[build.final_time][hkey(d - 1, y)])
        else:
            for y in range(d):
                indices.append(build.tracker.rounds[build.final_time][hkey(d - 1, y)])
    else:
        raise ValueError("basis must be X or Z")

    return indices


def default_observable_terms(build: MemoryBuild, *, shots: int = 0) -> list[str]:
    del shots
    terms = measurement_record_terms(build, default_observable_indices(build))
    text = append_observable(build, terms)
    stim.Circuit(text).detector_error_model(decompose_errors=True)
    return terms


def resolved_observable_method(requested_method: str) -> str:
    if requested_method not in ("default", "gf2"):
        raise ValueError("observable_method must be default or gf2")
    return requested_method


def choose_observable_terms(build: MemoryBuild, *, method: str = "default", shots: int = 0) -> list[str]:
    method = resolved_observable_method(method)
    if method == "default":
        return default_observable_terms(build, shots=shots)
    return infer_observable_terms(build, shots=shots)


def infer_observable_terms(build: MemoryBuild, *, shots: int) -> list[str]:
    circuit = stim.Circuit(build.text_without_observable)
    total_measurements = circuit.num_measurements
    if shots <= 0:
        shots = max(512, 4 * build.before_final_data_measurements)

    samples = circuit.compile_sampler().sample(shots).astype(np.uint8)
    # The requested final logical string is kept as a seed.  The correction is
    # allowed to use every other measurement, including non-logical final data
    # measurements; this lets the solver multiply by final stabilizer checks
    # without cancelling the requested logical string itself.
    final_logical = set(build.final_logical_indices)
    correction_indices = [i for i in range(total_measurements) if i not in final_logical]
    correction_samples = samples[:, correction_indices]
    target = np.bitwise_xor.reduce(samples[:, build.final_logical_indices], axis=1)
    correction_bits, nullspace = _solve_xor_system_with_nullspace(correction_samples, target)

    detector_span = _detector_span_basis(build.text_without_observable)
    seed_bits = 0
    for index in build.final_logical_indices:
        seed_bits ^= 1 << index

    def expand(bits: int) -> int:
        row = seed_bits
        while bits:
            bit = bits & -bits
            row ^= 1 << correction_indices[bit.bit_length() - 1]
            bits ^= bit
        return row

    chosen_bits = correction_bits
    chosen_row = expand(chosen_bits)
    if _is_in_span(chosen_row, detector_span):
        candidates = sorted(nullspace, key=int.bit_count)
        for delta in candidates:
            trial_bits = correction_bits ^ delta
            trial_row = expand(trial_bits)
            if _is_in_span(trial_row, detector_span):
                continue
            trial_indices = [i for i in range(total_measurements) if (trial_row >> i) & 1]
            trial_terms = " ".join(f"rec[{i - total_measurements}]" for i in trial_indices)
            trial_text = build.text_without_observable + f"OBSERVABLE_INCLUDE(0) {trial_terms}\n"
            try:
                stim.Circuit(trial_text).detector_error_model(decompose_errors=True)
            except ValueError:
                continue
            chosen_bits = trial_bits
            chosen_row = trial_row
            break
        else:
            raise ValueError("failed to infer a non-detector observable; increase --observable-shots")

    indices = [i for i in range(total_measurements) if (chosen_row >> i) & 1]
    return [f"rec[{i - total_measurements}]" for i in indices]


def append_observable(build: MemoryBuild, terms: list[str]) -> str:
    return build.text_without_observable + "OBSERVABLE_INCLUDE(0) " + " ".join(terms) + "\n"


def add_gate_noise(text: str, *, p1: float = 0.0, p2: float = 0.0) -> str:
    """Add simple circuit-level noise with separate 1q/2q probabilities."""
    if p1 <= 0 and p2 <= 0:
        return text
    noisy: list[str] = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            noisy.append(line)
            continue
        op = parts[0]
        targets = " ".join(parts[1:])
        if p1 > 0:
            if op in ("M", "MR") and targets:
                noisy.append(f"X_ERROR({p1}) {targets}")
            elif op == "MX" and targets:
                noisy.append(f"Z_ERROR({p1}) {targets}")
        noisy.append(line)
        if p1 > 0:
            if op == "H" and targets:
                noisy.append(f"DEPOLARIZE1({p1}) {targets}")
            elif op == "R" and targets:
                noisy.append(f"X_ERROR({p1}) {targets}")
            elif op == "RX" and targets:
                noisy.append(f"Z_ERROR({p1}) {targets}")
        if p2 > 0 and op == "CX" and targets:
            noisy.append(f"DEPOLARIZE2({p2}) {targets}")
    return "\n".join(noisy) + "\n"


def add_uniform_noise(text: str, p: float) -> str:
    return add_gate_noise(text, p1=p, p2=p)


def generate_memory_circuit(
    distance: int,
    basis: str,
    *,
    p1: float = 0.0,
    p2: float = 0.0,
    observable_shots: int = 0,
    observable_method: str = "default",
    rounds: int | None = None,
) -> str:
    """Generate a validated noisy/noiseless X/Z memory circuit."""
    build = build_memory_without_observable(distance, basis, rounds=rounds)
    terms = choose_observable_terms(build, method=observable_method, shots=observable_shots)
    text = append_observable(build, terms)
    stim.Circuit(text).detector_error_model(decompose_errors=True)
    text = add_gate_noise(text, p1=p1, p2=p2)
    stim.Circuit(text).detector_error_model(decompose_errors=True)
    return text


def count_detectors_by_time(text: str) -> dict[int, int]:
    counts: dict[int, int] = {}
    for line in text.splitlines():
        if line.startswith("DETECTOR"):
            coords = line.split(")", 1)[0].split("(", 1)[1].split(",")
            time = int(float(coords[2]))
            counts[time] = counts.get(time, 0) + 1
    return dict(sorted(counts.items()))


def simulate(circuit: stim.Circuit, *, shots: int) -> tuple[int, float]:
    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    dets, obs = circuit.compile_detector_sampler().sample(shots, separate_observables=True)
    predictions = matcher.decode_batch(dets)
    failures = np.any(predictions != obs, axis=1)
    return int(np.count_nonzero(failures)), float(np.mean(failures))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate/simulate wide-code X/Z memory experiments.")
    parser.add_argument("-d", "--distance", type=int, default=3, help="odd code distance")
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="number of syndrome measurement rounds before data readout; default is distance",
    )
    parser.add_argument("--basis", choices=["X", "Z"], default="Z", help="memory basis")
    parser.add_argument("-o", "--output", type=Path, default=None, help="write generated .stim circuit")
    parser.add_argument("--noise", type=float, default=None, help="uniform circuit-level noise probability")
    parser.add_argument("--p1", type=float, default=0.0, help="single-qubit/reset/measurement noise probability")
    parser.add_argument("--p2", type=float, default=0.0, help="two-qubit CX noise probability")
    parser.add_argument("--shots", type=int, default=0, help="run pymatching simulation with this many shots")
    parser.add_argument(
        "--observable-shots",
        type=int,
        default=0,
        help="ideal shots used to infer the observable frame; 0 chooses automatically",
    )
    parser.add_argument(
        "--observable-method",
        choices=["default", "gf2"],
        default="default",
        help="default uses the fixed wide-code boundary representative; gf2 uses the sampled GF2 search",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build = build_memory_without_observable(args.distance, args.basis, rounds=args.rounds)
    actual_observable_method = resolved_observable_method(args.observable_method)
    terms = choose_observable_terms(build, method=args.observable_method, shots=args.observable_shots)
    noiseless_text = append_observable(build, terms)
    stim.Circuit(noiseless_text).detector_error_model(decompose_errors=True)

    p1 = args.noise if args.noise is not None else args.p1
    p2 = args.noise if args.noise is not None else args.p2
    text = add_gate_noise(noiseless_text, p1=p1, p2=p2)
    circuit = stim.Circuit(text)
    circuit.detector_error_model(decompose_errors=True)

    print(f"distance={args.distance} basis={args.basis}")
    print(f"rounds={build.measurement_rounds}")
    print(f"observable_method={actual_observable_method}")
    print(f"detectors_by_time={count_detectors_by_time(noiseless_text)}")
    print(f"observable_terms={len(terms)}")
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote={args.output}")
    if args.shots:
        failures, rate = simulate(circuit, shots=args.shots)
        print(f"shots={args.shots} failures={failures} logical_error_rate={stim_num_text(rate)}")


if __name__ == "__main__":
    main()

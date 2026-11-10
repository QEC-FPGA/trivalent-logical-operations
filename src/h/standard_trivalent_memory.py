#!/usr/bin/env python3
"""Generate and simulate standard trivalent surface-code memory experiments."""

from __future__ import annotations

import argparse
import re
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pymatching
import stim

from standard_trivalent_surface_code import StandardTrivalentSurfaceCodeGenerator
from wide_h_generator import MeasurementTracker, stim_num_text
from wide_h_memory import (
    _detector_span_basis,
    _is_in_span,
    _solve_xor_system_with_nullspace,
    add_gate_noise,
)


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
    logical_line: int
    measurement_rounds: int
    text_without_observable: str
    tracker: MeasurementTracker
    final_time: int
    before_final_data_measurements: int
    final_logical_indices: list[int]


def data_qubits(g: StandardTrivalentSurfaceCodeGenerator) -> list[int]:
    return sorted(g.half.values())


def _int_keys(g: StandardTrivalentSurfaceCodeGenerator) -> set[CoordKey]:
    return {ikey(x, y) for x, y in g.integer}


def _is_data_key(g: StandardTrivalentSurfaceCodeGenerator, x: int, y: int) -> bool:
    return (x, y) in g.half


def _append_spec(
    specs: list[BoundarySpec],
    g: StandardTrivalentSurfaceCodeGenerator,
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
    if not all(k in ints for k in odd + even):
        return
    if not all(_is_data_key(g, (k[0] - 1) // 2, (k[1] - 1) // 2) for k in data):
        return
    specs.append(BoundarySpec(basis, odd, even, data, coord_key or (even[0] if even else odd[0])))


def odd_to_even_boundary_specs(g: StandardTrivalentSurfaceCodeGenerator) -> list[BoundarySpec]:
    """Detector templates for the first and final memory layers."""
    d = g.d
    specs: list[BoundarySpec] = []

    # Interior checks. Odd-half X stabilizers move right; Z stabilizers move up.
    for x in range(1, d):
        for y in range(1, d):
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

    # Bottom X boundary.
    n = 0
    while 3 + 2 * n <= d:
        _append_spec(
            specs,
            g,
            "X",
            [ikey(2 + 2 * n, 0), ikey(3 + 2 * n, 0)],
            [ikey(2 + 2 * n, 0), ikey(3 + 2 * n, 0)],
            [(1 + 2 * n, 0), (2 + 2 * n, 0)],
            ikey(2 + 2 * n, 0),
        )
        n += 1

    # Top X boundary.  The top edge alternates by one column between the two
    # half-cycles: odd x=1,3,... matches even x=2,4,....
    n = 0
    while 2 + 2 * n <= d:
        x0 = 1 + 2 * n
        x1 = 2 + 2 * n
        _append_spec(
            specs,
            g,
            "X",
            [ikey(x0, d)],
            [ikey(x1, d)],
            [(x0 - 1, d - 1), (x0, d - 1)],
            ikey(x1, d),
        )
        n += 1

    # Left Z boundary.
    n = 0
    while 2 + 2 * n <= d:
        _append_spec(
            specs,
            g,
            "Z",
            [ikey(0, 1 + 2 * n), ikey(0, 2 + 2 * n)],
            [ikey(0, 1 + 2 * n), ikey(0, 2 + 2 * n)],
            [(0, 2 * n), (0, 1 + 2 * n)],
            ikey(0, 1 + 2 * n),
        )
        n += 1

    # Right Z boundary.
    n = 0
    while 3 + 2 * n <= d:
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

    return specs


def even_to_odd_boundary_specs(g: StandardTrivalentSurfaceCodeGenerator) -> list[BoundarySpec]:
    """Detector templates whose data support lives on the even half-cycle."""
    d = g.d
    specs: list[BoundarySpec] = []

    # Interior checks.  Even-half X stabilizers move right; Z stabilizers move up.
    for x in range(1, d):
        for y in range(1, d):
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
                ikey(x, y),
            )

    # Bottom X boundary.
    n = 0
    while 2 + 2 * n <= d:
        x0 = 1 + 2 * n
        x1 = 2 + 2 * n
        _append_spec(
            specs,
            g,
            "X",
            [ikey(x0, 0), ikey(x1, 0)],
            [ikey(x0, 0), ikey(x1, 0)],
            [(x0 - 1, 0), (x1 - 1, 0)],
            ikey(x0, 0),
        )
        n += 1

    # Top X boundary.
    n = 0
    while 3 + 2 * n <= d:
        x0 = 2 + 2 * n
        x1 = 3 + 2 * n
        _append_spec(
            specs,
            g,
            "X",
            [ikey(x1, d)],
            [ikey(x0, d)],
            [(x0 - 1, d - 1), (x0, d - 1)],
            ikey(x0, d),
        )
        n += 1

    # Left Z boundary.
    n = 0
    while 3 + 2 * n <= d:
        y0 = 2 + 2 * n
        y1 = 3 + 2 * n
        _append_spec(
            specs,
            g,
            "Z",
            [ikey(0, y0), ikey(0, y1)],
            [ikey(0, y0), ikey(0, y1)],
            [(0, y0 - 1), (0, y1 - 1)],
            ikey(0, y0),
        )
        n += 1

    # Right Z boundary.
    n = 0
    while 2 + 2 * n <= d:
        y0 = 1 + 2 * n
        y1 = 2 + 2 * n
        _append_spec(
            specs,
            g,
            "Z",
            [ikey(d, y1)],
            [ikey(d, y0)],
            [(d - 1, y0 - 1), (d - 1, y0)],
            ikey(d, y0),
        )
        n += 1

    return specs


def append_detector(
    g: StandardTrivalentSurfaceCodeGenerator,
    lines: list[str],
    tracker: MeasurementTracker,
    refs: Iterable[tuple[int, CoordKey]],
    *,
    coord_keys: Iterable[CoordKey],
    time: int,
) -> None:
    ref_counts: dict[tuple[int, CoordKey], int] = {}
    for ref in refs:
        ref_counts[ref] = ref_counts.get(ref, 0) ^ 1
    terms = [tracker.rec(ref_time, key) for (ref_time, key), keep in ref_counts.items() if keep]
    if not terms:
        return

    coords = list(coord_keys)
    x = sum(key[0] / 2 for key in coords) / len(coords)
    y = sum(key[1] / 2 for key in coords) / len(coords)
    lines.append(f"DETECTOR({stim_num_text(x)}, {stim_num_text(y)}, {time}) {' '.join(terms)}")


def append_measurement_round(
    g: StandardTrivalentSurfaceCodeGenerator,
    lines: list[str],
    tracker: MeasurementTracker,
    *,
    time: int,
    second_half: bool,
) -> None:
    active = g.active_qubits()
    forward = g.forward_cx_layers()
    layers = g.second_cx_layers(forward) if second_half else forward
    prepare_h = g.h2_prepare() if second_half else g.h1_prepare()
    measure_h = g.h2_measure() if second_half else g.h1_measure()

    lines.append("TICK")
    lines.append("R " + " ".join(map(str, active)))
    lines.append("TICK")
    if prepare_h:
        lines.append("H " + " ".join(map(str, prepare_h)))
    for layer in layers:
        lines.append("TICK")
        cx = g.cx_line(layer)
        if cx is not None:
            lines.append(cx)
    lines.append("TICK")
    if measure_h:
        lines.append("H " + " ".join(map(str, measure_h)))
    lines.append("TICK")
    lines.append("M " + " ".join(map(str, active)))
    tracker.record_round(time, active)


def _has_round_key(tracker: MeasurementTracker, time: int, key: CoordKey) -> bool:
    return key in tracker.rounds.get(time, {})


def _append_if_present(
    g: StandardTrivalentSurfaceCodeGenerator,
    lines: list[str],
    tracker: MeasurementTracker,
    refs: Iterable[tuple[int, CoordKey]],
    *,
    coord_keys: Iterable[CoordKey],
    time: int,
) -> None:
    refs = tuple(refs)
    if all(_has_round_key(tracker, t, key) for t, key in refs):
        append_detector(g, lines, tracker, refs, coord_keys=coord_keys, time=time)


def append_detectors_between(
    g: StandardTrivalentSurfaceCodeGenerator,
    lines: list[str],
    tracker: MeasurementTracker,
    *,
    previous_time: int,
    current_time: int,
) -> None:
    if previous_time % 2 == 1 and current_time % 2 == 0:
        append_odd_to_even_detectors(g, lines, tracker, previous_time, current_time)
    elif previous_time % 2 == 0 and current_time % 2 == 1:
        append_even_to_odd_detectors(g, lines, tracker, previous_time, current_time)
    else:
        raise ValueError("detectors must connect adjacent odd/even half-cycles")


def append_odd_to_even_detectors(
    g: StandardTrivalentSurfaceCodeGenerator,
    lines: list[str],
    tracker: MeasurementTracker,
    odd_time: int,
    even_time: int,
) -> None:
    d = g.d

    for x in range(1, d):
        for y in range(1, d):
            current = ikey(x + 1, y) if (x + y) % 2 == 0 else ikey(x, y + 1)
            _append_if_present(
                g,
                lines,
                tracker,
                [(odd_time, ikey(x, y)), (even_time, current)],
                coord_keys=[current],
                time=even_time,
            )

    for n in range(1, d + 1):
        a = ikey(2 * n, 0)
        b = ikey(2 * n + 1, 0)
        _append_if_present(
            g,
            lines,
            tracker,
            [(odd_time, a), (odd_time, b), (even_time, a), (even_time, b)],
            coord_keys=[a, b],
            time=even_time,
        )

    for n in range(1, d + 1):
        y0 = 2 * n
        y1 = 2 * n + 1
        if 0 < y0 <= d and 0 < y1 <= d:
            _append_if_present(
                g,
                lines,
                tracker,
                [(odd_time, ikey(d, y0)), (even_time, ikey(d, y1))],
                coord_keys=[ikey(d, y1)],
                time=even_time,
            )

    for n in range(1, d):
        a = ikey(0, 2 * n - 1)
        b = ikey(0, 2 * n)
        _append_if_present(
            g,
            lines,
            tracker,
            [(odd_time, a), (odd_time, b), (even_time, a), (even_time, b)],
            coord_keys=[a, b],
            time=even_time,
        )

    for x0 in range(1, d, 2):
        x1 = x0 + 1
        _append_if_present(
            g,
            lines,
            tracker,
            [(odd_time, ikey(x0, d)), (even_time, ikey(x1, d))],
            coord_keys=[ikey(x1, d)],
            time=even_time,
        )


def append_even_to_odd_detectors(
    g: StandardTrivalentSurfaceCodeGenerator,
    lines: list[str],
    tracker: MeasurementTracker,
    even_time: int,
    odd_time: int,
) -> None:
    d = g.d

    for x in range(1, d):
        for y in range(1, d):
            current = ikey(x + 1, y) if (x + y) % 2 == 1 else ikey(x, y + 1)
            _append_if_present(
                g,
                lines,
                tracker,
                [(even_time, ikey(x, y)), (odd_time, current)],
                coord_keys=[current],
                time=odd_time,
            )

    for n in range(1, d):
        a = ikey(2 * n - 1, 0)
        b = ikey(2 * n, 0)
        _append_if_present(
            g,
            lines,
            tracker,
            [(even_time, a), (even_time, b), (odd_time, a), (odd_time, b)],
            coord_keys=[a, b],
            time=odd_time,
        )

    for n in range(1, d + 1):
        y0 = 2 * n - 1
        y1 = 2 * n
        if 0 < y0 <= d and 0 < y1 <= d:
            _append_if_present(
                g,
                lines,
                tracker,
                [(even_time, ikey(d, y0)), (odd_time, ikey(d, y1))],
                coord_keys=[ikey(d, y1)],
                time=odd_time,
            )

    for n in range(1, d):
        a = ikey(0, 2 * n)
        b = ikey(0, 2 * n + 1)
        _append_if_present(
            g,
            lines,
            tracker,
            [(even_time, a), (even_time, b), (odd_time, a), (odd_time, b)],
            coord_keys=[a, b],
            time=odd_time,
        )

    for x0 in range(2, d, 2):
        x1 = x0 + 1
        _append_if_present(
            g,
            lines,
            tracker,
            [(even_time, ikey(x0, d)), (odd_time, ikey(x1, d))],
            coord_keys=[ikey(x1, d)],
            time=odd_time,
        )


def append_memory_detector(
    g: StandardTrivalentSurfaceCodeGenerator,
    lines: list[str],
    tracker: MeasurementTracker,
    refs: Iterable[tuple[int, CoordKey]],
    coord_key: CoordKey,
    time: int,
) -> None:
    append_detector(g, lines, tracker, refs, coord_keys=[coord_key], time=time)


def build_memory_without_observable(
    distance: int,
    basis: str,
    *,
    logical_line: int | None = None,
    rounds: int | None = None,
) -> MemoryBuild:
    if basis not in ("X", "Z"):
        raise ValueError("basis must be X or Z")
    if logical_line is None:
        logical_line = distance - 1
    if not 0 <= logical_line < distance:
        raise ValueError("logical_line must satisfy 0 <= logical_line < distance")
    if rounds is None:
        rounds = distance
    if rounds < 1:
        raise ValueError("rounds must be >= 1")

    g = StandardTrivalentSurfaceCodeGenerator(distance, include_polygons=False)
    tracker = MeasurementTracker(g)
    lines = g.qubit_coord_lines()
    specs = odd_to_even_boundary_specs(g)
    data = data_qubits(g)

    lines.append("TICK")
    lines.append(("RX" if basis == "X" else "R") + " " + " ".join(map(str, data)))

    last_syndrome_time = rounds + 1
    for time in range(2, last_syndrome_time + 1):
        append_measurement_round(g, lines, tracker, time=time, second_half=(time % 2 == 0))
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
            append_detectors_between(g, lines, tracker, previous_time=time - 1, current_time=time)

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
        final_logical_indices = [tracker.rounds[final_time][hkey(logical_line, y)] for y in range(distance)]
    else:
        final_logical_indices = [tracker.rounds[final_time][hkey(x, logical_line)] for x in range(distance)]

    return MemoryBuild(
        distance=distance,
        basis=basis,
        logical_line=logical_line,
        measurement_rounds=rounds,
        text_without_observable="\n".join(lines) + "\n",
        tracker=tracker,
        final_time=final_time,
        before_final_data_measurements=before_final_data,
        final_logical_indices=final_logical_indices,
    )


def measurement_record_terms(build: MemoryBuild, indices: Iterable[int]) -> list[str]:
    total_measurements = stim.Circuit(build.text_without_observable).num_measurements
    return [f"rec[{index - total_measurements}]" for index in sorted(indices)]


def measurement_record_terms_from_row(build: MemoryBuild, row: int) -> list[str]:
    total_measurements = stim.Circuit(build.text_without_observable).num_measurements
    return [f"rec[{index - total_measurements}]" for index in range(total_measurements) if (row >> index) & 1]


def measurement_record_row_from_terms(build: MemoryBuild, terms: Iterable[str]) -> int:
    total_measurements = stim.Circuit(build.text_without_observable).num_measurements
    row = 0
    for term in terms:
        row ^= 1 << (total_measurements + int(term.removeprefix("rec[").removesuffix("]")))
    return row


def measurement_record_rows(text: str) -> tuple[list[int], int | None]:
    detector_rows: list[int] = []
    observable_row: int | None = None
    measurement_count = 0
    measurement_gates = {"M", "MX", "MY", "MR", "MRX", "MRY"}

    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        gate = line.split(" ", 1)[0].split("(", 1)[0]
        if gate in measurement_gates:
            measurement_count += len(line.split()) - 1
            continue
        if not (line.startswith("DETECTOR") or line.startswith("OBSERVABLE_INCLUDE")):
            continue

        row = 0
        for rec in re.findall(r"rec\[(-?\d+)\]", line):
            row ^= 1 << (measurement_count + int(rec))
        if line.startswith("DETECTOR"):
            detector_rows.append(row)
        else:
            observable_row = row

    return detector_rows, observable_row


def default_boundary_correction_indices(build: MemoryBuild) -> list[int]:
    if build.basis == "X":
        keys = [ikey(build.distance, y) for y in range(build.distance + 1)]
    else:
        keys = [ikey(x, build.distance) for x in range(build.distance + 1)]

    indices: list[int] = []
    for time in range(2, build.final_time):
        for key in keys:
            index = build.tracker.rounds[time].get(key)
            if index is not None:
                indices.append(index)
    return indices


def default_boundary_observable_row_and_deltas(build: MemoryBuild, *, shots: int) -> tuple[int, list[int]]:
    circuit = stim.Circuit(build.text_without_observable)
    if shots <= 0:
        shots = max(512, 4 * build.before_final_data_measurements)

    samples = circuit.compile_sampler().sample(shots).astype(np.uint8)
    target = np.bitwise_xor.reduce(samples[:, build.final_logical_indices], axis=1)
    correction_indices = default_boundary_correction_indices(build)
    correction_bits, nullspace = _solve_xor_system_with_nullspace(samples[:, correction_indices], target)

    row = 0
    for index in build.final_logical_indices:
        row ^= 1 << index
    for bit_index, measurement_index in enumerate(correction_indices):
        if (correction_bits >> bit_index) & 1:
            row ^= 1 << measurement_index

    delta_rows: list[int] = []
    for vector in nullspace:
        delta = 0
        for bit_index, measurement_index in enumerate(correction_indices):
            if (vector >> bit_index) & 1:
                delta ^= 1 << measurement_index
        delta_rows.append(delta)

    return row, delta_rows


def _on_default_boundary(build: MemoryBuild, x: float, y: float) -> bool:
    if build.basis == "X":
        return abs(x - build.distance) < 1e-9 or abs(x - (build.distance - 0.5)) < 1e-9
    return abs(y - build.distance) < 1e-9 or abs(y - (build.distance - 0.5)) < 1e-9


def _off_boundary_diagram_support(
    build: MemoryBuild,
    circuit: stim.Circuit,
    target: stim.GateTarget,
) -> set[tuple[int, int, str]]:
    support: set[tuple[int, int, str]] = set()
    for tick in range(circuit.num_ticks + 1):
        text = str(
            circuit.diagram(
                "detslice-text",
                tick=tick,
                filter_coords=[target],
            )
        )
        for line in text.splitlines():
            if ":D" not in line and ":L" not in line:
                continue
            match = re.search(r"q(\d+):\(([^,]+), ([^)]+)\).*([XYZ]):[DL]", line)
            if match is None:
                continue
            q = int(match.group(1))
            x = float(match.group(2))
            y = float(match.group(3))
            pauli = match.group(4)
            if not _on_default_boundary(build, x, y):
                support.add((tick, q, pauli))
    return support


def pin_observable_history_to_default_boundary(
    build: MemoryBuild,
    terms: list[str],
    *,
    delta_rows: list[int] | None = None,
) -> list[str]:
    text = append_observable(build, terms)
    circuit = stim.Circuit(text)
    detector_rows, observable_row = measurement_record_rows(text)
    if observable_row is None:
        raise ValueError("missing OBSERVABLE_INCLUDE row")

    observable_support = _off_boundary_diagram_support(
        build,
        circuit,
        stim.target_logical_observable_id(0),
    )
    if not observable_support:
        return terms

    if delta_rows:
        all_support = set(observable_support)
        delta_supports: list[set[tuple[int, int, str]]] = []
        for delta in delta_rows:
            delta_text = append_observable(build, measurement_record_terms_from_row(build, observable_row ^ delta))
            delta_circuit = stim.Circuit(delta_text)
            delta_circuit.detector_error_model(decompose_errors=True)
            support = _off_boundary_diagram_support(
                build,
                delta_circuit,
                stim.target_logical_observable_id(0),
            ) ^ observable_support
            delta_supports.append(support)
            all_support |= support

        ordered_support = sorted(all_support)
        support_index = {key: index for index, key in enumerate(ordered_support)}
        matrix = np.zeros((len(ordered_support), len(delta_supports)), dtype=np.uint8)
        target = np.zeros(len(ordered_support), dtype=np.uint8)

        for key in observable_support:
            target[support_index[key]] = 1
        for col, support in enumerate(delta_supports):
            for key in support:
                matrix[support_index[key], col] = 1

        try:
            delta_bits, _ = _solve_xor_system_with_nullspace(matrix, target)
        except ValueError:
            delta_bits = 0
        else:
            row = observable_row
            for index, delta in enumerate(delta_rows):
                if (delta_bits >> index) & 1:
                    row ^= delta
            pinned_terms = measurement_record_terms_from_row(build, row)
            pinned_text = append_observable(build, pinned_terms)
            pinned_circuit = stim.Circuit(pinned_text)
            pinned_circuit.detector_error_model(decompose_errors=True)
            if not _off_boundary_diagram_support(build, pinned_circuit, stim.target_logical_observable_id(0)):
                return pinned_terms

    detector_supports = [
        _off_boundary_diagram_support(build, circuit, stim.target_relative_detector_id(i))
        for i in range(circuit.num_detectors)
    ]
    all_support = sorted(set().union(observable_support, *detector_supports))
    support_index = {key: index for index, key in enumerate(all_support)}
    matrix = np.zeros((len(all_support), len(detector_supports)), dtype=np.uint8)
    target = np.zeros(len(all_support), dtype=np.uint8)

    for key in observable_support:
        target[support_index[key]] = 1
    for col, support in enumerate(detector_supports):
        for key in support:
            matrix[support_index[key], col] = 1

    try:
        detector_bits, _ = _solve_xor_system_with_nullspace(matrix, target)
    except ValueError:
        return terms

    row = observable_row
    for index, detector_row in enumerate(detector_rows):
        if (detector_bits >> index) & 1:
            row ^= detector_row

    pinned_terms = measurement_record_terms_from_row(build, row)
    pinned_text = append_observable(build, pinned_terms)
    pinned_circuit = stim.Circuit(pinned_text)
    pinned_circuit.detector_error_model(decompose_errors=True)
    if _off_boundary_diagram_support(build, pinned_circuit, stim.target_logical_observable_id(0)):
        return terms
    return pinned_terms


def default_observable_terms(build: MemoryBuild, *, shots: int = 0) -> list[str]:
    """Keep the default logical representative on the marked boundary."""
    if build.logical_line != build.distance - 1:
        raise ValueError("default observable is only defined for logical_line=distance-1")

    row, delta_rows = default_boundary_observable_row_and_deltas(build, shots=shots)
    terms = measurement_record_terms_from_row(build, row)
    text = append_observable(build, terms)
    stim.Circuit(text).detector_error_model(decompose_errors=True)
    return pin_observable_history_to_default_boundary(build, terms, delta_rows=delta_rows)


def resolved_observable_method(build: MemoryBuild, requested_method: str) -> str:
    if requested_method not in ("default", "gf2"):
        raise ValueError("observable_method must be default or gf2")
    if requested_method == "default" and build.logical_line != build.distance - 1:
        return "gf2"
    return requested_method


def choose_observable_terms(build: MemoryBuild, *, method: str = "default", shots: int = 0) -> list[str]:
    method = resolved_observable_method(build, method)
    if method == "default":
        return default_observable_terms(build, shots=shots)
    return infer_observable_terms(build, shots=shots)


def infer_observable_terms(
    build: MemoryBuild,
    *,
    shots: int,
    correction_indices: list[int] | None = None,
    allow_unrestricted_fallback: bool = True,
) -> list[str]:
    circuit = stim.Circuit(build.text_without_observable)
    total_measurements = circuit.num_measurements
    if shots <= 0:
        shots = max(512, 4 * build.before_final_data_measurements)

    samples = circuit.compile_sampler().sample(shots).astype(np.uint8)
    final_logical = set(build.final_logical_indices)
    target = np.bitwise_xor.reduce(samples[:, build.final_logical_indices], axis=1)
    detector_span = _detector_span_basis(build.text_without_observable)
    seed_bits = 0
    for index in build.final_logical_indices:
        seed_bits ^= 1 << index

    def expand(bits: int, correction_indices: list[int]) -> int:
        row = seed_bits
        while bits:
            bit = bits & -bits
            row ^= 1 << correction_indices[bit.bit_length() - 1]
            bits ^= bit
        return row

    def terms_from_row(row: int) -> list[str]:
        indices = [i for i in range(total_measurements) if (row >> i) & 1]
        return [f"rec[{i - total_measurements}]" for i in indices]

    def row_is_valid_observable(row: int) -> bool:
        if _is_in_span(row, detector_span):
            return False
        terms = terms_from_row(row)
        trial_text = append_observable(build, terms)
        try:
            stim.Circuit(trial_text).detector_error_model(decompose_errors=True)
        except ValueError:
            return False
        return True

    def choose_row(correction_indices: list[int]) -> int | None:
        correction_samples = samples[:, correction_indices]
        try:
            correction_bits, nullspace = _solve_xor_system_with_nullspace(correction_samples, target)
        except ValueError:
            return None

        chosen_bits = correction_bits
        chosen_row = expand(chosen_bits, correction_indices)
        if row_is_valid_observable(chosen_row):
            return chosen_row

        candidates = sorted(nullspace, key=int.bit_count)
        for delta in candidates:
            trial_bits = correction_bits ^ delta
            trial_row = expand(trial_bits, correction_indices)
            if not row_is_valid_observable(trial_row):
                continue
            return trial_row

        limit = min(64, len(candidates))
        for a in range(limit):
            for b in range(a + 1, limit):
                trial_bits = correction_bits ^ candidates[a] ^ candidates[b]
                trial_row = expand(trial_bits, correction_indices)
                if not row_is_valid_observable(trial_row):
                    continue
                return trial_row

        rng = random.Random(0x5EED + build.final_time)
        for density in (0.08, 0.16, 0.32):
            for _ in range(2048):
                trial_bits = correction_bits
                for vector in nullspace:
                    if rng.random() < density:
                        trial_bits ^= vector
                trial_row = expand(trial_bits, correction_indices)
                if not row_is_valid_observable(trial_row):
                    continue
                return trial_row
        return None

    if correction_indices is None:
        # Prefer observables whose final data support is exactly the requested
        # logical string.  If this is impossible, fall back to the older
        # unrestricted GF2 search, which may use other final data records.
        chosen_row = choose_row(list(range(build.before_final_data_measurements)))
    else:
        chosen_row = choose_row(correction_indices)
    if chosen_row is None and allow_unrestricted_fallback:
        unrestricted = [i for i in range(total_measurements) if i not in final_logical]
        chosen_row = choose_row(unrestricted)
    if chosen_row is None:
        raise ValueError("failed to infer a non-detector observable; increase --observable-shots")

    return terms_from_row(chosen_row)


def append_observable(build: MemoryBuild, terms: list[str]) -> str:
    return build.text_without_observable + "OBSERVABLE_INCLUDE(0) " + " ".join(terms) + "\n"


def generate_memory_circuit(
    distance: int,
    basis: str,
    *,
    p1: float = 0.0,
    p2: float = 0.0,
    observable_shots: int = 0,
    observable_method: str = "default",
    logical_line: int | None = None,
    rounds: int | None = None,
) -> str:
    build = build_memory_without_observable(distance, basis, logical_line=logical_line, rounds=rounds)
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
    parser = argparse.ArgumentParser(description="Generate/simulate standard trivalent-code memory experiments.")
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
        help="default uses the marked corner rule; gf2 restores the sampled GF2 search",
    )
    parser.add_argument(
        "--logical-line",
        type=int,
        default=None,
        help="final data column/row index for a selected observable; X selects a column, Z selects a row",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build = build_memory_without_observable(
        args.distance,
        args.basis,
        logical_line=args.logical_line,
        rounds=args.rounds,
    )
    actual_observable_method = resolved_observable_method(build, args.observable_method)
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
    print(f"logical_line={build.logical_line}")
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

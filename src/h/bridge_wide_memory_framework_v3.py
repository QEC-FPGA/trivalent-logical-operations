#!/usr/bin/env python3
"""ABANDONED as an experiment entry point / historical framework.

Use ``bridge_wide_memory_user_spec.py`` for the current bridge-wide memory
experiment. This module is retained for reference and for helper definitions
that the current implementation may import until they are factored out.

Generate and simulate the bridge wide-code memory experiment.

This code family is the wide-to-lower-standard bridge proposed for the H-flow.
Compared with ``wide_h_memory.py``, the corner half qubit
``(0.5, 2d - 0.5)`` is promoted to a data qubit and the integer qubit
``(0, 2d)`` is included as a reset/measured auxiliary.  The first half-cycle is
the pattern supplied for d=3; the second half-cycle uses the inverse CNOT
order with the two H layers swapped.  The original memory schedule starts with the second half-cycle at time 2.
This file additionally provides a strictly specified detector workflow whose
syndrome times start at 1: odd times use the user's first half-cycle and even
times use the user's second half-cycle.
"""

from __future__ import annotations

import argparse
import itertools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pymatching
import stim

from wide_h_generator import BLUE, RED, MeasurementTracker, Point, coord_text, stim_num_text
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
    protected_observable_row: int | None = None


@dataclass(frozen=True)
class ShortestGraphlikeErrorReport:
    weight: int | None
    error_lines: list[str]
    representative_locations: list[str]
    failure: str | None = None


@dataclass(frozen=True)
class RoundProduct:
    signature: int
    coeffs: int


@dataclass(frozen=True)
class StrictFlowRelation:
    """A local equality between strict stabilizer variables in adjacent halves.

    ``previous`` names refer to the earlier syndrome half-cycle; ``current``
    names refer to the later one.  The names are intentionally geometric:
    ``ZP(x,y)``/``XP(x,y)`` are four-body plaquettes with lower-left data
    coordinate ``(x+0.5, y+0.5)``, ``ZD`` is the lower Z boundary, ``ZRU`` is
    the upper part of the right Z boundary, ``XRD`` is the lower part of the
    right X boundary, and ``XL`` is the left X boundary.
    """

    previous: tuple[str, ...]
    current: tuple[str, ...]


@dataclass(frozen=True)
class StrictMeasuredCheck:
    name: str
    color: str
    qubits: tuple[int, ...]
    center: tuple[float, float]
    record_row: int
    kind: str = "strict"


@dataclass(frozen=True)
class ExplicitDetectorRelation:
    """One repeat detector written directly in auxiliary-coordinate form.

    Coordinates are integer auxiliary coordinates in exactly the order used by
    the user's formulas.  No geometric completion, symmetry substitution, or
    inferred shift is applied by this class.
    """

    family: str
    previous_time: int
    current_time: int
    previous_keys: tuple[CoordKey, ...]
    current_keys: tuple[CoordKey, ...]


@dataclass(frozen=True)
class SpecifiedFirstHalfStabilizer:
    """One first-half stabilizer in the user's explicit final-closure form."""

    name: str
    color: str
    auxiliary_keys: tuple[CoordKey, ...]
    final_data_keys: tuple[CoordKey, ...]
    center: tuple[float, float]


def is_second_half_time(time: int, *, start_second_half: bool = True) -> bool:
    """Return which half-cycle is used at a syndrome time."""
    return time % 2 == 0 if start_second_half else time % 2 == 1


class BridgeWideGenerator:
    """Generator for the bridge wide-code syndrome-extraction round."""

    def __init__(self, distance: int, *, include_polygons: bool = True) -> None:
        if distance < 3 or distance % 2 != 1:
            raise ValueError("distance must be an odd integer >= 3")
        self.d = distance
        self.include_polygons = include_polygons
        self.coords: dict[int, Point] = {}
        self.integer: dict[tuple[int, int], int] = {}
        self.half: dict[tuple[int, int], int] = {}
        self._build_qubits()
        self._round_measurement_signatures: dict[bool, dict[CoordKey, tuple[int, int]]] = {}

    def _build_qubits(self) -> None:
        q = 0
        for column2 in range(2 * self.d + 1):
            if column2 % 2 == 0:
                x = column2 // 2
                y_values = range(1, 2 * self.d + 1) if x == 0 else range(0, 2 * self.d + 1)
                for y in y_values:
                    self.integer[(x, y)] = q
                    self.coords[q] = Point(2 * x, 2 * y)
                    q += 1
            else:
                x = column2 // 2
                for y in range(2 * self.d):
                    self.half[(x, y)] = q
                    self.coords[q] = Point(2 * x + 1, 2 * y + 1)
                    q += 1

    def int_q(self, x: int, y: int) -> int | None:
        return self.integer.get((x, y))

    def half_q(self, x: int, y: int) -> int | None:
        return self.half.get((x, y))

    @staticmethod
    def ikey(x: int, y: int) -> CoordKey:
        return ikey(x, y)

    def active_qubits(self) -> list[int]:
        return sorted(self.integer.values())

    def data_qubits(self) -> list[int]:
        return sorted(self.half.values())

    def data_keys(self) -> list[CoordKey]:
        return [hkey(x, y) for x, y in sorted(self.half)]

    def h1_prepare(self) -> list[int]:
        qs: set[int] = set()

        # Bottom row and the new left X boundary are prepared through H.
        for x in range(1, self.d + 1):
            qs.add(self.integer[(x, 0)])
        for y in range(1, 2 * self.d + 1):
            qs.add(self.integer[(0, y)])

        # Interior checkerboard from the supplied d=3 round.
        for x in range(1, self.d):
            for y in range(1, 2 * self.d):
                if (x + y) % 2 == 1:
                    qs.add(self.integer[(x, y)])

        # Upper-right boundary follows the old wide-code convention, except
        # the top-right corner y=2d is no longer in this H layer.
        for y in range(1, 2 * self.d):
            if y <= self.d:
                use = (self.d + y) % 2 == 1
            else:
                use = True
            if use:
                qs.add(self.integer[(self.d, y)])

        return sorted(qs)

    def h1_measure(self) -> list[int]:
        qs: set[int] = set()

        for x in range(1, self.d + 1):
            qs.add(self.integer[(x, 0)])
        for y in range(1, 2 * self.d + 1):
            qs.add(self.integer[(0, y)])

        for x in range(1, self.d):
            for y in range(1, 2 * self.d):
                # At the end of this bridge period the code-internal
                # stabilizer ancillas use the opposite H layer from the
                # beginning.  Boundary ancillas keep their separate rules.
                if (x + y) % 2 == 0:
                    qs.add(self.integer[(x, y)])

        for y in range(1, 2 * self.d):
            if y < self.d:
                use = (self.d + y) % 2 == 0
            else:
                use = True
            if use:
                qs.add(self.integer[(self.d, y)])

        return sorted(qs)

    def h2_prepare(self) -> list[int]:
        qs = set(self.h1_measure())
        # The second half-cycle expands the upper/right corner.  Without this
        # H, the corner reports a spurious deterministic one-body Z while the
        # intended right-edge two-body X carries reset randomness.
        qs.add(self.integer[(self.d, 2 * self.d)])
        return sorted(qs)

    def h2_measure(self) -> list[int]:
        return self.h1_prepare()

    @staticmethod
    def _check_matching(pairs: Iterable[tuple[int, int]], label: str) -> None:
        used: set[int] = set()
        for a, b in pairs:
            if a in used or b in used:
                raise AssertionError(f"{label} is not a disjoint CNOT layer")
            used.add(a)
            used.add(b)

    def forward_cx_layers(self) -> list[list[tuple[int, int]]]:
        layer1: list[tuple[int, int]] = []
        layer2: list[tuple[int, int]] = []
        layer3: list[tuple[int, int]] = []
        layer4: list[tuple[int, int]] = []

        for y in range(1, 2 * self.d + 1):
            for x in range(1, self.d + 1):
                iq = self.integer[(x, y)]
                hq = self.half[(x - 1, y - 1)]
                if (x + y) % 2 == 0:
                    layer1.append((hq, iq))
                    layer4.append((iq, hq))
                else:
                    layer1.append((iq, hq))
                    layer4.append((hq, iq))

        for y in range(0, 2 * self.d):
            for x in range(1, self.d + 1):
                iq = self.integer[(x, y)]
                hq = self.half[(x - 1, y)]
                if (x + y) % 2 == 0:
                    layer3.append((iq, hq))
                else:
                    layer2.append((iq, hq))

        for y in range(1, 2 * self.d + 1):
            for x in range(0, self.d):
                iq = self.int_q(x, y)
                hq = self.half_q(x, y - 1)
                if iq is None or hq is None:
                    continue
                if (x + y) % 2 == 0:
                    layer2.append((hq, iq))
                else:
                    layer3.append((hq, iq))

        layers = [layer1, layer2, layer3, layer4]
        for index, layer in enumerate(layers, start=1):
            self._check_matching(layer, f"forward layer {index}")
        return layers

    def second_cx_layers(self, forward: list[list[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
        layers = [list(layer) for layer in reversed(forward)]
        for index, layer in enumerate(layers, start=1):
            self._check_matching(layer, f"second layer {index}")
        return layers

    def qubit_coord_lines(self) -> list[str]:
        lines: list[str] = []
        for q in sorted(self.coords):
            p = self.coords[q]
            lines.append(f"QUBIT_COORDS({coord_text(p.x2)}, {coord_text(p.y2)}) {q}")
        return lines

    def _strict_stabilizer_groups(self, *, second_half: bool) -> tuple[list[list[int]], list[list[int]]]:
        h = self.half
        blue: list[list[int]] = []
        red: list[list[int]] = []

        def add_z(qs: list[int]) -> None:
            blue.append(qs)

        def add_x(qs: list[int]) -> None:
            red.append(qs)

        # Interior four-body stabilizers.  There are
        # (d-1)(2d-1)/2 of each type.
        for y in range(2 * self.d - 1):
            for x in range(self.d - 1):
                is_z = (x + y) % 2 == (1 if second_half else 0)
                qs = [h[(x, y)], h[(x + 1, y)], h[(x + 1, y + 1)], h[(x, y + 1)]]
                (add_z if is_z else add_x)(qs)

        # Coordinate convention here follows the user's boundary language:
        # y=0 is the upper boundary and y=2d-1 is the lower boundary.

        # Lower boundary: (d-1)/2 two-body Z.
        lower_start = 0 if second_half else 1
        for x in range(lower_start, self.d - 1, 2):
            add_z([h[(x, 2 * self.d - 1)], h[(x + 1, 2 * self.d - 1)]])

        # Right boundary upper side: (d-1)/2 two-body Z.
        right_z_start = 1 if second_half else 0
        for y in range(right_z_start, self.d - 1, 2):
            add_z([h[(self.d - 1, y)], h[(self.d - 1, y + 1)]])

        # Right boundary lower side: (d-1)/2 two-body X.
        right_x_start = self.d + 1 if second_half else self.d
        for y in range(right_x_start, 2 * self.d - 1, 2):
            add_x([h[(self.d - 1, y)], h[(self.d - 1, y + 1)]])

        if second_half:
            # Second half: left boundary has d-1 two-body X, upper boundary
            # has (d-1)/2 two-body X, and the upper-left corner is one-body X.
            for y in range(1, 2 * self.d - 1, 2):
                add_x([h[(0, y)], h[(0, y + 1)]])
            for x in range(1, self.d - 1, 2):
                add_x([h[(x, 0)], h[(x + 1, 0)]])
            add_x([h[(0, 0)]])
        else:
            # First half: left boundary has d two-body X and upper boundary
            # has (d-1)/2 two-body X.
            for y in range(0, 2 * self.d, 2):
                add_x([h[(0, y)], h[(0, y + 1)]])
            for x in range(0, self.d - 1, 2):
                add_x([h[(x, 0)], h[(x + 1, 0)]])

        return blue, red

    def polygon_lines(self, *, after_h: bool) -> list[str]:
        blue, red = self._strict_stabilizer_groups(second_half=after_h)
        lines: list[str] = []
        for color, groups in ((BLUE, blue), (RED, red)):
            for qs in groups:
                lines.append(f"#!pragma POLYGON({color}) {' '.join(map(str, qs))}")
        return lines

    @staticmethod
    def op_line(name: str, qubits: Iterable[int]) -> str | None:
        qs = list(qubits)
        if not qs:
            return None
        return f"{name} {' '.join(map(str, qs))}"

    @staticmethod
    def cx_line(pairs: Iterable[tuple[int, int]]) -> str | None:
        flat: list[int] = []
        for control, target in pairs:
            flat.extend([control, target])
        if not flat:
            return None
        return f"CX {' '.join(map(str, flat))}"

    def append_tick(self, lines: list[str]) -> None:
        lines.append("TICK")

    def append_op(self, lines: list[str], name: str, qubits: Iterable[int]) -> None:
        line = self.op_line(name, qubits)
        if line is not None:
            lines.append(line)

    def append_cx_layers(self, lines: list[str], layers: Iterable[list[tuple[int, int]]]) -> None:
        for layer in layers:
            self.append_tick(lines)
            line = self.cx_line(layer)
            if line is not None:
                lines.append(line)

    def round_unitary_circuit(self, *, second_half: bool) -> stim.Circuit:
        lines: list[str] = []
        forward = self.forward_cx_layers()
        layers = self.second_cx_layers(forward) if second_half else forward
        prepare_h = self.h2_prepare() if second_half else self.h1_prepare()
        measure_h = self.h2_measure() if second_half else self.h1_measure()
        self.append_op(lines, "H", prepare_h)
        for layer in layers:
            line = self.cx_line(layer)
            if line is not None:
                lines.append(line)
        self.append_op(lines, "H", measure_h)
        return stim.Circuit("\n".join(lines))

    def append_measurement_round(
        self,
        lines: list[str],
        tracker: MeasurementTracker | None,
        *,
        time: int,
        second_half: bool,
    ) -> None:
        active = self.active_qubits()
        forward = self.forward_cx_layers()
        layers = self.second_cx_layers(forward) if second_half else forward
        prepare_h = self.h2_prepare() if second_half else self.h1_prepare()
        measure_h = self.h2_measure() if second_half else self.h1_measure()

        if self.include_polygons:
            lines.extend(self.polygon_lines(after_h=second_half))

        self.append_tick(lines)
        self.append_op(lines, "R", active)
        self.append_tick(lines)
        self.append_op(lines, "H", prepare_h)
        self.append_cx_layers(lines, layers)
        self.append_tick(lines)
        self.append_op(lines, "H", measure_h)
        self.append_tick(lines)
        self.append_op(lines, "M", active)
        if tracker is not None:
            tracker.record_round(time, active)

    def round_measurement_signatures(self, *, second_half: bool) -> dict[CoordKey, tuple[int, int]]:
        if second_half in self._round_measurement_signatures:
            return self._round_measurement_signatures[second_half]

        unitary = self.round_unitary_circuit(second_half=second_half)
        data = self.data_qubits()
        q_to_data_index = {q: i for i, q in enumerate(data)}
        active = self.active_qubits()
        q_to_aux_index = {q: i for i, q in enumerate(active)}
        signatures: dict[CoordKey, tuple[int, int]] = {}
        num_qubits = max(self.coords) + 1

        for q in active:
            p = stim.PauliString(num_qubits)
            p[q] = "Z"
            before = p.before(unitary)

            data_signature = 0
            for dq, index in q_to_data_index.items():
                value = before[dq]
                if value in (1, 2):
                    data_signature ^= 1 << index
                if value in (2, 3):
                    data_signature ^= 1 << (len(data) + index)

            aux_random_signature = 0
            for aq, index in q_to_aux_index.items():
                value = before[aq]
                if value in (1, 2):
                    aux_random_signature ^= 1 << index

            key = self.coords[q]
            signatures[(key.x2, key.y2)] = (data_signature, aux_random_signature)

        self._round_measurement_signatures[second_half] = signatures
        return signatures

    def append_detector_terms(
        self,
        lines: list[str],
        terms: list[str],
        signature: int,
        *,
        time: int,
    ) -> None:
        if not terms:
            return
        keys = signature_data_keys(self, signature)
        if keys:
            x = sum(k[0] / 2 for k in keys) / len(keys)
            y = sum(k[1] / 2 for k in keys) / len(keys)
        else:
            x = y = 0
        lines.append(f"DETECTOR({stim_num_text(x)}, {stim_num_text(y)}, {time}) {' '.join(terms)}")


def data_qubits(g: BridgeWideGenerator) -> list[int]:
    return g.data_qubits()


def _measurement_count_for_line(parts: list[str]) -> int:
    op = parts[0]
    if op in ("M", "MX", "MY", "MR", "MRX", "MRY"):
        return len(parts) - 1
    return 0


def _row_from_indices(indices: Iterable[int]) -> int:
    row = 0
    for index in indices:
        row ^= 1 << index
    return row


def _sample_rows_to_bit_rows(samples: np.ndarray, cols: list[int]) -> list[int]:
    rows: list[int] = []
    for sample in samples[:, cols]:
        rows.append(_bits_to_int(int(v) for v in sample))
    return rows


def _bits_to_int(bits: Iterable[int]) -> int:
    value = 0
    for index, bit in enumerate(bits):
        if bit:
            value |= 1 << index
    return value


def _gf2_rref(rows: Iterable[int], ncols: int) -> tuple[list[int], list[int]]:
    rows = [row for row in rows if row]
    pivots: list[int] = []
    rank = 0
    for col in range(ncols):
        bit = 1 << col
        pivot = None
        for row_index in range(rank, len(rows)):
            if rows[row_index] & bit:
                pivot = row_index
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for row_index in range(len(rows)):
            if row_index != rank and (rows[row_index] & bit):
                rows[row_index] ^= rows[rank]
        pivots.append(col)
        rank += 1
        if rank == len(rows):
            break
    return rows[:rank], pivots


def _nullspace(rows: Iterable[int], ncols: int) -> list[int]:
    rref, pivots = _gf2_rref(rows, ncols)
    pivot_set = set(pivots)
    basis: list[int] = []
    for free in range(ncols):
        if free in pivot_set:
            continue
        vector = 1 << free
        for row, pivot in zip(rref, pivots):
            if row & (1 << free):
                vector |= 1 << pivot
        basis.append(vector)
    return basis


def _add_to_basis(row: int, basis: dict[int, int]) -> bool:
    trial = row
    while trial:
        pivot = trial.bit_length() - 1
        if pivot not in basis:
            basis[pivot] = trial
            return True
        trial ^= basis[pivot]
    return False


def _xor_selected(rows: list[int], coeffs: int) -> int:
    out = 0
    index = 0
    while coeffs:
        bit = coeffs & -coeffs
        index = bit.bit_length() - 1
        out ^= rows[index]
        coeffs ^= bit
    return out


def intersection_relations(rows_a: list[int], rows_b: list[int]) -> list[tuple[int, int, int]]:
    """Return independent shared signatures with coeffs in A and B."""
    ncols = len(rows_a) + len(rows_b)
    all_rows = rows_a + rows_b
    max_bit = max([row.bit_length() for row in all_rows] or [0])
    constraints: list[int] = []
    for bit_index in range(max_bit):
        row = 0
        mask = 1 << bit_index
        for col, sig in enumerate(all_rows):
            if sig & mask:
                row |= 1 << col
        constraints.append(row)

    sig_basis: dict[int, int] = {}
    out: list[tuple[int, int, int]] = []
    for relation in _nullspace(constraints, ncols):
        a = relation & ((1 << len(rows_a)) - 1)
        b = relation >> len(rows_a)
        sig = _xor_selected(rows_a, a)
        if sig == 0:
            continue
        if _add_to_basis(sig, sig_basis):
            out.append((sig, a, b))
    return out


def basis_data_rows(g: BridgeWideGenerator, basis: str) -> list[int]:
    n = len(g.data_qubits())
    rows: list[int] = []
    for index in range(n):
        if basis == "X":
            rows.append(1 << index)
        elif basis == "Z":
            rows.append(1 << (n + index))
        else:
            raise ValueError("basis must be X or Z")
    return rows


def signature_data_keys(g: BridgeWideGenerator, signature: int) -> list[CoordKey]:
    keys = g.data_keys()
    n = len(keys)
    out: list[CoordKey] = []
    for index, key in enumerate(keys):
        if ((signature >> index) & 1) or ((signature >> (n + index)) & 1):
            out.append(key)
    return out


def _terms_from_coeffs(
    tracker: MeasurementTracker,
    time: int,
    keys: list[CoordKey],
    coeffs: int,
) -> list[str]:
    terms: list[str] = []
    for index, key in enumerate(keys):
        if (coeffs >> index) & 1:
            terms.append(tracker.rec(time, key))
    return terms


def deterministic_round_products(
    g: BridgeWideGenerator,
    keys: list[CoordKey],
    *,
    second_half: bool,
) -> list[RoundProduct]:
    """Return independent data products whose same-round reset noise cancels."""
    sigs_by_key = g.round_measurement_signatures(second_half=second_half)
    data_rows = [sigs_by_key[key][0] for key in keys]
    aux_rows = [sigs_by_key[key][1] for key in keys]
    max_aux_bit = max([row.bit_length() for row in aux_rows] or [0])

    constraints: list[int] = []
    for bit_index in range(max_aux_bit):
        row = 0
        mask = 1 << bit_index
        for col, sig in enumerate(aux_rows):
            if sig & mask:
                row |= 1 << col
        constraints.append(row)

    sig_basis: dict[int, int] = {}
    products: list[RoundProduct] = []
    for coeffs in _nullspace(constraints, len(keys)):
        sig = _xor_selected(data_rows, coeffs)
        if sig == 0:
            continue
        if _add_to_basis(sig, sig_basis):
            products.append(RoundProduct(signature=sig, coeffs=coeffs))
    return products


def _combine_product_coeffs(products: list[RoundProduct], coeffs: int) -> int:
    out = 0
    for index, product in enumerate(products):
        if (coeffs >> index) & 1:
            out ^= product.coeffs
    return out


def append_initial_detectors(
    g: BridgeWideGenerator,
    lines: list[str],
    tracker: MeasurementTracker,
    *,
    time: int,
    basis: str,
    second_half: bool,
) -> None:
    sigs_by_key = g.round_measurement_signatures(second_half=second_half)
    keys = [key for key in tracker.rounds[time] if key in sigs_by_key]
    products = deterministic_round_products(g, keys, second_half=second_half)
    current_rows = [product.signature for product in products]
    init_rows = basis_data_rows(g, basis)
    for sig, product_coeffs, _ in intersection_relations(current_rows, init_rows):
        current_coeffs = _combine_product_coeffs(products, product_coeffs)
        terms = _terms_from_coeffs(tracker, time, keys, current_coeffs)
        g.append_detector_terms(lines, terms, sig, time=time)


def append_detectors_between(
    g: BridgeWideGenerator,
    lines: list[str],
    tracker: MeasurementTracker,
    *,
    previous_time: int,
    current_time: int,
    previous_second_half: bool,
    current_second_half: bool,
) -> None:
    prev_sigs = g.round_measurement_signatures(second_half=previous_second_half)
    curr_sigs = g.round_measurement_signatures(second_half=current_second_half)
    prev_keys = [key for key in tracker.rounds[previous_time] if key in prev_sigs]
    curr_keys = [key for key in tracker.rounds[current_time] if key in curr_sigs]
    prev_products = deterministic_round_products(g, prev_keys, second_half=previous_second_half)
    curr_products = deterministic_round_products(g, curr_keys, second_half=current_second_half)
    prev_rows = [product.signature for product in prev_products]
    curr_rows = [product.signature for product in curr_products]
    for sig, prev_product_coeffs, curr_product_coeffs in intersection_relations(prev_rows, curr_rows):
        prev_coeffs = _combine_product_coeffs(prev_products, prev_product_coeffs)
        curr_coeffs = _combine_product_coeffs(curr_products, curr_product_coeffs)
        terms = _terms_from_coeffs(tracker, previous_time, prev_keys, prev_coeffs)
        terms += _terms_from_coeffs(tracker, current_time, curr_keys, curr_coeffs)
        g.append_detector_terms(lines, terms, sig, time=current_time)


def append_final_detectors(
    g: BridgeWideGenerator,
    lines: list[str],
    tracker: MeasurementTracker,
    *,
    previous_time: int,
    final_time: int,
    previous_second_half: bool,
    basis: str,
) -> None:
    prev_sigs = g.round_measurement_signatures(second_half=previous_second_half)
    prev_keys = [key for key in tracker.rounds[previous_time] if key in prev_sigs]
    prev_products = deterministic_round_products(g, prev_keys, second_half=previous_second_half)
    prev_rows = [product.signature for product in prev_products]
    final_keys = g.data_keys()
    final_rows = basis_data_rows(g, basis)
    for sig, prev_product_coeffs, final_coeffs in intersection_relations(prev_rows, final_rows):
        prev_coeffs = _combine_product_coeffs(prev_products, prev_product_coeffs)
        terms = _terms_from_coeffs(tracker, previous_time, prev_keys, prev_coeffs)
        terms += _terms_from_coeffs(tracker, final_time, final_keys, final_coeffs)
        g.append_detector_terms(lines, terms, sig, time=final_time)


def _measurement_index_metadata(g: BridgeWideGenerator, tracker: MeasurementTracker) -> dict[int, tuple[int, CoordKey]]:
    out: dict[int, tuple[int, CoordKey]] = {}
    for time, mapping in tracker.rounds.items():
        for key, index in mapping.items():
            out[index] = (time, key)
    return out


def _detector_line_from_indices(
    g: BridgeWideGenerator,
    metadata: dict[int, tuple[int, CoordKey]],
    indices: list[int],
    *,
    total_measurements: int,
    time: int,
) -> str:
    terms = [f"rec[{index - total_measurements}]" for index in sorted(indices)]
    keys = [metadata[index][1] for index in indices if index in metadata]
    if keys:
        x = sum(key[0] / 2 for key in keys) / len(keys)
        y = sum(key[1] / 2 for key in keys) / len(keys)
    else:
        x = y = 0
    return f"DETECTOR({stim_num_text(x)}, {stim_num_text(y)}, {time}) {' '.join(terms)}"


def _rebuild_detector_basis(lines: list[str], total_measurements: int) -> dict[int, int]:
    basis: dict[int, int] = {}
    for line in lines:
        row = 0
        for offset_text in re.findall(r"rec\[(-?\d+)\]", line):
            row ^= 1 << (total_measurements + int(offset_text))
        _add_to_basis(row, basis)
    return basis


def infer_local_detector_lines(
    g: BridgeWideGenerator,
    tracker: MeasurementTracker,
    raw_text: str,
    *,
    shots: int = 0,
    protected_rows: Iterable[int] = (),
) -> list[str]:
    """Infer local deterministic detectors and exact-filter them with Stim.

    The bridge round is gauge-like: individual auxiliary measurements may carry
    random same-round reset components.  Instead of hand-writing every local
    product, this routine samples local measurement windows to propose parity
    checks, then keeps only checks accepted by Stim's exact detector analysis.
    """
    circuit = stim.Circuit(raw_text)
    total = circuit.num_measurements
    if shots <= 0:
        shots = max(8192, 16 * total)
    samples = circuit.compile_sampler(seed=0xB71D6E).sample(shots).astype(np.uint8)
    metadata = _measurement_index_metadata(g, tracker)
    protected_rows = tuple(protected_rows)

    times = sorted(tracker.rounds)
    syndrome_times = times[:-1]
    final_time = times[-1]
    windows: list[tuple[int, list[int]]] = []
    if syndrome_times:
        windows.append((syndrome_times[0], list(tracker.rounds[syndrome_times[0]].values())))
    for previous, current in zip(syndrome_times, syndrome_times[1:]):
        windows.append((current, list(tracker.rounds[previous].values()) + list(tracker.rounds[current].values())))
    if syndrome_times:
        windows.append((final_time, list(tracker.rounds[syndrome_times[-1]].values()) + list(tracker.rounds[final_time].values())))

    detector_lines: list[str] = []
    detector_basis: dict[int, int] = {}
    for time, cols in windows:
        bit_rows = _sample_rows_to_bit_rows(samples, cols)
        candidates = _nullspace(bit_rows, len(cols))
        for candidate in sorted(candidates, key=lambda row: (row.bit_count(), row)):
            indices = [cols[index] for index in range(len(cols)) if (candidate >> index) & 1]
            row = _row_from_indices(indices)
            trial_basis = dict(detector_basis)
            if not _add_to_basis(row, trial_basis):
                continue
            if any(_is_in_span(protected_row, trial_basis) for protected_row in protected_rows):
                continue

            line = _detector_line_from_indices(
                g,
                metadata,
                indices,
                total_measurements=total,
                time=time,
            )
            trial_lines = detector_lines + [line]
            trial_text = raw_text + "\n".join(trial_lines) + "\n"
            try:
                stim.Circuit(trial_text).detector_error_model(decompose_errors=True)
            except ValueError:
                continue
            detector_lines.append(line)
            detector_basis = trial_basis

    return detector_lines


def strict_stabilizer_flow_relations(distance: int, *, previous_second_half: bool) -> list[StrictFlowRelation]:
    """Return the d=3/5-derived local flow rules for the strict bridge checks.

    The rules deliberately only mention the stabilizers allowed by
    ``BridgeWideGenerator._strict_stabilizer_groups``:

    * internal four-body ``X``/``Z`` plaquettes,
    * lower two-body ``Z`` boundary checks,
    * upper-right two-body ``Z`` boundary checks,
    * lower-right two-body ``X`` boundary checks,
    * left/top ``X`` boundary checks and the second-half upper-left single X.

    The pattern seen at d=3 and d=5 is a rigid shift.  From the first half to
    the second half, Z plaquettes move one row downward while X plaquettes move
    one column rightward.  From the second half back to the first half the same
    shifts apply on the opposite checkerboard.  The only non-pair relations are
    the lower Z boundary and lower-right X boundary, where a disappearing
    boundary check is absorbed into the adjacent plaquette.
    """
    if distance < 3 or distance % 2 != 1:
        raise ValueError("distance must be an odd integer >= 3")

    d = distance
    relations: list[StrictFlowRelation] = []

    if not previous_second_half:
        # First half -> second half.
        for x in range(d - 1):
            for y in range(2 * d - 2):
                if (x + y) % 2 == 0:
                    relations.append(StrictFlowRelation((f"ZP({x},{y})",), (f"ZP({x},{y + 1})",)))
        for y in range(0, d - 1, 2):
            relations.append(StrictFlowRelation((f"ZRU({y})",), (f"ZRU({y + 1})",)))
        for x in range(0, d - 1, 2):
            relations.append(
                StrictFlowRelation((f"ZP({x},{2 * d - 2})", f"ZD({x + 1})"), (f"ZD({x})",))
            )

        for y in range(0, 2 * d - 1, 2):
            relations.append(StrictFlowRelation((f"XL({y})",), (f"XP(0,{y})",)))
        for x in range(d - 2):
            for y in range(2 * d - 1):
                if (x + y) % 2 == 1:
                    relations.append(StrictFlowRelation((f"XP({x},{y})",), (f"XP({x + 1},{y})",)))
        for y in range(d + 1, 2 * d - 2, 2):
            relations.append(
                StrictFlowRelation((f"XP({d - 2},{y})", f"XRD({y + 1})"), (f"XRD({y})",))
            )
    else:
        # Second half -> first half.
        for x in range(d - 1):
            for y in range(2 * d - 2):
                if (x + y) % 2 == 1:
                    relations.append(StrictFlowRelation((f"ZP({x},{y})",), (f"ZP({x},{y + 1})",)))
        for y in range(1, d - 2, 2):
            relations.append(StrictFlowRelation((f"ZRU({y})",), (f"ZRU({y + 1})",)))
        for x in range(1, d - 2, 2):
            relations.append(
                StrictFlowRelation((f"ZP({x},{2 * d - 2})", f"ZD({x + 1})"), (f"ZD({x})",))
            )

        for x in range(d - 2):
            for y in range(2 * d - 1):
                if (x + y) % 2 == 0:
                    relations.append(StrictFlowRelation((f"XP({x},{y})",), (f"XP({x + 1},{y})",)))
        for y in range(d, 2 * d - 2, 2):
            relations.append(
                StrictFlowRelation((f"XP({d - 2},{y})", f"XRD({y + 1})"), (f"XRD({y})",))
            )

    return relations


def strict_same_half_gauge_names(distance: int, *, current_second_half: bool, first_round: bool) -> list[str]:
    """Return strict same-half gauge checks inferred from d=3 and d=5.

    These are not extra stabilizers.  They are deterministic local products of
    the existing strict stabilizer variables after the first bridge half has
    projected the gauge.  The first syndrome round has no such same-half gauge.
    """
    if first_round:
        return []
    if current_second_half:
        return [f"XL({y})" for y in range(1, 2 * distance - 1, 2)]
    return [f"XL({y})" for y in range(0, 2 * distance - 1, 2)] + [
        f"XP(0,{y})" for y in range(1, 2 * distance - 1, 2)
    ]


def _strict_stabilizer_name(g: BridgeWideGenerator, color: str, qubits: Iterable[int]) -> str:
    coords = [((g.coords[q].x2 - 1) // 2, (g.coords[q].y2 - 1) // 2) for q in qubits]
    xs = [x for x, _ in coords]
    ys = [y for _, y in coords]
    qs = list(qubits)
    if len(qs) == 4:
        return f"{color}P({min(xs)},{min(ys)})"
    if len(qs) == 1:
        return f"{color}UL"
    if len(set(ys)) == 1:
        y = ys[0]
        if y == 0:
            return f"{color}U({min(xs)})"
        if y == 2 * g.d - 1:
            return f"{color}D({min(xs)})"
    if len(set(xs)) == 1:
        x = xs[0]
        if x == 0:
            return f"{color}L({min(ys)})"
        if x == g.d - 1 and min(ys) < g.d:
            return f"{color}RU({min(ys)})"
        if x == g.d - 1:
            return f"{color}RD({min(ys)})"
    return f"{color}?{coords}"


def _strict_stabilizer_signature(g: BridgeWideGenerator, color: str, qubits: Iterable[int]) -> int:
    data = g.data_qubits()
    q_to_index = {q: i for i, q in enumerate(data)}
    n = len(data)
    row = 0
    for q in qubits:
        i = q_to_index[q]
        row ^= 1 << (i if color == "X" else n + i)
    return row


def _strict_stabilizer_definitions(
    g: BridgeWideGenerator, *, second_half: bool
) -> list[tuple[str, str, tuple[int, ...], tuple[float, float], int]]:
    blue, red = g._strict_stabilizer_groups(second_half=second_half)
    out: list[tuple[str, str, tuple[int, ...], tuple[float, float], int]] = []
    for color, groups in (("Z", blue), ("X", red)):
        for group in groups:
            qubits = tuple(group)
            pts = [(g.coords[q].x2 / 2, g.coords[q].y2 / 2) for q in qubits]
            center = (sum(x for x, _ in pts) / len(pts), sum(y for _, y in pts) / len(pts))
            out.append(
                (
                    _strict_stabilizer_name(g, color, qubits),
                    color,
                    qubits,
                    center,
                    _strict_stabilizer_signature(g, color, qubits),
                )
            )
    return out


def _minimal_strict_measurement_coeffs(
    g: BridgeWideGenerator,
    *,
    second_half: bool,
    target_signature: int,
    center: tuple[float, float],
) -> tuple[tuple[int, ...], list[CoordKey]]:
    sigs = g.round_measurement_signatures(second_half=second_half)
    keys = list(sigs)
    shift = 2 * len(g.data_qubits())
    full_rows = [data_sig | (aux_sig << shift) for data_sig, aux_sig in sigs.values()]
    cx, cy = center
    candidates: list[tuple[float, int]] = []
    for index, key in enumerate(keys):
        if full_rows[index] == 0:
            continue
        x = key[0] / 2
        y = key[1] / 2
        candidates.append((max(abs(x - cx), abs(y - cy)), index))

    for radius in (0.5, 1, 1.5, 2, 3, 99):
        indices = [index for dist, index in candidates if dist <= radius]
        for weight in range(1, min(4, len(indices)) + 1):
            for combo in itertools.combinations(indices, weight):
                row = 0
                for index in combo:
                    row ^= full_rows[index]
                if row == target_signature:
                    return tuple(combo), keys
    raise ValueError("failed to express strict stabilizer as a local deterministic measurement product")


def _strict_measured_checks(
    g: BridgeWideGenerator,
    tracker: MeasurementTracker,
    *,
    time: int,
    start_second_half: bool = True,
) -> dict[str, StrictMeasuredCheck]:
    second_half = is_second_half_time(time, start_second_half=start_second_half)
    out: dict[str, StrictMeasuredCheck] = {}
    for name, color, qubits, center, signature in _strict_stabilizer_definitions(g, second_half=second_half):
        coeffs, keys = _minimal_strict_measurement_coeffs(
            g,
            second_half=second_half,
            target_signature=signature,
            center=center,
        )
        record_row = 0
        for index in coeffs:
            record_row ^= 1 << tracker.rounds[time][keys[index]]
        out[name] = StrictMeasuredCheck(
            name=name,
            color=color,
            qubits=qubits,
            center=center,
            record_row=record_row,
        )
    return out


def _final_measured_checks(
    g: BridgeWideGenerator,
    tracker: MeasurementTracker,
    *,
    final_time: int,
    basis: str,
) -> list[StrictMeasuredCheck]:
    out: list[StrictMeasuredCheck] = []
    for key in g.data_keys():
        out.append(
            StrictMeasuredCheck(
                name=f"{basis}F({key[0] // 2},{key[1] // 2})",
                color=basis,
                qubits=(),
                center=(key[0] / 2, key[1] / 2),
                record_row=1 << tracker.rounds[final_time][key],
                kind="final",
            )
        )
    return out


def _detector_line_from_checks(
    checks: Iterable[StrictMeasuredCheck],
    *,
    total_measurements: int,
    time: int,
) -> tuple[str, int]:
    selected = list(checks)
    row = 0
    for check in selected:
        row ^= check.record_row
    terms = [f"rec[{index - total_measurements}]" for index in _indices_from_row(row)]
    x = sum(check.center[0] for check in selected) / len(selected)
    y = sum(check.center[1] for check in selected) / len(selected)
    return f"DETECTOR({stim_num_text(x)}, {stim_num_text(y)}, {time}) {' '.join(terms)}", row


def _append_independent_detector_from_checks(
    lines: list[str],
    basis: dict[int, int],
    checks: Iterable[StrictMeasuredCheck],
    *,
    total_measurements: int,
    time: int,
    protected_row: int = 0,
) -> bool:
    line, row = _detector_line_from_checks(checks, total_measurements=total_measurements, time=time)
    trial_basis = dict(basis)
    if not _add_to_basis(row, trial_basis):
        return False
    if protected_row and _is_in_span(protected_row, trial_basis):
        return False
    lines.append(line)
    basis.clear()
    basis.update(trial_basis)
    return True


def _sampled_nullspace_for_checks(samples: np.ndarray, checks: list[StrictMeasuredCheck]) -> list[int]:
    matrix = np.zeros((samples.shape[0], len(checks)), dtype=np.uint8)
    for col, check in enumerate(checks):
        indices = _indices_from_row(check.record_row)
        if indices:
            matrix[:, col] = np.bitwise_xor.reduce(samples[:, indices], axis=1)
    return _nullspace(_sample_rows_to_bit_rows(matrix, list(range(len(checks)))), len(checks))


def _check_with_kind(check: StrictMeasuredCheck, kind: str) -> StrictMeasuredCheck:
    return StrictMeasuredCheck(
        name=check.name,
        color=check.color,
        qubits=check.qubits,
        center=check.center,
        record_row=check.record_row,
        kind=kind,
    )


def _final_virtual_stabilizer_checks(
    g: BridgeWideGenerator,
    tracker: MeasurementTracker,
    *,
    final_time: int,
    basis: str,
    next_second_half: bool,
) -> dict[str, StrictMeasuredCheck]:
    out: dict[str, StrictMeasuredCheck] = {}
    for name, color, qubits, center, _ in _strict_stabilizer_definitions(g, second_half=next_second_half):
        if color != basis:
            continue
        row = 0
        for q in qubits:
            p = g.coords[q]
            row ^= 1 << tracker.rounds[final_time][(p.x2, p.y2)]
        out[name] = StrictMeasuredCheck(
            name=name,
            color=color,
            qubits=tuple(qubits),
            center=center,
            record_row=row,
            kind="final",
        )
    return out


def _strict_relation_bbox(checks: Iterable[StrictMeasuredCheck]) -> float:
    selected = list(checks)
    if not selected:
        return 0
    xs = [check.center[0] for check in selected]
    ys = [check.center[1] for check in selected]
    return (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)


def _strict_check_signature(g: BridgeWideGenerator, check: StrictMeasuredCheck) -> int:
    return _strict_stabilizer_signature(g, check.color, check.qubits)


def _strict_signature_relation_candidates(
    g: BridgeWideGenerator,
    checks: list[StrictMeasuredCheck],
    *,
    required_kinds: set[str],
    forbidden_kinds: set[str] | None = None,
    max_weight: int,
    radii: tuple[float, ...],
) -> list[int]:
    forbidden_kinds = forbidden_kinds or set()
    centers = sorted(
        {
            (round(check.center[0] * 2) / 2, round(check.center[1] * 2) / 2)
            for check in checks
        }
    )
    relations: dict[int, tuple[int, float, int]] = {}
    for radius in radii:
        for cx, cy in centers:
            indices = [
                i
                for i, check in enumerate(checks)
                if max(abs(check.center[0] - cx), abs(check.center[1] - cy)) <= radius
            ]
            if len(indices) < 1:
                continue
            signatures = {index: _strict_check_signature(g, checks[index]) for index in indices}
            for weight in range(1, max_weight + 1):
                for combo in itertools.combinations(indices, weight):
                    selected = [checks[index] for index in combo]
                    kinds = {check.kind for check in selected}
                    if not required_kinds <= kinds:
                        continue
                    if forbidden_kinds & kinds:
                        continue
                    signature = 0
                    for index in combo:
                        signature ^= signatures[index]
                    if signature != 0:
                        continue
                    relation = 0
                    for index in combo:
                        relation |= 1 << index
                    relations[relation] = (weight, _strict_relation_bbox(selected), relation)
    return [
        relation
        for relation, _ in sorted(
            relations.items(),
            key=lambda item: item[1],
        )
    ]


def _append_valid_strict_detector(
    lines: list[str],
    basis_rows: dict[int, int],
    checks: Iterable[StrictMeasuredCheck],
    *,
    raw_text: str,
    total_measurements: int,
    time: int,
    protected_row: int,
    validate_exact: bool = False,
    validate_noisy_graphlike: bool = False,
) -> bool:
    selected = list(checks)
    if not selected:
        return False
    line, row = _detector_line_from_checks(selected, total_measurements=total_measurements, time=time)
    trial_basis = dict(basis_rows)
    if not _add_to_basis(row, trial_basis):
        return False
    if protected_row and _is_in_span(protected_row, trial_basis):
        return False
    if validate_exact or validate_noisy_graphlike:
        trial_text = raw_text + "\n".join(lines + [line]) + "\n"
        try:
            if validate_exact:
                stim.Circuit(trial_text).detector_error_model(decompose_errors=True)
            if validate_noisy_graphlike:
                stim.Circuit(add_gate_noise(trial_text, p1=0.001, p2=0.001)).detector_error_model(
                    decompose_errors=True
                )
        except ValueError:
            return False
    lines.append(line)
    basis_rows.clear()
    basis_rows.update(trial_basis)
    return True


def _raw_measurement_check(
    tracker: MeasurementTracker,
    *,
    time: int,
    key: CoordKey,
    color: str,
    name: str,
    kind: str,
) -> StrictMeasuredCheck:
    return StrictMeasuredCheck(
        name=name,
        color=color,
        qubits=(),
        center=(key[0] / 2, key[1] / 2),
        record_row=1 << tracker.rounds[time][key],
        kind=kind,
    )


def _append_z_right_bottom_flow_detector(
    g: BridgeWideGenerator,
    tracker: MeasurementTracker,
    detector_lines: list[str],
    detector_basis: dict[int, int],
    *,
    raw_text: str,
    total_measurements: int,
    previous_time: int,
    current_time: int,
    protected_row: int,
    start_second_half: bool,
) -> None:
    """Append the local right-bottom Z observable flow detector.

    The second-half right-bottom auxiliary ``(d, 2d)`` flows to the next
    first-half lower-boundary auxiliary ``(d-1, 2d)``.  This is the local
    detector that catches the bridge hook at the lower-right corner while
    preserving the user's continuously flowing Z observable.
    """
    if not is_second_half_time(previous_time, start_second_half=start_second_half):
        return
    if is_second_half_time(current_time, start_second_half=start_second_half):
        return
    previous_key = ikey(g.d, 2 * g.d)
    current_key = ikey(g.d - 1, 2 * g.d)
    if previous_key not in tracker.rounds.get(previous_time, {}):
        return
    if current_key not in tracker.rounds.get(current_time, {}):
        return

    previous = _raw_measurement_check(
        tracker,
        time=previous_time,
        key=previous_key,
        color="Z",
        name=f"ZRB({previous_time})",
        kind="z-right-bottom-flow",
    )
    current = _raw_measurement_check(
        tracker,
        time=current_time,
        key=current_key,
        color="Z",
        name=f"ZBD({current_time})",
        kind="z-right-bottom-flow",
    )
    _append_valid_strict_detector(
        detector_lines,
        detector_basis,
        [previous, current],
        raw_text=raw_text,
        total_measurements=total_measurements,
        time=current_time,
        protected_row=protected_row,
        validate_exact=True,
    )


def _append_z_right_bottom_final_detector(
    g: BridgeWideGenerator,
    tracker: MeasurementTracker,
    detector_lines: list[str],
    detector_basis: dict[int, int],
    final_virtual: dict[str, StrictMeasuredCheck],
    *,
    raw_text: str,
    total_measurements: int,
    last_syndrome_time: int,
    final_time: int,
    protected_row: int,
    previous_second_half: bool,
    next_second_half: bool,
) -> None:
    """Close the second-half right-bottom Z flow against final data readout."""
    if not previous_second_half or next_second_half:
        return
    previous_key = ikey(g.d, 2 * g.d)
    current_name = f"ZD({g.d - 2})"
    if previous_key not in tracker.rounds.get(last_syndrome_time, {}):
        return
    if current_name not in final_virtual:
        return

    previous = _raw_measurement_check(
        tracker,
        time=last_syndrome_time,
        key=previous_key,
        color="Z",
        name=f"ZRB({last_syndrome_time})",
        kind="z-right-bottom-final",
    )
    _append_valid_strict_detector(
        detector_lines,
        detector_basis,
        [previous, final_virtual[current_name]],
        raw_text=raw_text,
        total_measurements=total_measurements,
        time=final_time,
        protected_row=protected_row,
        validate_exact=True,
    )



def _explicit_relation(
    g: BridgeWideGenerator,
    *,
    family: str,
    previous_time: int,
    current_time: int,
    previous_coords: Iterable[tuple[int, int]],
    current_coords: Iterable[tuple[int, int]],
) -> ExplicitDetectorRelation:
    """Create one relation and reject coordinates absent from the supplied layout."""
    previous_keys = tuple(ikey(x, y) for x, y in previous_coords)
    current_keys = tuple(ikey(x, y) for x, y in current_coords)
    for time, keys in ((previous_time, previous_keys), (current_time, current_keys)):
        for key in keys:
            x = key[0] // 2
            y = key[1] // 2
            if g.int_q(x, y) is None:
                raise ValueError(
                    f"{family}: auxiliary ({x}, {y}) at time {time} is not present; "
                    "the relation was not modified or completed automatically"
                )
    return ExplicitDetectorRelation(
        family=family,
        previous_time=previous_time,
        current_time=current_time,
        previous_keys=previous_keys,
        current_keys=current_keys,
    )


def specified_repeat_detector_relations(
    g: BridgeWideGenerator,
    *,
    previous_time: int,
    current_time: int,
) -> list[ExplicitDetectorRelation]:
    """Return repeat relations exactly in the user's corrected framework.

    Odd times are the first half-cycle and even times are the second half-cycle.
    The color of an internal *measurement record* is determined at the end of
    the half-cycle.  In particular, the first-half stabilizer initially spanned
    around auxiliary (1,1) can contract and be read at (2,1); this routine does
    not identify an initial stabilizer center with its final readout coordinate.

    For formulas whose integer n was not assigned an explicit lower bound, the
    ranges below include every coordinate-valid term.  No endpoint is removed
    by symmetry or by an inferred stabilizer count.
    """
    if current_time != previous_time + 1:
        raise ValueError("repeat detector times must be consecutive")
    if (previous_time & 1) == (current_time & 1):
        raise ValueError("consecutive half-cycles must have opposite parity")

    d = g.d
    first_to_second = previous_time % 2 == 1
    relations: list[ExplicitDetectorRelation] = []

    def add(
        family: str,
        previous_coords: Iterable[tuple[int, int]],
        current_coords: Iterable[tuple[int, int]],
    ) -> None:
        relations.append(
            _explicit_relation(
                g,
                family=family,
                previous_time=previous_time,
                current_time=current_time,
                previous_coords=previous_coords,
                current_coords=current_coords,
            )
        )

    # Internal readouts.  First-half (2,1) is the contracted Z readout and
    # first-half (1,1) is an X readout.  Every source auxiliary is internal;
    # the shifted target is allowed to lie on x=d or y=2d exactly as written.
    if first_to_second:
        for x in range(1, d):
            for y in range(1, 2 * d):
                if (x + y) % 2 == 1:
                    add("internal-Z first->second", [(x, y)], [(x, y + 1)])
                else:
                    add("internal-X first->second", [(x, y)], [(x + 1, y)])
    else:
        for x in range(1, d):
            for y in range(1, 2 * d):
                if (x + y) % 2 == 0:
                    add("internal-Z second->first", [(x, y)], [(x, y + 1)])
                else:
                    add("internal-X second->first", [(x, y)], [(x + 1, y)])

    if first_to_second:
        # Upper X boundary (corrected coordinate orientation):
        # (2n,0,A)+(2n+1,0,A) xor (2n,0,B)+(2n+1,0,B).
        # n=0 is excluded because auxiliary (0,0) does not participate.
        for n in range(1, (d - 1) // 2 + 1):
            add(
                "upper-X first->second",
                [(2 * n, 0), (2 * n + 1, 0)],
                [(2 * n, 0), (2 * n + 1, 0)],
            )

        # Right Z boundary: all written vertical coordinates are in (0,d].
        for n in range(1, (d - 1) // 2 + 1):
            add("right-Z first->second", [(d, 2 * n)], [(d, 2 * n + 1)])

        # Right X boundary: all written vertical coordinates are in (d,2d].
        for n in range((d + 3) // 2, d + 1):
            add(
                "right-X first->second",
                [(d, 2 * n - 1), (d, 2 * n)],
                [(d, 2 * n - 1)],
            )

        # Left X boundary:
        # (0,2n,A)+(0,2n+1,A) xor (1,2n,B).
        for n in range(1, d):
            add(
                "left-X first->second",
                [(0, 2 * n), (0, 2 * n + 1)],
                [(1, 2 * n)],
            )

        # Lower Z boundary (latest correction):
        # (2n-1,2d,A)+(2n,2d,A) xor (2n-1,2d,B).
        for n in range(1, (d - 1) // 2 + 1):
            add(
                "lower-Z first->second",
                [(2 * n - 1, 2 * d), (2 * n, 2 * d)],
                [(2 * n - 1, 2 * d)],
            )

        # Explicit upper-left corner repeat detector.
        add("upper-left corner first->second", [(0, 1)], [(0, 1)])
    else:
        # Upper X boundary (corrected coordinate orientation):
        # (2n-1,0,B)+(2n,0,B) xor the same pair at A_next.
        for n in range(1, (d - 1) // 2 + 1):
            add(
                "upper-X second->first",
                [(2 * n - 1, 0), (2 * n, 0)],
                [(2 * n - 1, 0), (2 * n, 0)],
            )

        # Right Z boundary: all written vertical coordinates are in (0,d].
        for n in range(1, (d - 1) // 2 + 1):
            add("right-Z second->first", [(d, 2 * n - 1)], [(d, 2 * n)])

        # Right X boundary: all written vertical coordinates are in (d,2d].
        for n in range((d + 1) // 2, d):
            add(
                "right-X second->first",
                [(d, 2 * n), (d, 2 * n + 1)],
                [(d, 2 * n)],
            )

        # Left X boundary (latest correction):
        # (0,2n-1,B)+(0,2n,B) xor (1,2n-1,A_next).
        # All coordinate-valid n=1,...,d are retained verbatim.
        for n in range(1, d + 1):
            add(
                "left-X second->first",
                [(0, 2 * n - 1), (0, 2 * n)],
                [(1, 2 * n - 1)],
            )

        # Lower Z boundary (latest correction):
        # (2n,2d,B)+(2n+1,2d,B) xor (2n,2d,A_next).
        # The user explicitly excludes n=0 for the lower-Z boundary.
        for n in range(1, (d - 1) // 2 + 1):
            add(
                "lower-Z second->first",
                [(2 * n, 2 * d), (2 * n + 1, 2 * d)],
                [(2 * n, 2 * d)],
            )

    return relations


def _record_row_for_explicit_relation(
    tracker: MeasurementTracker,
    relation: ExplicitDetectorRelation,
) -> int:
    row = 0
    for time, keys in (
        (relation.previous_time, relation.previous_keys),
        (relation.current_time, relation.current_keys),
    ):
        if time not in tracker.rounds:
            raise ValueError(f"{relation.family}: syndrome time {time} was not recorded")
        for key in keys:
            if key not in tracker.rounds[time]:
                raise ValueError(
                    f"{relation.family}: auxiliary coordinate "
                    f"({key[0] / 2:g}, {key[1] / 2:g}) is absent at time {time}"
                )
            row ^= 1 << tracker.rounds[time][key]
    return row


def _detector_line_from_record_row_at_center(
    row: int,
    *,
    total_measurements: int,
    center: tuple[float, float],
    time: int,
) -> str:
    terms = [f"rec[{index - total_measurements}]" for index in _indices_from_row(row)]
    if not terms:
        raise ValueError("a specified detector reduced to an empty parity")
    return (
        f"DETECTOR({stim_num_text(center[0])}, {stim_num_text(center[1])}, {time}) "
        + " ".join(terms)
    )


def _append_exact_detector_line(
    detector_lines: list[str],
    *,
    raw_text: str,
    line: str,
    label: str,
    validate_exact: bool,
) -> None:
    """Append exactly one requested detector; never drop it as dependent."""
    if validate_exact:
        trial_text = raw_text + "\n".join(detector_lines + [line]) + "\n"
        try:
            stim.Circuit(trial_text).detector_error_model(decompose_errors=True)
        except ValueError as ex:
            raise ValueError(
                f"specified detector is not deterministic: {label}\n{line}\n"
                f"Stim: {str(ex).splitlines()[0]}"
            ) from ex
    detector_lines.append(line)


def _data_key_from_doubled(x2: int, y2: int) -> CoordKey:
    if x2 % 2 != 1 or y2 % 2 != 1:
        raise ValueError(f"data coordinate ({x2}/2,{y2}/2) is not half-integer")
    return (x2, y2)


def specified_first_half_stabilizers(g: BridgeWideGenerator) -> list[SpecifiedFirstHalfStabilizer]:
    """Return the first-half stabilizers in the corrected final-closure form.

    Each item stores exactly the auxiliary measurement records and final data
    measurements stated by the user.  These same auxiliary products are used
    for the first detector layer, while the final layer XORs them with the
    listed data measurements.  No support is inferred from the old strict
    polygon model.
    """
    d = g.d
    out: list[SpecifiedFirstHalfStabilizer] = []

    def add(
        name: str,
        color: str,
        aux_coords: Iterable[tuple[int, int]],
        data_keys: Iterable[CoordKey],
        center: tuple[float, float],
    ) -> None:
        aux_keys = tuple(ikey(x, y) for x, y in aux_coords)
        final_data_keys = tuple(data_keys)
        for key in aux_keys:
            x, y = key[0] // 2, key[1] // 2
            if g.int_q(x, y) is None:
                raise ValueError(f"{name}: auxiliary ({x},{y}) is absent")
        valid_data = set(g.data_keys())
        for key in final_data_keys:
            if key not in valid_data:
                raise ValueError(
                    f"{name}: data coordinate ({key[0] / 2:g},{key[1] / 2:g}) is absent"
                )
        out.append(
            SpecifiedFirstHalfStabilizer(
                name=name,
                color=color,
                auxiliary_keys=aux_keys,
                final_data_keys=final_data_keys,
                center=center,
            )
        )

    # Internal four-body closures.  At the end of the first half, (a,b) is an
    # X readout when a+b is even and a Z readout when a+b is odd, as fixed by
    # the supplied (1,1) X and (2,1) Z examples.
    for a in range(1, d):
        for b in range(1, 2 * d):
            color = "X" if (a + b) % 2 == 0 else "Z"
            add(
                f"internal-{color}({a},{b})",
                color,
                [(a, b)],
                [
                    _data_key_from_doubled(2 * a - 1, 2 * b - 1),
                    _data_key_from_doubled(2 * a + 1, 2 * b - 1),
                    _data_key_from_doubled(2 * a - 1, 2 * b + 1),
                    _data_key_from_doubled(2 * a + 1, 2 * b + 1),
                ],
                (float(a), float(b)),
            )

    # Upper X boundary:
    # data (1.5+2n,0.5)+(2.5+2n,0.5), auxiliary (2+2n,0)+(3+2n,0).
    for n in range(0, (d - 3) // 2 + 1):
        add(
            f"upper-X({n})",
            "X",
            [(2 + 2 * n, 0), (3 + 2 * n, 0)],
            [
                _data_key_from_doubled(3 + 4 * n, 1),
                _data_key_from_doubled(5 + 4 * n, 1),
            ],
            (2 + 2 * n, 0.5),
        )

    # Left X boundary:
    # data (0.5,1.5+2n)+(0.5,2.5+2n), auxiliary (0,2+2n)+(0,3+2n).
    for n in range(0, d - 1):
        add(
            f"left-X({n})",
            "X",
            [(0, 2 + 2 * n), (0, 3 + 2 * n)],
            [
                _data_key_from_doubled(1, 3 + 4 * n),
                _data_key_from_doubled(1, 5 + 4 * n),
            ],
            (0.5, 2 + 2 * n),
        )

    # Lower Z boundary (corrected comma):
    # data (0.5+2n,2d-0.5)+(1.5+2n,2d-0.5),
    # auxiliary (1+2n,2d)+(2+2n,2d).
    for n in range(0, (d - 3) // 2 + 1):
        add(
            f"lower-Z({n})",
            "Z",
            [(1 + 2 * n, 2 * d), (2 + 2 * n, 2 * d)],
            [
                _data_key_from_doubled(1 + 4 * n, 4 * d - 1),
                _data_key_from_doubled(3 + 4 * n, 4 * d - 1),
            ],
            (1 + 2 * n, 2 * d - 0.5),
        )

    # Right Z boundary:
    # data (d-0.5,1.5+2n)+(d-0.5,2.5+2n), auxiliary (d,2+2n).
    for n in range(0, (d - 3) // 2 + 1):
        add(
            f"right-Z({n})",
            "Z",
            [(d, 2 + 2 * n)],
            [
                _data_key_from_doubled(2 * d - 1, 3 + 4 * n),
                _data_key_from_doubled(2 * d - 1, 5 + 4 * n),
            ],
            (d - 0.5, 2 + 2 * n),
        )

    # Right X boundary:
    # data (d-0.5,0.5+2n)+(d-0.5,1.5+2n),
    # auxiliary (d,1+2n)+(d,2+2n), with both auxiliary rows above d.
    for n in range((d + 1) // 2, d):
        add(
            f"right-X({n})",
            "X",
            [(d, 1 + 2 * n), (d, 2 + 2 * n)],
            [
                _data_key_from_doubled(2 * d - 1, 1 + 4 * n),
                _data_key_from_doubled(2 * d - 1, 3 + 4 * n),
            ],
            (d - 0.5, 1 + 2 * n),
        )

    # Upper-left corner: data (0.5,0.5) xor auxiliary (1,0).
    add(
        "upper-left-corner-X",
        "X",
        [(1, 0)],
        [_data_key_from_doubled(1, 1)],
        (0.5, 0.5),
    )

    expected = 2 * d * d - 1
    if len(out) != expected:
        raise AssertionError(
            f"first-half stabilizer count is {len(out)}, expected {expected}; "
            "no stabilizer was added or removed automatically"
        )
    return out


def _record_row_for_auxiliary_product(
    tracker: MeasurementTracker,
    *,
    time: int,
    auxiliary_keys: Iterable[CoordKey],
    label: str,
) -> int:
    if time not in tracker.rounds:
        raise ValueError(f"{label}: syndrome time {time} was not recorded")
    row = 0
    for key in auxiliary_keys:
        if key not in tracker.rounds[time]:
            raise ValueError(
                f"{label}: auxiliary coordinate ({key[0] / 2:g},{key[1] / 2:g}) "
                f"is absent at time {time}"
            )
        row ^= 1 << tracker.rounds[time][key]
    return row


def infer_specified_detector_lines(
    g: BridgeWideGenerator,
    tracker: MeasurementTracker,
    raw_text: str,
    *,
    basis: str,
    final_time: int,
    validate_exact: bool = True,
) -> list[str]:
    """Build initial, repeat, and final detectors in the corrected framework.

    The observable is deliberately not constructed here.  Initial and final
    layers use the explicitly supplied first-half stabilizers; repeat layers use
    the corrected coordinate equations verbatim.  No detector is discarded for
    linear dependence.
    """
    if basis not in ("X", "Z"):
        raise ValueError("basis must be X or Z")
    total = stim.Circuit(raw_text).num_measurements
    syndrome_times = sorted(time for time in tracker.rounds if time != final_time)
    if not syndrome_times:
        return []
    if syndrome_times[0] != 1:
        raise ValueError("the specified workflow requires the first syndrome time to be 1")
    if syndrome_times[-1] % 2 != 1:
        raise ValueError(
            "the specified final closure requires the last syndrome round to be "
            "the first (odd) half-cycle"
        )

    detector_lines: list[str] = []
    first_half_stabilizers = specified_first_half_stabilizers(g)

    # Initial detector layer: only stabilizers matching the initialized basis.
    first_time = syndrome_times[0]
    initial_count = 0
    for stabilizer in first_half_stabilizers:
        if stabilizer.color != basis:
            continue
        row = _record_row_for_auxiliary_product(
            tracker,
            time=first_time,
            auxiliary_keys=stabilizer.auxiliary_keys,
            label=f"initial {stabilizer.name}",
        )
        line = _detector_line_from_record_row_at_center(
            row,
            total_measurements=total,
            center=stabilizer.center,
            time=first_time,
        )
        _append_exact_detector_line(
            detector_lines,
            raw_text=raw_text,
            line=line,
            label=f"initial {stabilizer.name}",
            validate_exact=validate_exact,
        )
        initial_count += 1

    # Repeat detector layers: retain every relation supplied by the formulas.
    repeat_counts: dict[tuple[int, int], int] = {}
    for previous_time, current_time in zip(syndrome_times, syndrome_times[1:]):
        relations = specified_repeat_detector_relations(
            g,
            previous_time=previous_time,
            current_time=current_time,
        )
        repeat_counts[(previous_time, current_time)] = len(relations)
        seen_rows: dict[int, str] = {}
        for relation_index, relation in enumerate(relations):
            row = _record_row_for_explicit_relation(tracker, relation)
            if row in seen_rows:
                raise ValueError(
                    f"two specified repeat rules produce the same detector parity at "
                    f"time {current_time}: {seen_rows[row]!r} and {relation.family!r}; "
                    "the implementation did not delete either rule"
                )
            seen_rows[row] = f"{relation.family}#{relation_index}"
            current_points = [(key[0] / 2, key[1] / 2) for key in relation.current_keys]
            center = (
                sum(x for x, _ in current_points) / len(current_points),
                sum(y for _, y in current_points) / len(current_points),
            )
            line = _detector_line_from_record_row_at_center(
                row,
                total_measurements=total,
                center=center,
                time=current_time,
            )
            _append_exact_detector_line(
                detector_lines,
                raw_text=raw_text,
                line=line,
                label=f"{relation.family}#{relation_index}",
                validate_exact=validate_exact,
            )

    # Final detector layer: the same first-half auxiliary product XORed with
    # exactly the final data measurements listed in the corrected formulas.
    last_time = syndrome_times[-1]
    final_count = 0
    for stabilizer in first_half_stabilizers:
        if stabilizer.color != basis:
            continue
        row = _record_row_for_auxiliary_product(
            tracker,
            time=last_time,
            auxiliary_keys=stabilizer.auxiliary_keys,
            label=f"final {stabilizer.name}",
        )
        for key in stabilizer.final_data_keys:
            if key not in tracker.rounds[final_time]:
                raise ValueError(
                    f"final {stabilizer.name}: data coordinate "
                    f"({key[0] / 2:g},{key[1] / 2:g}) was not measured"
                )
            row ^= 1 << tracker.rounds[final_time][key]
        line = _detector_line_from_record_row_at_center(
            row,
            total_measurements=total,
            center=stabilizer.center,
            time=final_time,
        )
        _append_exact_detector_line(
            detector_lines,
            raw_text=raw_text,
            line=line,
            label=f"final {stabilizer.name}",
            validate_exact=validate_exact,
        )
        final_count += 1

    if final_count != initial_count:
        raise ValueError(
            f"initial/final detector count mismatch: {initial_count} versus {final_count}; "
            "no detector was added or removed automatically"
        )

    final_text = raw_text + "\n".join(detector_lines) + "\n"
    stim.Circuit(final_text).detector_error_model(decompose_errors=True)
    return detector_lines


def build_specified_memory_without_observable(
    distance: int,
    basis: str,
    *,
    rounds: int | None = None,
    include_polygons: bool = False,
    validate_exact: bool = True,
) -> MemoryBuild:
    """Build the corrected detector workflow, deliberately without an observable."""
    if basis not in ("X", "Z"):
        raise ValueError("basis must be X or Z")
    if rounds is None:
        rounds = distance
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    if rounds % 2 != 1:
        raise ValueError(
            "the supplied workflow starts and ends with the first half-cycle, "
            "so the number of syndrome rounds must be odd"
        )

    g = BridgeWideGenerator(distance, include_polygons=include_polygons)
    tracker = MeasurementTracker(g)
    lines = g.qubit_coord_lines()
    data = data_qubits(g)

    lines.append("TICK")
    lines.append(("RX" if basis == "X" else "R") + " " + " ".join(map(str, data)))

    for time in range(1, rounds + 1):
        g.append_measurement_round(
            lines,
            tracker,
            time=time,
            second_half=(time % 2 == 0),
        )

    lines.append("TICK")
    lines.append(("MX" if basis == "X" else "M") + " " + " ".join(map(str, data)))
    final_time = rounds + 1
    before_final_data = tracker.count
    tracker.record_round(final_time, data)

    raw_text = "\n".join(lines) + "\n"
    lines.extend(
        infer_specified_detector_lines(
            g,
            tracker,
            raw_text,
            basis=basis,
            final_time=final_time,
            validate_exact=validate_exact,
        )
    )

    return MemoryBuild(
        distance=distance,
        basis=basis,
        measurement_rounds=rounds,
        text_without_observable="\n".join(lines) + "\n",
        tracker=tracker,
        final_time=final_time,
        before_final_data_measurements=before_final_data,
        final_logical_indices=[],
        protected_observable_row=None,
    )


def generate_specified_detector_circuit(
    distance: int,
    basis: str,
    *,
    p1: float = 0.0,
    p2: float = 0.0,
    rounds: int | None = None,
    include_polygons: bool = False,
    validate_exact: bool = True,
) -> str:
    """Generate the detector-only circuit; no OBSERVABLE_INCLUDE is inserted."""
    build = build_specified_memory_without_observable(
        distance,
        basis,
        rounds=rounds,
        include_polygons=include_polygons,
        validate_exact=validate_exact,
    )
    text = build.text_without_observable
    stim.Circuit(text).detector_error_model(decompose_errors=True)
    text = add_gate_noise(text, p1=p1, p2=p2)
    stim.Circuit(text).detector_error_model(decompose_errors=True)
    return text


def generate_specified_memory_circuit(*args, **kwargs) -> str:
    """Observable-aware generation is intentionally disabled pending its definition."""
    raise ValueError(
        "the corrected detector circuit is available through "
        "generate_specified_detector_circuit(); OBSERVABLE_INCLUDE is intentionally "
        "not generated until the user supplies the concrete observable rule"
    )

def infer_strict_flow_detector_lines(
    g: BridgeWideGenerator,
    tracker: MeasurementTracker,
    raw_text: str,
    *,
    basis: str,
    final_time: int,
    protected_row: int,
    shots: int = 0,
    start_second_half: bool = True,
) -> list[str]:
    """Build detectors using only the allowed bridge stabilizer shapes."""
    circuit = stim.Circuit(raw_text)
    total = circuit.num_measurements
    if shots <= 0:
        shots = max(12000, 64 * total)

    syndrome_times = sorted(time for time in tracker.rounds if time != final_time)
    if not syndrome_times:
        return []

    checks_by_time = {
        time: _strict_measured_checks(
            g,
            tracker,
            time=time,
            start_second_half=start_second_half,
        )
        for time in syndrome_times
    }
    detector_lines: list[str] = []
    detector_basis: dict[int, int] = {}
    validate_candidates = start_second_half

    first_time = syndrome_times[0]
    for check in checks_by_time[first_time].values():
        if check.color == basis:
            _append_valid_strict_detector(
                detector_lines,
                detector_basis,
                [check],
                raw_text=raw_text,
                total_measurements=total,
                time=first_time,
                protected_row=protected_row,
                validate_exact=validate_candidates,
            )

    for previous_time, current_time in zip(syndrome_times, syndrome_times[1:]):
        current = checks_by_time[current_time]
        for name in strict_same_half_gauge_names(
            g.d,
            current_second_half=is_second_half_time(current_time, start_second_half=start_second_half),
            first_round=False,
        ):
            if name in current:
                _append_valid_strict_detector(
                    detector_lines,
                    detector_basis,
                    [current[name]],
                    raw_text=raw_text,
                    total_measurements=total,
                    time=current_time,
                    protected_row=protected_row,
                    validate_exact=validate_candidates,
                )

        previous = checks_by_time[previous_time]
        for relation in strict_stabilizer_flow_relations(
            g.d,
            previous_second_half=is_second_half_time(previous_time, start_second_half=start_second_half),
        ):
            if not all(name in previous for name in relation.previous):
                continue
            if not all(name in current for name in relation.current):
                continue
            _append_valid_strict_detector(
                detector_lines,
                detector_basis,
                [previous[name] for name in relation.previous] + [current[name] for name in relation.current],
                raw_text=raw_text,
                total_measurements=total,
                time=current_time,
                protected_row=protected_row,
                validate_exact=validate_candidates,
            )

        if basis == "Z":
            _append_z_right_bottom_flow_detector(
                g,
                tracker,
                detector_lines,
                detector_basis,
                raw_text=raw_text,
                total_measurements=total,
                previous_time=previous_time,
                current_time=current_time,
                protected_row=protected_row,
                start_second_half=start_second_half,
            )

    last_time = syndrome_times[-1]
    previous_second_half = is_second_half_time(last_time, start_second_half=start_second_half)
    next_second_half = not previous_second_half
    previous = checks_by_time[last_time]
    final_virtual = _final_virtual_stabilizer_checks(
        g,
        tracker,
        final_time=final_time,
        basis=basis,
        next_second_half=next_second_half,
    )
    for relation in strict_stabilizer_flow_relations(g.d, previous_second_half=previous_second_half):
        if not all(name in previous for name in relation.previous):
            continue
        if not all(name in final_virtual for name in relation.current):
            continue
        selected = [previous[name] for name in relation.previous] + [
            final_virtual[name] for name in relation.current
        ]
        _append_valid_strict_detector(
            detector_lines,
            detector_basis,
            selected,
            raw_text=raw_text,
            total_measurements=total,
            time=final_time,
            protected_row=protected_row,
            validate_exact=validate_candidates,
        )

    if basis == "X":
        for name in strict_same_half_gauge_names(
            g.d,
            current_second_half=next_second_half,
            first_round=False,
        ):
            if name not in final_virtual:
                continue
            _append_valid_strict_detector(
                detector_lines,
                detector_basis,
                [final_virtual[name]],
                raw_text=raw_text,
                total_measurements=total,
                time=final_time,
                protected_row=protected_row,
                validate_exact=True,
            )
    elif basis == "Z":
        _append_z_right_bottom_final_detector(
            g,
            tracker,
            detector_lines,
            detector_basis,
            final_virtual,
            raw_text=raw_text,
            total_measurements=total,
            last_syndrome_time=last_time,
            final_time=final_time,
            protected_row=protected_row,
            previous_second_half=previous_second_half,
            next_second_half=next_second_half,
        )

    text = raw_text + "\n".join(detector_lines) + "\n"
    stim.Circuit(text).detector_error_model(decompose_errors=True)
    return detector_lines


def infer_observable_row_from_raw_text(
    raw_text: str,
    seed_indices: list[int],
    *,
    shots: int = 0,
) -> int:
    circuit = stim.Circuit(raw_text)
    total_measurements = circuit.num_measurements
    if shots <= 0:
        shots = max(4096, 8 * total_measurements)
    samples = circuit.compile_sampler(seed=0xB10C).sample(shots).astype(np.uint8)
    target = np.bitwise_xor.reduce(samples[:, seed_indices], axis=1)
    correction_indices = [i for i in range(total_measurements) if i not in set(seed_indices)]
    correction_bits, _ = _solve_xor_system_with_nullspace(samples[:, correction_indices], target)

    row = _row_from_indices(seed_indices)
    while correction_bits:
        bit = correction_bits & -correction_bits
        row ^= 1 << correction_indices[bit.bit_length() - 1]
        correction_bits ^= bit
    return row


def _indices_from_row(row: int) -> list[int]:
    indices: list[int] = []
    while row:
        bit = row & -row
        indices.append(bit.bit_length() - 1)
        row ^= bit
    return indices


def _detector_line_from_row(
    row: int,
    *,
    total_measurements: int,
    index: int,
    metadata: dict[int, tuple[int, CoordKey]] | None = None,
) -> str:
    indices = _indices_from_row(row)
    terms = [f"rec[{index - total_measurements}]" for index in indices]
    if metadata is not None:
        records = [metadata[i] for i in indices if i in metadata]
        if records:
            time = max(record_time for record_time, _ in records)
            x = sum(key[0] / 2 for _, key in records) / len(records)
            y = sum(key[1] / 2 for _, key in records) / len(records)
        else:
            time = index
            x = y = 0
        return f"DETECTOR({stim_num_text(x)}, {stim_num_text(y)}, {time}) {' '.join(terms)}"
    return f"DETECTOR(0, 0, {index}) {' '.join(terms)}"


def measurement_record_terms_from_total(total_measurements: int, row: int) -> list[str]:
    return [f"rec[{index - total_measurements}]" for index in range(total_measurements) if (row >> index) & 1]


def infer_global_deterministic_basis(raw_text: str, *, shots: int = 0) -> list[int]:
    """Return an exact-validated basis for all deterministic record parities."""
    circuit = stim.Circuit(raw_text)
    total = circuit.num_measurements
    if shots <= 0:
        shots = max(40000, 128 * total)
    samples = circuit.compile_sampler(seed=0xD37EC7).sample(shots).astype(np.uint8)
    candidates = _nullspace(_sample_rows_to_bit_rows(samples, list(range(total))), total)

    basis_rows: list[int] = []
    detector_lines: list[str] = []
    detector_basis: dict[int, int] = {}
    for row in sorted(candidates, key=lambda item: (item.bit_count(), item)):
        trial_basis = dict(detector_basis)
        if not _add_to_basis(row, trial_basis):
            continue
        line = _detector_line_from_row(row, total_measurements=total, index=len(detector_lines))
        try:
            stim.Circuit(raw_text + "\n".join(detector_lines + [line]) + "\n").detector_error_model(
                decompose_errors=True
            )
        except ValueError:
            continue
        basis_rows.append(row)
        detector_lines.append(line)
        detector_basis = trial_basis
    return basis_rows


def _coordinates_in_basis(basis_rows: list[int], target: int) -> int:
    pivot_rows: dict[int, tuple[int, int]] = {}
    for index, row in enumerate(basis_rows):
        trial = row
        coord = 1 << index
        while trial:
            pivot = trial.bit_length() - 1
            if pivot not in pivot_rows:
                pivot_rows[pivot] = (trial, coord)
                break
            old_row, old_coord = pivot_rows[pivot]
            trial ^= old_row
            coord ^= old_coord

    trial = target
    coord = 0
    while trial:
        pivot = trial.bit_length() - 1
        if pivot not in pivot_rows:
            raise ValueError("target row is not in the deterministic basis span")
        old_row, old_coord = pivot_rows[pivot]
        trial ^= old_row
        coord ^= old_coord
    return coord


def _row_from_basis_coordinates(basis_rows: list[int], coord: int) -> int:
    row = 0
    while coord:
        bit = coord & -coord
        row ^= basis_rows[bit.bit_length() - 1]
        coord ^= bit
    return row


def _kernel_rows_for_lambda(basis_rows: list[int], lam: int) -> list[int]:
    pivot = (lam & -lam).bit_length() - 1
    rows: list[int] = []
    for index in range(len(basis_rows)):
        if index == pivot:
            continue
        coord = 1 << index
        if (lam >> index) & 1:
            coord ^= 1 << pivot
        rows.append(_row_from_basis_coordinates(basis_rows, coord))
    return rows


def _basis_error_signatures(raw_text: str, basis_rows: list[int]) -> set[int]:
    total = stim.Circuit(raw_text).num_measurements
    detector_lines = [
        _detector_line_from_row(row, total_measurements=total, index=index)
        for index, row in enumerate(basis_rows)
    ]
    text = raw_text + "\n".join(detector_lines) + "\n"
    dem = stim.Circuit(add_gate_noise(text, p1=0.001, p2=0.001)).detector_error_model(
        decompose_errors=True
    )

    signatures: set[int] = set()
    for line in str(dem).splitlines():
        if not line.startswith("error"):
            continue
        signature = 0
        for detector_text in re.findall(r"D(\d+)", line):
            signature ^= 1 << int(detector_text)
        if signature:
            signatures.add(signature)
    return signatures


def _choose_distance3_lambda(
    *,
    basis_size: int,
    observable_coord: int,
    error_signatures: set[int],
) -> int:
    forbidden = set(error_signatures)
    signatures = list(error_signatures)
    for i, a in enumerate(signatures):
        for b_sig in signatures[i:]:
            value = a ^ b_sig
            if value:
                forbidden.add(value)

    for weight in range(1, min(6, basis_size) + 1):
        for combo in itertools.combinations(range(basis_size), weight):
            lam = 0
            for index in combo:
                lam ^= 1 << index
            if (lam & observable_coord).bit_count() % 2 == 0:
                continue
            if lam in forbidden:
                continue
            return lam
    raise ValueError("failed to find a distance-3 detector complement")


def infer_distance3_detector_lines(
    raw_text: str,
    observable_row: int,
    tracker: MeasurementTracker,
) -> list[str]:
    """Choose d=3 global detectors whose complement has no weight-1/2 error."""
    basis_rows = infer_global_deterministic_basis(raw_text)
    observable_coord = _coordinates_in_basis(basis_rows, observable_row)
    error_signatures = _basis_error_signatures(raw_text, basis_rows)
    lam = _choose_distance3_lambda(
        basis_size=len(basis_rows),
        observable_coord=observable_coord,
        error_signatures=error_signatures,
    )
    detector_rows = _kernel_rows_for_lambda(basis_rows, lam)
    total = stim.Circuit(raw_text).num_measurements
    metadata = _measurement_index_metadata(BridgeWideGenerator(3, include_polygons=False), tracker)
    return [
        _detector_line_from_row(row, total_measurements=total, index=index, metadata=metadata)
        for index, row in enumerate(detector_rows)
    ]


def fixed_logical_data_keys(distance: int, basis: str, *, half: int) -> list[CoordKey]:
    """Return the requested fixed-position logical data string.

    For Z, ``half`` labels the representative entering that bridge half-cycle:
    half 1 uses y=d-0.5..2d-0.5 and half 2 uses y=d+0.5..2d-0.5.
    """
    if basis == "X":
        return [hkey(x, distance) for x in range(distance)]
    if basis == "Z":
        start_y = distance - 1 if half == 1 else distance
        return [hkey(distance - 1, y) for y in range(start_y, 2 * distance)]
    raise ValueError("basis must be X or Z")


def final_logical_half(
    distance: int,
    basis: str,
    *,
    last_syndrome_time: int,
    start_second_half: bool = True,
) -> int:
    del distance
    if basis == "X":
        return 1
    if basis == "Z":
        # A first-half round maps the half-1 Z representative to half 2; a
        # second-half round maps half 2 back to half 1.
        return 1 if is_second_half_time(last_syndrome_time, start_second_half=start_second_half) else 2
    raise ValueError("basis must be X or Z")


def fixed_bridge_observable_indices(
    distance: int,
    basis: str,
    tracker: MeasurementTracker,
    *,
    final_time: int,
    last_syndrome_time: int,
    start_second_half: bool = True,
) -> tuple[list[int], list[int]]:
    half = final_logical_half(
        distance,
        basis,
        last_syndrome_time=last_syndrome_time,
        start_second_half=start_second_half,
    )
    final_indices = [tracker.rounds[final_time][key] for key in fixed_logical_data_keys(distance, basis, half=half)]
    indices = list(final_indices)

    if basis == "X":
        first_half_times = [
            time
            for time in range(2, last_syndrome_time + 1)
            if not is_second_half_time(time, start_second_half=start_second_half)
        ]
        if not first_half_times:
            raise ValueError("X observable requires at least one first-half syndrome round")
        last_first_half_time = first_half_times[-1]
        indices.append(tracker.rounds[last_first_half_time][ikey(distance, distance + 1)])
    elif basis == "Z":
        for time in range(2, last_syndrome_time + 1):
            if is_second_half_time(time, start_second_half=start_second_half):
                indices.append(tracker.rounds[time][ikey(distance, distance)])
                indices.append(tracker.rounds[time][ikey(distance, 2 * distance)])
    else:
        raise ValueError("basis must be X or Z")

    return final_indices, indices


def build_memory_without_observable(
    distance: int,
    basis: str,
    *,
    rounds: int | None = None,
    include_polygons: bool = False,
    detector_method: str = "strict",
    start_second_half: bool = True,
) -> MemoryBuild:
    if basis not in ("X", "Z"):
        raise ValueError("basis must be X or Z")
    if detector_method not in ("strict", "legacy"):
        raise ValueError("detector_method must be strict or legacy")
    if rounds is None:
        rounds = distance
    if rounds < 1:
        raise ValueError("rounds must be >= 1")

    g = BridgeWideGenerator(distance, include_polygons=include_polygons)
    tracker = MeasurementTracker(g)
    lines = g.qubit_coord_lines()
    data = data_qubits(g)

    lines.append("TICK")
    lines.append(("RX" if basis == "X" else "R") + " " + " ".join(map(str, data)))

    last_syndrome_time = rounds + 1
    for time in range(2, last_syndrome_time + 1):
        second_half = is_second_half_time(time, start_second_half=start_second_half)
        g.append_measurement_round(lines, tracker, time=time, second_half=second_half)

    lines.append("TICK")
    lines.append(("MX" if basis == "X" else "M") + " " + " ".join(map(str, data)))
    final_time = last_syndrome_time + 1
    before_final_data = tracker.count
    tracker.record_round(final_time, data)

    raw_text = "\n".join(lines) + "\n"
    final_logical_indices, observable_indices = fixed_bridge_observable_indices(
        distance,
        basis,
        tracker,
        final_time=final_time,
        last_syndrome_time=last_syndrome_time,
        start_second_half=start_second_half,
    )
    protected_observable_row = _row_from_indices(observable_indices)
    if detector_method == "strict":
        lines.extend(
            infer_strict_flow_detector_lines(
                g,
                tracker,
                raw_text,
                basis=basis,
                final_time=final_time,
                protected_row=protected_observable_row,
                start_second_half=start_second_half,
            )
        )
    elif distance == 3:
        lines.extend(infer_distance3_detector_lines(raw_text, protected_observable_row, tracker))
    else:
        lines.extend(
            infer_local_detector_lines(
                g,
                tracker,
                raw_text,
                protected_rows=[protected_observable_row],
            )
        )

    return MemoryBuild(
        distance=distance,
        basis=basis,
        measurement_rounds=rounds,
        text_without_observable="\n".join(lines) + "\n",
        tracker=tracker,
        final_time=final_time,
        before_final_data_measurements=before_final_data,
        final_logical_indices=final_logical_indices,
        protected_observable_row=protected_observable_row,
    )


def measurement_record_terms_from_row(build: MemoryBuild, row: int) -> list[str]:
    total_measurements = stim.Circuit(build.text_without_observable).num_measurements
    return [f"rec[{index - total_measurements}]" for index in range(total_measurements) if (row >> index) & 1]


def append_observable(build: MemoryBuild, terms: list[str]) -> str:
    return build.text_without_observable + "OBSERVABLE_INCLUDE(0) " + " ".join(terms) + "\n"


def infer_observable_terms(build: MemoryBuild, *, shots: int) -> list[str]:
    circuit = stim.Circuit(build.text_without_observable)
    total_measurements = circuit.num_measurements
    if shots <= 0:
        shots = max(1024, 4 * build.before_final_data_measurements)

    samples = circuit.compile_sampler().sample(shots).astype(np.uint8)
    final_logical = set(build.final_logical_indices)
    target = np.bitwise_xor.reduce(samples[:, build.final_logical_indices], axis=1)
    detector_span = _detector_span_basis(build.text_without_observable)
    seed_bits = 0
    for index in build.final_logical_indices:
        seed_bits ^= 1 << index

    def terms_from_row(row: int) -> list[str]:
        return [f"rec[{i - total_measurements}]" for i in range(total_measurements) if (row >> i) & 1]

    def row_is_valid(row: int) -> bool:
        if _is_in_span(row, detector_span):
            return False
        try:
            stim.Circuit(append_observable(build, terms_from_row(row))).detector_error_model(decompose_errors=True)
        except ValueError:
            return False
        return True

    correction_indices = [i for i in range(total_measurements) if i not in final_logical]
    correction_samples = samples[:, correction_indices]
    correction_bits, nullspace = _solve_xor_system_with_nullspace(correction_samples, target)

    def expand(bits: int) -> int:
        row = seed_bits
        while bits:
            bit = bits & -bits
            row ^= 1 << correction_indices[bit.bit_length() - 1]
            bits ^= bit
        return row

    row = expand(correction_bits)
    if row_is_valid(row):
        return terms_from_row(row)

    for delta in sorted(nullspace, key=int.bit_count):
        row = expand(correction_bits ^ delta)
        if row_is_valid(row):
            return terms_from_row(row)

    raise ValueError("failed to infer a non-detector observable; increase --observable-shots")


def choose_observable_terms(build: MemoryBuild, *, method: str = "gf2", shots: int = 0) -> list[str]:
    if method not in ("default", "gf2"):
        raise ValueError("observable_method must be default or gf2")
    # The bridge-code default representative is not yet fixed by hand; use GF2.
    if build.protected_observable_row is not None:
        detector_span = _detector_span_basis(build.text_without_observable)
        if not _is_in_span(build.protected_observable_row, detector_span):
            terms = measurement_record_terms_from_row(build, build.protected_observable_row)
            stim.Circuit(append_observable(build, terms)).detector_error_model(decompose_errors=True)
            return terms
    return infer_observable_terms(build, shots=shots)


def generate_memory_circuit(
    distance: int,
    basis: str,
    *,
    p1: float = 0.0,
    p2: float = 0.0,
    observable_shots: int = 0,
    observable_method: str = "gf2",
    rounds: int | None = None,
    include_polygons: bool = False,
    detector_method: str = "strict",
    start_second_half: bool = True,
) -> str:
    build = build_memory_without_observable(
        distance,
        basis,
        rounds=rounds,
        include_polygons=include_polygons,
        detector_method=detector_method,
        start_second_half=start_second_half,
    )
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


def shortest_graphlike_error_report(
    circuit: stim.Circuit,
    *,
    max_explanations: int = 3,
) -> ShortestGraphlikeErrorReport:
    """Return the shortest graphlike logical error and representative faults."""
    dem = circuit.detector_error_model(decompose_errors=True)
    try:
        shortest = dem.shortest_graphlike_error()
    except ValueError as ex:
        return ShortestGraphlikeErrorReport(
            weight=None,
            error_lines=[],
            representative_locations=[],
            failure=str(ex).splitlines()[0],
        )

    error_lines = [line for line in str(shortest).splitlines() if line.startswith("error")]
    locations: list[str] = []
    for line in error_lines[:max_explanations]:
        try:
            explained = circuit.explain_detector_error_model_errors(
                dem_filter=stim.DetectorErrorModel(line),
                reduce_to_one_representative_error=True,
            )
        except ValueError as ex:
            locations.append(f"{line}: {type(ex).__name__}: {str(ex).splitlines()[0]}")
            continue
        if not explained or not explained[0].circuit_error_locations:
            locations.append(f"{line}: no single circuit-error representative")
            continue
        location = str(explained[0].circuit_error_locations[0])
        useful_lines = [
            part.strip()
            for part in location.splitlines()
            if "flipped_pauli_product:" in part or "resolving to " in part
        ]
        locations.append(f"{line}: {'; '.join(useful_lines)}")

    return ShortestGraphlikeErrorReport(
        weight=len(shortest),
        error_lines=error_lines,
        representative_locations=locations,
    )


def format_shortest_graphlike_error_report(report: ShortestGraphlikeErrorReport) -> str:
    if report.failure is not None:
        return f"shortest_graphlike_error: unavailable ({report.failure})"
    lines = [f"shortest_graphlike_error_weight={report.weight}"]
    lines.extend(f"shortest_graphlike_error[{index}]={line}" for index, line in enumerate(report.error_lines))
    lines.extend(
        f"representative_fault[{index}]={location}"
        for index, location in enumerate(report.representative_locations)
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate/simulate bridge wide-code memory experiments.")
    parser.add_argument("-d", "--distance", type=int, default=3, help="odd code distance")
    parser.add_argument("--rounds", type=int, default=None, help="syndrome rounds; default is distance")
    parser.add_argument("--basis", choices=["X", "Z"], default="Z", help="memory basis")
    parser.add_argument("-o", "--output", type=Path, default=None, help="write generated .stim circuit")
    parser.add_argument("--noise", type=float, default=None, help="uniform circuit-level noise probability")
    parser.add_argument("--p1", type=float, default=0.0, help="single-qubit/reset/measurement noise probability")
    parser.add_argument("--p2", type=float, default=0.0, help="two-qubit CX noise probability")
    parser.add_argument("--shots", type=int, default=0, help="run pymatching simulation with this many shots")
    parser.add_argument(
        "--diagnose-shortest",
        action="store_true",
        help="print the shortest graphlike logical error and representative circuit faults",
    )
    parser.add_argument("--observable-shots", type=int, default=0, help="ideal shots used for GF2 observable inference")
    parser.add_argument("--polygons", action="store_true", help="include #!pragma POLYGON visual overlays")
    parser.add_argument(
        "--observable-method",
        choices=["default", "gf2"],
        default="gf2",
        help="currently both choices use GF2 for this bridge code",
    )
    parser.add_argument(
        "--detector-method",
        choices=["strict", "legacy"],
        default="strict",
        help="strict uses the d=3/d=5 stabilizer-flow construction; legacy uses the older GF2/local detector basis",
    )
    parser.add_argument(
        "--start-second-half",
        action="store_true",
        default=True,
        help="start the memory schedule with the second half-cycle at time 2 (default)",
    )
    parser.add_argument(
        "--start-first-half",
        dest="start_second_half",
        action="store_false",
        help="use the previous schedule with the first half-cycle at time 2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build = build_memory_without_observable(
        args.distance,
        args.basis,
        rounds=args.rounds,
        include_polygons=args.polygons,
        detector_method=args.detector_method,
        start_second_half=args.start_second_half,
    )
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
    print(f"detectors_by_time={count_detectors_by_time(noiseless_text)}")
    print(f"observable_terms={len(terms)}")
    if args.diagnose_shortest:
        print(format_shortest_graphlike_error_report(shortest_graphlike_error_report(circuit)))
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote={args.output}")
    if args.shots:
        failures, rate = simulate(circuit, shots=args.shots)
        print(f"shots={args.shots} failures={failures} logical_error_rate={stim_num_text(rate)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate wide-qubit H-gate circuits for odd-distance trivalent surface code.

The generated circuit follows the wide-code deformation rules used for the
transversal-H boundary reset:

* qubits occupy every integer/integer and half/half coordinate in
  0 <= x <= d and 0 <= y <= 2d, except (0, 0) and (0, 2d);
* every half/half coordinate is data except (0.5, 2d - 0.5);
* every integer/integer coordinate plus (0.5, 2d - 0.5) is an ancilla;
* CNOT layers use the same diagonal-neighbor scheduling rule as the d=3
  reference circuit;
* detectors can be generated for the repeated syndrome-extraction body.

The POLYGON pragmas are visual aids for Stim's diagram viewer. They do not
affect simulation semantics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BLUE = "0,0,1,0.25"
RED = "1,0,0,0.25"


@dataclass(frozen=True)
class Point:
    x2: int
    y2: int

    @property
    def x(self) -> float:
        return self.x2 / 2

    @property
    def y(self) -> float:
        return self.y2 / 2


def coord_text(value2: int) -> str:
    if value2 % 2 == 0:
        return str(value2 // 2)
    return f"{value2 // 2}.5"


def stim_num_text(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.8g}"


class MeasurementTracker:
    def __init__(self, generator: "WideHGenerator") -> None:
        self.generator = generator
        self.count = 0
        self.rounds: dict[int, dict[tuple[int, int], int]] = {}

    def record_round(self, time: int, qubits: Iterable[int]) -> None:
        mapping: dict[tuple[int, int], int] = {}
        for q in qubits:
            p = self.generator.coords[q]
            mapping[(p.x2, p.y2)] = self.count
            self.count += 1
        self.rounds[time] = mapping

    def rec(self, time: int, key: tuple[int, int]) -> str:
        try:
            index = self.rounds[time][key]
        except KeyError as ex:
            x = key[0] / 2
            y = key[1] / 2
            raise KeyError(f"missing measurement at ({x}, {y}, {time})") from ex
        return f"rec[{index - self.count}]"


class WideHGenerator:
    """Circuit generator for odd distances d = 2n + 1."""

    def __init__(self, distance: int, *, include_polygons: bool = True) -> None:
        if distance < 3 or distance % 2 != 1:
            raise ValueError("distance must be an odd integer >= 3")
        self.d = distance
        self.include_polygons = include_polygons
        self.coords: dict[int, Point] = {}
        self.integer: dict[tuple[int, int], int] = {}
        self.half: dict[tuple[int, int], int] = {}
        self._build_qubits()

    def _build_qubits(self) -> None:
        q = 0
        for column2 in range(2 * self.d + 1):
            if column2 % 2 == 0:
                x = column2 // 2
                y_values = range(1, 2 * self.d) if x == 0 else range(0, 2 * self.d + 1)
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
    def ikey(x: int, y: int) -> tuple[int, int]:
        return (2 * x, 2 * y)

    def corner_half_key(self) -> tuple[int, int]:
        return (1, 4 * self.d - 1)

    def active_qubits(self) -> list[int]:
        """Qubits reset/measured in the wide-qubit deformation round."""
        qs = set(self.integer.values())
        qs.add(self.half[(0, 2 * self.d - 1)])
        return sorted(qs)

    def h1_prepare(self) -> list[int]:
        qs: set[int] = set()

        # Bottom boundary: all integer ancillas on row 0 are X stabilizers in
        # this half-cycle.
        for x in range(1, self.d + 1):
            qs.add(self.integer[(x, 0)])

        # Interior X stabilizers.
        for x in range(1, self.d):
            for y in range(1, 2 * self.d):
                if (x, y) == (1, 2 * self.d - 1):
                    continue
                if (x + y) % 2 == 1:
                    qs.add(self.integer[(x, y)])

        # Right boundary: before the corner bend it follows the interior
        # parity; after the bend every non-top ancilla is X.
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

        for x in range(1, self.d):
            for y in range(1, 2 * self.d):
                if (x + y) % 2 == 0:
                    qs.add(self.integer[(x, y)])

        # The right boundary kink still participates in the X-flow detector
        # from the previous half-cycle, so row d is measured in the X basis.
        for y in range(1, 2 * self.d + 1):
            if y < self.d:
                use = (self.d + y) % 2 == 0
            elif y > self.d:
                use = True
            else:
                use = True
            if use:
                qs.add(self.integer[(self.d, y)])

        qs.add(self.half[(0, 2 * self.d - 1)])
        qs.add(self.integer[(1, 2 * self.d - 1)])
        qs.add(self.integer[(self.d, 2 * self.d)])
        return sorted(qs)

    def h2_prepare(self) -> list[int]:
        qs = set(self.h1_measure())
        qs.add(self.integer[(self.d, self.d)])
        return sorted(qs)

    def h2_measure(self) -> list[int]:
        return self.h1_prepare()

    def forward_cx_layers(self) -> list[list[tuple[int, int]]]:
        """Four CNOT layers for the first wide-qubit measurement round."""
        layer1: list[tuple[int, int]] = []
        layer2: list[tuple[int, int]] = []
        layer3: list[tuple[int, int]] = []
        layer4: list[tuple[int, int]] = []

        # SW diagonal edges are applied in layer 1 and again in layer 4 with
        # opposite CNOT direction. This reproduces the d=3 scheduling pattern.
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

        # NW diagonal edges: integer qubit controls the half-integer qubit.
        for y in range(0, 2 * self.d):
            for x in range(1, self.d + 1):
                iq = self.integer[(x, y)]
                hq = self.half[(x - 1, y)]
                pair = (iq, hq)
                if (x + y) % 2 == 0:
                    layer3.append(pair)
                else:
                    layer2.append(pair)

        # SE diagonal edges: half-integer qubit controls the integer qubit.
        for y in range(1, 2 * self.d + 1):
            for x in range(0, self.d):
                iq = self.int_q(x, y)
                hq = self.half_q(x, y - 1)
                if iq is None or hq is None:
                    continue
                pair = (hq, iq)
                if (x + y) % 2 == 0:
                    layer2.append(pair)
                else:
                    layer3.append(pair)

        layers = [layer1, layer2, layer3, layer4]
        for index, layer in enumerate(layers, start=1):
            self._check_matching(layer, f"forward layer {index}")
        return layers

    def second_cx_layers(self, forward: list[list[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
        """CNOT layers for the post-H boundary-reset measurement round."""
        layers = [list(layer) for layer in reversed(forward)]
        for index, layer in enumerate(layers, start=1):
            self._check_matching(layer, f"second layer {index}")
        return layers

    @staticmethod
    def _check_matching(pairs: Iterable[tuple[int, int]], label: str) -> None:
        used: set[int] = set()
        for a, b in pairs:
            if a in used or b in used:
                raise AssertionError(f"{label} is not a disjoint CNOT layer")
            used.add(a)
            used.add(b)

    def qubit_coord_lines(self) -> list[str]:
        lines: list[str] = []
        for q in sorted(self.coords):
            p = self.coords[q]
            lines.append(f"QUBIT_COORDS({coord_text(p.x2)}, {coord_text(p.y2)}) {q}")
        return lines

    def polygon_lines(self, *, after_h: bool) -> list[str]:
        blue: list[list[int]] = []
        red: list[list[int]] = []

        def add(color: str, qs: list[int]) -> None:
            if color == BLUE:
                blue.append(qs)
            else:
                red.append(qs)

        # Interior plaquettes. The second half swaps X/Z colors after the
        # transversal H.
        for j in range(0, 2 * self.d - 1):
            for i in range(0, self.d - 1):
                even = (i + j) % 2 == 0
                color = BLUE if even else RED
                if after_h:
                    color = RED if color == BLUE else BLUE
                if i == 0 and j == 2 * self.d - 2:
                    qs = [self.half[(0, j)], self.half[(1, j)], self.half[(1, j + 1)]]
                else:
                    qs = [
                        self.half[(i, j)],
                        self.half[(i + 1, j)],
                        self.half[(i + 1, j + 1)],
                        self.half[(i, j + 1)],
                    ]
                add(color, qs)

        # Boundary checks. These rules are the odd-distance continuation of the
        # d=3 wide-qubit boundary pattern.
        if not after_h:
            for j in range(1, 2 * self.d - 2, 2):
                add(BLUE, [self.half[(0, j)], self.half[(0, j + 1)]])
            for j in range(0, self.d - 1, 2):
                add(BLUE, [self.half[(self.d - 1, j)], self.half[(self.d - 1, j + 1)]])
            for j in range(self.d, 2 * self.d - 1, 2):
                add(RED, [self.half[(self.d - 1, j)], self.half[(self.d - 1, j + 1)]])
            for i in range(0, self.d - 1, 2):
                add(RED, [self.half[(i + 1, 0)], self.half[(i, 0)]])
            for i in range(1, self.d - 1, 2):
                add(BLUE, [self.half[(i + 1, 2 * self.d - 1)], self.half[(i, 2 * self.d - 1)]])
        else:
            for j in range(0, 2 * self.d - 2, 2):
                add(BLUE, [self.half[(0, j)], self.half[(0, j + 1)]])
            for j in range(1, self.d - 1, 2):
                add(BLUE, [self.half[(self.d - 1, j)], self.half[(self.d - 1, j + 1)]])
            for j in range(self.d + 1, 2 * self.d - 1, 2):
                add(RED, [self.half[(self.d - 1, j)], self.half[(self.d - 1, j + 1)]])
            for i in range(1, self.d - 1, 2):
                add(RED, [self.half[(i + 1, 0)], self.half[(i, 0)]])
            add(BLUE, [self.half[(0, 2 * self.d - 2)], self.half[(1, 2 * self.d - 1)]])

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

    def append_detector(
        self,
        lines: list[str],
        tracker: MeasurementTracker,
        refs: Iterable[tuple[int, tuple[int, int]]],
        *,
        coord_keys: Iterable[tuple[int, int]],
        time: int,
    ) -> None:
        ref_counts: dict[tuple[int, tuple[int, int]], int] = {}
        for ref in refs:
            ref_counts[ref] = ref_counts.get(ref, 0) ^ 1
        terms = [tracker.rec(ref_time, key) for (ref_time, key), keep in ref_counts.items() if keep]
        if not terms:
            return

        coords = list(coord_keys)
        x = sum(key[0] / 2 for key in coords) / len(coords)
        y = sum(key[1] / 2 for key in coords) / len(coords)
        lines.append(f"DETECTOR({stim_num_text(x)}, {stim_num_text(y)}, {time}) {' '.join(terms)}")

    def append_detectors_between(
        self,
        lines: list[str],
        tracker: MeasurementTracker,
        *,
        previous_time: int,
        current_time: int,
    ) -> None:
        if previous_time % 2 == 1 and current_time % 2 == 0:
            self.append_odd_to_even_detectors(lines, tracker, previous_time, current_time)
        elif previous_time % 2 == 0 and current_time % 2 == 1:
            self.append_even_to_odd_detectors(lines, tracker, previous_time, current_time)
        else:
            raise ValueError("detectors must connect adjacent odd/even half-cycles")

    def append_odd_to_even_detectors(
        self,
        lines: list[str],
        tracker: MeasurementTracker,
        odd_time: int,
        even_time: int,
    ) -> None:
        d = self.d

        # Interior: odd half-cycle X stabilizers move right, Z stabilizers move up.
        for x in range(1, d):
            for y in range(1, 2 * d):
                if (x, y) == (1, 2 * d - 1):
                    continue
                if (x + y) % 2 == 0:
                    current = self.ikey(x + 1, y)
                else:
                    current = self.ikey(x, y + 1)
                if current in tracker.rounds[even_time]:
                    previous = self.ikey(x, y)
                    self.append_detector(
                        lines,
                        tracker,
                        [(odd_time, previous), (even_time, current)],
                        coord_keys=[current],
                        time=even_time,
                    )

        # y=0 boundary, unshifted even/odd adjacent pairs. In the notes this
        # is written as (0, 2n), i.e. row 0 and column 2n.
        for n in range(1, d + 1):
            a = self.ikey(2 * n, 0)
            b = self.ikey(2 * n + 1, 0)
            if a in tracker.rounds[odd_time] and b in tracker.rounds[odd_time]:
                self.append_detector(
                    lines,
                    tracker,
                    [(odd_time, a), (odd_time, b), (even_time, a), (even_time, b)],
                    coord_keys=[a, b],
                    time=even_time,
                )

        # Right Z boundary, y <= d.
        for n in range(1, d + 1):
            y0 = 2 * n
            y1 = 2 * n + 1
            if 0 < y0 <= d and 0 < y1 <= d:
                previous = self.ikey(d, y0)
                current = self.ikey(d, y1)
                self.append_detector(
                    lines,
                    tracker,
                    [(odd_time, previous), (even_time, current)],
                    coord_keys=[current],
                    time=even_time,
                )

        # Right X boundary, y > d.
        for n in range(1, d + 1):
            y0 = 2 * n - 1
            y1 = 2 * n
            if d < y0 <= 2 * d and d < y1 <= 2 * d:
                a = self.ikey(d, y0)
                b = self.ikey(d, y1)
                self.append_detector(
                    lines,
                    tracker,
                    [(odd_time, a), (odd_time, b), (even_time, a)],
                    coord_keys=[a],
                    time=even_time,
                )

        # x=0 Z boundary, shifted adjacent pairs.
        for n in range(1, d):
            a = self.ikey(0, 2 * n - 1)
            b = self.ikey(0, 2 * n)
            self.append_detector(
                lines,
                tracker,
                [(odd_time, a), (odd_time, b), (even_time, a), (even_time, b)],
                coord_keys=[a, b],
                time=even_time,
            )

        # y=2d Z boundary. The first expression in the notes repeats one
        # coordinate; this keeps the left-moving boundary convention.
        for n in range(1, d + 1):
            x0 = 2 * n - 1
            x1 = 2 * n
            if 2 <= x0 <= d and 1 <= x1 <= d:
                a = self.ikey(x0, 2 * d)
                b = self.ikey(x1, 2 * d)
                self.append_detector(
                    lines,
                    tracker,
                    [(odd_time, a), (odd_time, b), (even_time, a)],
                    coord_keys=[a],
                    time=even_time,
                )

        # Left-bottom corner.
        corner = self.corner_half_key()
        self.append_detector(
            lines,
            tracker,
            [
                (odd_time, self.ikey(0, 2 * d - 1)),
                (odd_time, self.ikey(1, 2 * d)),
                (odd_time, self.ikey(2, 2 * d)),
                (even_time, self.ikey(0, 2 * d - 1)),
                (even_time, corner),
            ],
            coord_keys=[self.ikey(0, 2 * d - 1), corner],
            time=even_time,
        )
        self.append_detector(
            lines,
            tracker,
            [
                (odd_time, self.ikey(1, 2 * d - 1)),
                (odd_time, corner),
                (even_time, self.ikey(2, 2 * d - 1)),
            ],
            coord_keys=[self.ikey(2, 2 * d - 1)],
            time=even_time,
        )

    def append_even_to_odd_detectors(
        self,
        lines: list[str],
        tracker: MeasurementTracker,
        even_time: int,
        odd_time: int,
    ) -> None:
        d = self.d

        # Interior: even half-cycle X stabilizers move right, Z stabilizers move up.
        for x in range(1, d):
            for y in range(1, 2 * d):
                if (x, y) == (1, 2 * d - 1):
                    continue
                if (x + y) % 2 == 1:
                    current = self.ikey(x + 1, y)
                else:
                    current = self.ikey(x, y + 1)
                if current in tracker.rounds[odd_time]:
                    previous = self.ikey(x, y)
                    self.append_detector(
                        lines,
                        tracker,
                        [(even_time, previous), (odd_time, current)],
                        coord_keys=[current],
                        time=odd_time,
                    )

        # y=0 boundary.
        for n in range(1, d):
            a = self.ikey(2 * n - 1, 0)
            b = self.ikey(2 * n, 0)
            if a in tracker.rounds[even_time] and b in tracker.rounds[even_time]:
                self.append_detector(
                    lines,
                    tracker,
                    [(even_time, a), (even_time, b), (odd_time, a), (odd_time, b)],
                    coord_keys=[a, b],
                    time=odd_time,
                )

        # Right Z boundary, y <= d.
        for n in range(1, d + 1):
            y0 = 2 * n - 1
            y1 = 2 * n
            if 0 < y0 <= d and 0 < y1 <= d:
                previous = self.ikey(d, y0)
                current = self.ikey(d, y1)
                self.append_detector(
                    lines,
                    tracker,
                    [(even_time, previous), (odd_time, current)],
                    coord_keys=[current],
                    time=odd_time,
                )

        # Right X boundary, y > d.
        for n in range(1, d + 1):
            y0 = 2 * n
            y1 = 2 * n + 1
            if d < y0 <= 2 * d and d < y1 <= 2 * d:
                a = self.ikey(d, y0)
                b = self.ikey(d, y1)
                self.append_detector(
                    lines,
                    tracker,
                    [(even_time, a), (even_time, b), (odd_time, a)],
                    coord_keys=[a],
                    time=odd_time,
                )

        # x=0 Z boundary.
        for n in range(1, d):
            a = self.ikey(0, 2 * n)
            b = self.ikey(0, 2 * n + 1)
            if b in tracker.rounds[even_time]:
                self.append_detector(
                    lines,
                    tracker,
                    [(even_time, a), (even_time, b), (odd_time, a), (odd_time, b)],
                    coord_keys=[a, b],
                    time=odd_time,
                )

        # y=2d Z boundary.
        for n in range(1, d + 1):
            x0 = 2 * n
            x1 = 2 * n + 1
            if 1 <= x0 <= d and 1 <= x1 <= d:
                a = self.ikey(x0, 2 * d)
                b = self.ikey(x1, 2 * d)
                self.append_detector(
                    lines,
                    tracker,
                    [(even_time, a), (even_time, b), (odd_time, a)],
                    coord_keys=[a],
                    time=odd_time,
                )

        # Left-bottom corner.
        corner = self.corner_half_key()
        self.append_detector(
            lines,
            tracker,
            [
                (even_time, self.ikey(1, 2 * d - 1)),
                (even_time, corner),
                (odd_time, self.ikey(1, 2 * d)),
            ],
            coord_keys=[self.ikey(1, 2 * d)],
            time=odd_time,
        )

    def repeat_circuit_lines(self, *, cycles: int) -> list[str]:
        if cycles < 1:
            raise ValueError("cycles must be >= 1")
        lines = self.qubit_coord_lines()
        tracker = MeasurementTracker(self)
        for time in range(1, 2 * cycles + 1):
            self.append_measurement_round(
                lines,
                tracker,
                time=time,
                second_half=time % 2 == 0,
            )
            if time > 1:
                self.append_detectors_between(
                    lines,
                    tracker,
                    previous_time=time - 1,
                    current_time=time,
                )
        return lines

    def circuit_lines(self) -> list[str]:
        lines = self.qubit_coord_lines()
        if self.include_polygons:
            lines.extend(self.polygon_lines(after_h=False))

        active = self.active_qubits()
        forward = self.forward_cx_layers()
        second = self.second_cx_layers(forward)

        self.append_tick(lines)
        self.append_op(lines, "R", active)
        self.append_tick(lines)
        self.append_op(lines, "H", self.h1_prepare())
        self.append_cx_layers(lines, forward)
        self.append_tick(lines)
        self.append_op(lines, "H", self.h1_measure())
        self.append_tick(lines)
        self.append_op(lines, "M", active)

        if self.include_polygons:
            lines.extend(self.polygon_lines(after_h=True))

        self.append_tick(lines)
        self.append_op(lines, "R", active)
        self.append_tick(lines)
        self.append_op(lines, "H", self.h2_prepare())
        self.append_cx_layers(lines, second)
        self.append_tick(lines)
        self.append_op(lines, "H", self.h2_measure())
        self.append_tick(lines)
        self.append_op(lines, "M", active)

        return lines

    def circuit_text(self, *, detectors: bool = False, cycles: int = 2) -> str:
        if detectors:
            return "\n".join(self.repeat_circuit_lines(cycles=cycles)) + "\n"
        return "\n".join(self.circuit_lines()) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Stim wide-qubit H-gate circuit for odd distance d=2n+1.",
    )
    parser.add_argument(
        "-d",
        "--distance",
        type=int,
        default=3,
        help="odd code distance, e.g. 3, 5, 7",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=None,
        help="use d=2*n+1 instead of passing --distance directly",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="write the generated .stim circuit to this path; stdout is used if omitted",
    )
    parser.add_argument(
        "--no-polygons",
        action="store_true",
        help="omit #!pragma POLYGON visualization comments",
    )
    parser.add_argument(
        "--detectors",
        action="store_true",
        help="generate detectors for the repeated syndrome-extraction body",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=2,
        help="number of two-half-cycle repeat-body cycles when --detectors is used",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distance = 2 * args.n + 1 if args.n is not None else args.distance
    circuit = WideHGenerator(distance, include_polygons=not args.no_polygons).circuit_text(
        detectors=args.detectors,
        cycles=args.cycles,
    )
    if args.output is None:
        print(circuit, end="")
    else:
        args.output.write_text(circuit, encoding="utf-8")


if __name__ == "__main__":
    main()

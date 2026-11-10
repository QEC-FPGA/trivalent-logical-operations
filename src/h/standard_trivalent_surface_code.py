#!/usr/bin/env python3
"""Generate standard trivalent surface-code syndrome extraction circuits.

This is the non-wide, d by d version of the trivalent surface-code layout.
Data qubits are placed on half-integer coordinates

    (0.5, 0.5) ... (d - 0.5, d - 0.5)

and ancilla qubits are placed on integer coordinates in 0 <= x,y <= d,
except for the missing corner (0, 0).  The two flowing half-cycles use the
same diagonal CNOT schedule as the wide-code generator, with the vertical
extent clipped from 2d to d.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from wide_h_generator import BLUE, RED, Point, coord_text


class StandardTrivalentSurfaceCodeGenerator:
    """Circuit generator for odd-distance standard trivalent surface codes."""

    def __init__(self, distance: int, *, include_polygons: bool = True) -> None:
        if distance < 3 or distance % 2 != 1:
            raise ValueError("distance must be an odd integer >= 3")
        self.d = distance
        self.include_polygons = include_polygons
        self.coords: dict[int, Point] = {}
        self.integer: dict[tuple[int, int], int] = {}
        self.half: dict[tuple[int, int], int] = {}
        self._build_qubits()
        self._validate_h_schedule()

    def _build_qubits(self) -> None:
        q = 0
        for column2 in range(2 * self.d + 1):
            if column2 % 2 == 0:
                x = column2 // 2
                y_values = range(1, self.d + 1) if x == 0 else range(0, self.d + 1)
                for y in y_values:
                    self.integer[(x, y)] = q
                    self.coords[q] = Point(2 * x, 2 * y)
                    q += 1
            else:
                x = column2 // 2
                for y in range(self.d):
                    self.half[(x, y)] = q
                    self.coords[q] = Point(2 * x + 1, 2 * y + 1)
                    q += 1

    def int_q(self, x: int, y: int) -> int | None:
        return self.integer.get((x, y))

    def half_q(self, x: int, y: int) -> int | None:
        return self.half.get((x, y))

    def qubit_coord_lines(self) -> list[str]:
        lines: list[str] = []
        for q in sorted(self.coords):
            p = self.coords[q]
            lines.append(f"QUBIT_COORDS({coord_text(p.x2)}, {coord_text(p.y2)}) {q}")
        return lines

    def active_qubits(self) -> list[int]:
        return sorted(self.integer.values())

    def h1_prepare(self) -> list[int]:
        qs: set[int] = set()

        # Bottom boundary is X in this half-cycle.
        for x in range(1, self.d + 1):
            qs.add(self.integer[(x, 0)])

        # Interior checks, plus top/right checks following the same parity.
        for x in range(1, self.d + 1):
            for y in range(1, self.d + 1):
                if (x, y) in self.integer and (x + y) % 2 == 1:
                    qs.add(self.integer[(x, y)])

        return sorted(qs)

    def h1_measure(self) -> list[int]:
        qs: set[int] = set()

        for x in range(1, self.d + 1):
            qs.add(self.integer[(x, 0)])

        for x in range(1, self.d + 1):
            for y in range(1, self.d + 1):
                if (x, y) in self.integer and (x + y) % 2 == 0:
                    qs.add(self.integer[(x, y)])

        return sorted(qs)

    def h2_prepare(self) -> list[int]:
        return self.h1_measure()

    def h2_measure(self) -> list[int]:
        return self.h1_prepare()

    def _validate_h_schedule(self) -> None:
        corner = self.integer[(self.d, self.d)]
        if corner not in self.h1_measure():
            raise AssertionError("top-right corner ancilla is missing from h1_measure")
        if corner not in self.h2_prepare():
            raise AssertionError("top-right corner ancilla is missing from h2_prepare")

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

        # SW diagonal edges.
        for y in range(1, self.d + 1):
            for x in range(1, self.d + 1):
                iq = self.integer[(x, y)]
                hq = self.half[(x - 1, y - 1)]
                if (x + y) % 2 == 0:
                    layer1.append((hq, iq))
                    layer4.append((iq, hq))
                else:
                    layer1.append((iq, hq))
                    layer4.append((hq, iq))

        # NW diagonal edges.
        for y in range(0, self.d):
            for x in range(1, self.d + 1):
                iq = self.integer[(x, y)]
                hq = self.half[(x - 1, y)]
                pair = (iq, hq)
                if (x + y) % 2 == 0:
                    layer3.append(pair)
                else:
                    layer2.append(pair)

        # SE diagonal edges.
        for y in range(1, self.d + 1):
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
        layers = [list(layer) for layer in reversed(forward)]
        for index, layer in enumerate(layers, start=1):
            self._check_matching(layer, f"second layer {index}")
        return layers

    def polygon_lines(self, *, after_h: bool) -> list[str]:
        blue: list[list[int]] = []
        red: list[list[int]] = []

        def add(color: str, qs: list[int]) -> None:
            if color == BLUE:
                blue.append(qs)
            else:
                red.append(qs)

        for j in range(0, self.d - 1):
            for i in range(0, self.d - 1):
                color = BLUE if (i + j) % 2 == 0 else RED
                if after_h:
                    color = RED if color == BLUE else BLUE
                add(
                    color,
                    [
                        self.half[(i, j)],
                        self.half[(i + 1, j)],
                        self.half[(i + 1, j + 1)],
                        self.half[(i, j + 1)],
                    ],
                )

        if not after_h:
            for j in range(1, self.d - 1, 2):
                add(BLUE, [self.half[(0, j)], self.half[(0, j + 1)]])
            for j in range(0, self.d - 1, 2):
                add(BLUE, [self.half[(self.d - 1, j)], self.half[(self.d - 1, j + 1)]])
            for i in range(0, self.d - 1, 2):
                add(RED, [self.half[(i + 1, 0)], self.half[(i, 0)]])
            for i in range(1, self.d - 1, 2):
                add(RED, [self.half[(i + 1, self.d - 1)], self.half[(i, self.d - 1)]])
        else:
            for j in range(0, self.d - 1, 2):
                add(BLUE, [self.half[(0, j)], self.half[(0, j + 1)]])
            for j in range(1, self.d - 1, 2):
                add(BLUE, [self.half[(self.d - 1, j)], self.half[(self.d - 1, j + 1)]])
            for i in range(1, self.d - 1, 2):
                add(RED, [self.half[(i + 1, 0)], self.half[(i, 0)]])
            for i in range(0, self.d - 1, 2):
                add(RED, [self.half[(i + 1, self.d - 1)], self.half[(i, self.d - 1)]])

        lines: list[str] = []
        for color, groups in ((BLUE, blue), (RED, red)):
            for qs in groups:
                lines.append(f"#!pragma POLYGON({color}) {' '.join(map(str, qs))}")
        return lines

    def data_logical_mark_lines(self) -> list[str]:
        """Mark data-supported logical representatives from the d=5 example."""
        x_logical = [self.half[(self.d - 1, y)] for y in range(self.d)]
        z_logical = [self.half[(x, self.d - 1)] for x in range(self.d)]
        return [
            f"#!pragma MARKX(0) {' '.join(map(str, x_logical))}",
            f"#!pragma MARKZ(2) {' '.join(map(str, z_logical))}",
        ]

    def flow_logical_mark_lines(self) -> list[str]:
        """Mark the flowing Pauli representatives on boundary ancillas."""
        z0 = [self.integer[(self.d, y)] for y in range(0, self.d, 2)]
        z2 = [self.integer[(0, self.d)]]
        z2.extend(self.integer[(x, self.d)] for x in range(1, self.d + 1, 2))
        return [
            f"#!pragma MARKZ(0) {' '.join(map(str, z0))}",
            f"#!pragma MARKZ(2) {' '.join(map(str, z2))}",
        ]

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
        *,
        second_half: bool,
        mark_after_reset: bool = False,
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
        if mark_after_reset:
            lines.extend(self.flow_logical_mark_lines())
        self.append_tick(lines)
        self.append_op(lines, "H", prepare_h)
        self.append_cx_layers(lines, layers)
        self.append_tick(lines)
        self.append_op(lines, "H", measure_h)
        self.append_tick(lines)
        self.append_op(lines, "M", active)

    def circuit_lines(self, *, half_cycles: int = 4, mark_logicals: bool = True) -> list[str]:
        if half_cycles < 1:
            raise ValueError("half_cycles must be >= 1")

        lines = self.qubit_coord_lines()
        if mark_logicals:
            lines.extend(self.data_logical_mark_lines())
        for t in range(1, half_cycles + 1):
            second_half = t % 2 == 0
            self.append_measurement_round(
                lines,
                second_half=second_half,
                mark_after_reset=mark_logicals and t == 1,
            )
        return lines

    def circuit_text(self, *, half_cycles: int = 4, mark_logicals: bool = True) -> str:
        return "\n".join(self.circuit_lines(half_cycles=half_cycles, mark_logicals=mark_logicals)) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a standard trivalent surface-code Stim circuit.",
    )
    parser.add_argument("-d", "--distance", type=int, default=5, help="odd code distance")
    parser.add_argument("-n", type=int, default=None, help="use d=2*n+1")
    parser.add_argument(
        "--half-cycles",
        type=int,
        default=4,
        help="number of flowing half-cycle measurement rounds to generate",
    )
    parser.add_argument("--no-polygons", action="store_true", help="omit POLYGON visualization comments")
    parser.add_argument("--no-logical-marks", action="store_true", help="omit MARKX/MARKZ logical comments")
    parser.add_argument("-o", "--output", type=Path, default=None, help="write output .stim file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distance = 2 * args.n + 1 if args.n is not None else args.distance
    text = StandardTrivalentSurfaceCodeGenerator(
        distance,
        include_polygons=not args.no_polygons,
    ).circuit_text(
        half_cycles=args.half_cycles,
        mark_logicals=not args.no_logical_marks,
    )
    if args.output is None:
        print(text, end="")
    else:
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

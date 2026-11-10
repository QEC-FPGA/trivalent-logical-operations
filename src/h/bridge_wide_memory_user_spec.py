#!/usr/bin/env python3
"""ACTIVE bridge-wide memory generator.

User-specified bridge wide-code memory generator.

This file is intentionally separate from ``bridge_wide_memory.py``.  It keeps
the original bridge-wide CNOT/H extraction skeleton, but constructs detectors
from the coordinate equations supplied for the wide-memory experiment:

* syndrome time 1, 3, 5, ... is the first half-cycle;
* syndrome time 2, 4, 6, ... is the second half-cycle;
* repeat detectors are written directly in auxiliary-coordinate form;
* initial/final detector layers use the supplied first-half closure formulas;
* observable terms use the supplied fixed X/Z representatives and auxiliary
  contraction coordinates.

The generator validates the requested detector set by default.  Validation is
deliberately strict: a detector is never dropped, completed, or replaced by a
GF(2)-inferred substitute.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import stim

import bridge_wide_memory_framework_v3 as framework
from wide_h_generator import MeasurementTracker, stim_num_text


CoordKey = tuple[int, int]


def ikey(x: int, y: int) -> CoordKey:
    return (2 * x, 2 * y)


def hkey(x: int, y: int) -> CoordKey:
    return (2 * x + 1, 2 * y + 1)


@dataclass(frozen=True)
class SpecDetector:
    label: str
    row: int
    center: tuple[float, float]
    time: int


@dataclass(frozen=True)
class SpecValidationFailure:
    label: str
    line: str
    reason: str


class SpecValidationError(ValueError):
    def __init__(self, failures: list[SpecValidationFailure]) -> None:
        self.failures = failures
        preview = "\n".join(
            f"- {failure.label}: {failure.reason}" for failure in failures[:12]
        )
        if len(failures) > 12:
            preview += f"\n- ... {len(failures) - 12} more"
        super().__init__("specified detector set is not deterministic:\n" + preview)


@dataclass
class SpecMemoryBuild:
    distance: int
    basis: str
    rounds: int
    raw_text: str
    text_without_observable: str
    tracker: MeasurementTracker
    final_time: int
    before_final_data_measurements: int
    final_logical_indices: list[int]
    observable_indices: list[int]
    detector_failures: list[SpecValidationFailure]


class SpecBridgeWideGenerator(framework.BridgeWideGenerator):
    """Bridge-wide skeleton with the user's time-reversal symmetric H layers."""

    def h1_measure(self) -> list[int]:
        qs = set(super().h1_measure())
        qs.add(self.integer[(self.d, 2 * self.d)])
        return sorted(qs)

    def h2_prepare(self) -> list[int]:
        return self.h1_measure()


def _indices_from_row(row: int) -> list[int]:
    out: list[int] = []
    while row:
        bit = row & -row
        out.append(bit.bit_length() - 1)
        row ^= bit
    return out


def detector_line(detector: SpecDetector, *, total_measurements: int) -> str:
    terms = [f"rec[{index - total_measurements}]" for index in _indices_from_row(detector.row)]
    if not terms:
        raise ValueError(f"{detector.label}: detector parity is empty")
    return (
        f"DETECTOR({stim_num_text(detector.center[0])}, "
        f"{stim_num_text(detector.center[1])}, {detector.time}) "
        + " ".join(terms)
    )


def _aux_row(
    tracker: MeasurementTracker,
    *,
    time: int,
    keys: Iterable[CoordKey],
    label: str,
) -> int:
    if time not in tracker.rounds:
        raise ValueError(f"{label}: syndrome time {time} is absent")
    row = 0
    for key in keys:
        if key not in tracker.rounds[time]:
            raise ValueError(
                f"{label}: auxiliary ({key[0] / 2:g},{key[1] / 2:g}) "
                f"is absent at time {time}"
            )
        row ^= 1 << tracker.rounds[time][key]
    return row


def _relation_center(keys: Iterable[CoordKey]) -> tuple[float, float]:
    points = [(key[0] / 2, key[1] / 2) for key in keys]
    return (
        sum(x for x, _ in points) / len(points),
        sum(y for _, y in points) / len(points),
    )


def specified_detector_rows(
    g: framework.BridgeWideGenerator,
    tracker: MeasurementTracker,
    *,
    basis: str,
    final_time: int,
) -> list[SpecDetector]:
    """Return detector rows from the user's formulas, with no inference."""
    if basis not in ("X", "Z"):
        raise ValueError("basis must be X or Z")

    syndrome_times = sorted(time for time in tracker.rounds if time != final_time)
    if not syndrome_times:
        return []
    if syndrome_times[0] != 1:
        raise ValueError("specified workflow requires syndrome time 1")
    if syndrome_times[-1] % 2 != 1:
        raise ValueError("specified final closure requires an odd last syndrome time")

    rows: list[SpecDetector] = []
    stabs = framework.specified_first_half_stabilizers(g)

    first_time = syndrome_times[0]
    for index, relation in enumerate(
        framework.specified_repeat_detector_relations(
            g,
            previous_time=2,
            current_time=3,
        )
    ):
        if f"-{basis} " not in relation.family and not relation.family.endswith(f"-{basis} second->first"):
            continue
        row = _aux_row(
            tracker,
            time=first_time,
            keys=relation.current_keys,
            label=f"initial {relation.family}#{index}",
        )
        rows.append(
            SpecDetector(
                f"initial {relation.family}#{index}",
                row,
                _relation_center(relation.current_keys),
                first_time,
            )
        )

    for previous_time, current_time in zip(syndrome_times, syndrome_times[1:]):
        for index, relation in enumerate(
            framework.specified_repeat_detector_relations(
                g,
                previous_time=previous_time,
                current_time=current_time,
            )
        ):
            row = 0
            row ^= _aux_row(
                tracker,
                time=previous_time,
                keys=relation.previous_keys,
                label=f"repeat {relation.family}#{index} previous",
            )
            row ^= _aux_row(
                tracker,
                time=current_time,
                keys=relation.current_keys,
                label=f"repeat {relation.family}#{index} current",
            )
            rows.append(
                SpecDetector(
                    f"repeat {previous_time}->{current_time} {relation.family}#{index}",
                    row,
                    _relation_center(relation.current_keys),
                    current_time,
                )
            )

    for time in syndrome_times:
        for y in range(1, 2 * g.d + 1):
            key = ikey(0, y)
            if key not in tracker.rounds[time]:
                continue
            row = 1 << tracker.rounds[time][key]
            rows.append(
                SpecDetector(
                    f"column-0 singleton ({0},{y}) t={time}",
                    row,
                    (0.0, float(y)),
                    time,
                )
            )

    last_time = syndrome_times[-1]
    for stab in stabs:
        if stab.color != basis:
            continue
        row = _aux_row(
            tracker,
            time=last_time,
            keys=stab.auxiliary_keys,
            label=f"final {stab.name}",
        )
        for key in stab.final_data_keys:
            if key not in tracker.rounds[final_time]:
                raise ValueError(
                    f"final {stab.name}: data ({key[0] / 2:g},{key[1] / 2:g}) "
                    "was not measured"
                )
            row ^= 1 << tracker.rounds[final_time][key]
        rows.append(SpecDetector(f"final {stab.name}", row, stab.center, final_time))

    initial_count = sum(1 for detector in rows if detector.label.startswith("initial "))
    final_count = sum(1 for detector in rows if detector.label.startswith("final "))
    if initial_count != final_count:
        raise ValueError(f"initial/final detector count mismatch: {initial_count} != {final_count}")
    return rows


def validate_specified_detectors(
    raw_text: str,
    detectors: list[SpecDetector],
) -> list[SpecValidationFailure]:
    """Validate every specified detector without deleting failed rows."""
    total = stim.Circuit(raw_text).num_measurements
    accepted: list[str] = []
    failures: list[SpecValidationFailure] = []
    for detector in detectors:
        line = detector_line(detector, total_measurements=total)
        trial = raw_text + "\n".join(accepted + [line]) + "\n"
        try:
            stim.Circuit(trial).detector_error_model(decompose_errors=True)
        except ValueError as ex:
            failures.append(
                SpecValidationFailure(
                    label=detector.label,
                    line=line,
                    reason=str(ex).splitlines()[0],
                )
            )
            continue
        accepted.append(line)
    return failures


def _specified_logical_data_keys(distance: int, basis: str, *, half: int) -> list[CoordKey]:
    if basis == "X":
        return [hkey(x, distance) for x in range(distance)]
    if basis == "Z":
        start_y = distance - 1 if half == 1 else distance
        return [hkey(distance - 1, y) for y in range(start_y, 2 * distance)]
    raise ValueError("basis must be X or Z")


def specified_observable_indices(
    distance: int,
    basis: str,
    tracker: MeasurementTracker,
    *,
    final_time: int,
    rounds: int,
) -> tuple[list[int], list[int]]:
    """Return final logical data indices and the explicit observable row.

    The parity convention follows the user's text: odd syndrome times use the
    first half-cycle and even syndrome times use the second half-cycle.
    """
    if basis not in ("X", "Z"):
        raise ValueError("basis must be X or Z")

    final_half = 1 if basis == "X" else 2
    final_keys = _specified_logical_data_keys(distance, basis, half=final_half)
    final_indices = [tracker.rounds[final_time][key] for key in final_keys]
    indices = list(final_indices)

    if basis == "X":
        for time in range(1, rounds + 1, 2):
            indices.append(tracker.rounds[time][ikey(distance, distance + 1)])
    else:
        for time in range(2, rounds + 1, 2):
            indices.append(tracker.rounds[time][ikey(distance, distance)])
            indices.append(tracker.rounds[time][ikey(distance, 2 * distance)])

    return final_indices, indices


def measurement_record_terms_from_indices(total_measurements: int, indices: Iterable[int]) -> list[str]:
    return [f"rec[{index - total_measurements}]" for index in sorted(indices)]


def append_observable(build: SpecMemoryBuild) -> str:
    total = stim.Circuit(build.text_without_observable).num_measurements
    terms = measurement_record_terms_from_indices(total, build.observable_indices)
    if not terms:
        raise ValueError("observable is empty")
    return build.text_without_observable + "OBSERVABLE_INCLUDE(0) " + " ".join(terms) + "\n"


def build_memory_without_observable(
    distance: int,
    basis: str,
    *,
    rounds: int | None = None,
    include_polygons: bool = False,
    validate: bool = True,
) -> SpecMemoryBuild:
    if basis not in ("X", "Z"):
        raise ValueError("basis must be X or Z")
    if rounds is None:
        rounds = distance
    if rounds < 1 or rounds % 2 != 1:
        raise ValueError("rounds must be an odd positive integer")

    g = SpecBridgeWideGenerator(distance, include_polygons=include_polygons)
    tracker = MeasurementTracker(g)
    lines = g.qubit_coord_lines()
    data = g.data_qubits()

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
    detector_rows = specified_detector_rows(g, tracker, basis=basis, final_time=final_time)
    detector_failures = validate_specified_detectors(raw_text, detector_rows)
    if validate and detector_failures:
        raise SpecValidationError(detector_failures)

    total = stim.Circuit(raw_text).num_measurements
    detector_lines = [detector_line(detector, total_measurements=total) for detector in detector_rows]
    text_without_observable = raw_text + "\n".join(detector_lines) + "\n"

    final_logical_indices, observable_indices = specified_observable_indices(
        distance,
        basis,
        tracker,
        final_time=final_time,
        rounds=rounds,
    )

    return SpecMemoryBuild(
        distance=distance,
        basis=basis,
        rounds=rounds,
        raw_text=raw_text,
        text_without_observable=text_without_observable,
        tracker=tracker,
        final_time=final_time,
        before_final_data_measurements=before_final_data,
        final_logical_indices=final_logical_indices,
        observable_indices=observable_indices,
        detector_failures=detector_failures,
    )


def generate_memory_circuit(
    distance: int,
    basis: str,
    *,
    p1: float = 0.0,
    p2: float = 0.0,
    rounds: int | None = None,
    include_polygons: bool = False,
    validate: bool = True,
) -> str:
    build = build_memory_without_observable(
        distance,
        basis,
        rounds=rounds,
        include_polygons=include_polygons,
        validate=validate,
    )
    text = append_observable(build)
    if validate:
        stim.Circuit(text).detector_error_model(decompose_errors=True)
    text = framework.add_gate_noise(text, p1=p1, p2=p2)
    if validate:
        stim.Circuit(text).detector_error_model(decompose_errors=True)
    return text


def count_detectors_by_time(text: str) -> dict[int, int]:
    counts: dict[int, int] = {}
    for line in text.splitlines():
        if not line.startswith("DETECTOR"):
            continue
        coords = line.split(")", 1)[0].split("(", 1)[1].split(",")
        time = int(float(coords[2]))
        counts[time] = counts.get(time, 0) + 1
    return dict(sorted(counts.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the user-specified bridge wide-memory circuit.")
    parser.add_argument("-d", "--distance", type=int, default=3, help="odd code distance")
    parser.add_argument("--rounds", type=int, default=None, help="odd syndrome rounds; default is distance")
    parser.add_argument("--basis", choices=["X", "Z"], default="Z", help="memory basis")
    parser.add_argument("-o", "--output", type=Path, default=None, help="write generated .stim circuit")
    parser.add_argument("--noise", type=float, default=None, help="uniform circuit-level noise probability")
    parser.add_argument("--p1", type=float, default=0.0, help="single-qubit/reset/measurement noise probability")
    parser.add_argument("--p2", type=float, default=0.0, help="two-qubit CX noise probability")
    parser.add_argument("--polygons", action="store_true", help="include #!pragma POLYGON visual overlays")
    parser.add_argument(
        "--allow-invalid",
        action="store_true",
        help="write the exact specified detector set even when strict validation fails",
    )
    parser.add_argument(
        "--show-failures",
        action="store_true",
        help="print every detector validation failure",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    p1 = args.noise if args.noise is not None else args.p1
    p2 = args.noise if args.noise is not None else args.p2

    build = build_memory_without_observable(
        args.distance,
        args.basis,
        rounds=args.rounds,
        include_polygons=args.polygons,
        validate=not args.allow_invalid,
    )
    text = append_observable(build)
    text = framework.add_gate_noise(text, p1=p1, p2=p2)

    print(f"distance={args.distance} basis={args.basis} rounds={build.rounds}")
    print(f"detectors_by_time={count_detectors_by_time(build.text_without_observable)}")
    print(f"observable_terms={len(build.observable_indices)}")
    print(f"detector_validation_failures={len(build.detector_failures)}")
    if args.show_failures:
        for failure in build.detector_failures:
            print(f"{failure.label}: {failure.reason}")
            print(f"  {failure.line}")
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote={args.output}")


if __name__ == "__main__":
    main()

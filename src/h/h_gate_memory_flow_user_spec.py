#!/usr/bin/env python3
"""Assemble the current user-specified H-gate memory flow.

The active route is:

1. transposed top standard block: physical left/right X boundaries;
2. transversal H, then ordinary wide-H memory;
3. promote the lower-left wide auxiliary half-qubit into bridge data, then run
   the user-specified bridge-wide memory;
4. measure the upper data and keep the bottom d rows as a transposed lower
   standard block, run one closing syndrome round, then measure the data.

    The long-range observable follows the explicit X/Z flow supplied by the
    user.  Observable basis labels are the post-H output labels: output X comes
    from the initial Z logical, and output Z comes from the initial X logical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import stim

import bridge_wide_memory_framework_v3 as bridge_framework
import bridge_wide_memory_user_spec as bridge_spec
import standard_trivalent_memory as standard_memory
import wide_h_memory as wide_memory
from wide_h_generator import MeasurementTracker, WideHGenerator, stim_num_text
from wide_h_memory import add_gate_noise, count_detectors_by_time


CoordKey = tuple[int, int]


def ikey(x: int, y: int) -> CoordKey:
    return (2 * x, 2 * y)


def hkey(x: int, y: int) -> CoordKey:
    return (2 * x + 1, 2 * y + 1)


@dataclass
class HGateUserSpecBuild:
    text_without_observable: str
    tracker: MeasurementTracker
    input_basis: str
    output_basis: str
    standard_last_time: int
    wide_last_time: int | None
    bridge_last_time: int | None
    shrink_time: int | None
    post_standard_last_time: int | None
    final_time: int | None
    before_final_data_measurements: int
    final_logical_indices: list[int]
    observable_indices: list[int]
    observable_indices_by_basis: dict[str, list[int]]
    observable_basis: str
    initial_standard_data: list[int]
    wide_initialized_data: list[int]
    bridge_promoted_data: list[int]
    upper_measured_data: list[int]
    final_standard_data: list[int]
    standard_rounds: int
    wide_rounds: int
    bridge_rounds: int
    post_standard_rounds: int


class WideOnBridgeAdapter(WideHGenerator):
    """Run the ordinary wide-H schedule on the bridge superset of qubits."""

    def __init__(
        self,
        bridge: bridge_framework.BridgeWideGenerator,
        *,
        include_polygons: bool = False,
    ) -> None:
        self.bridge = bridge
        self.d = bridge.d
        self.include_polygons = include_polygons
        self.coords = bridge.coords
        self.integer: dict[tuple[int, int], int] = {}
        self.half: dict[tuple[int, int], int] = {}
        for x in range(self.d + 1):
            y_values = range(1, 2 * self.d) if x == 0 else range(0, 2 * self.d + 1)
            for y in y_values:
                self.integer[(x, y)] = bridge.integer[(x, y)]
        for x in range(self.d):
            for y in range(2 * self.d):
                self.half[(x, y)] = bridge.half[(x, y)]


class StandardOnBridgeAdapter:
    """Use a d by d standard block inside the bridge coordinate system."""

    def __init__(
        self,
        bridge: bridge_framework.BridgeWideGenerator,
        *,
        placement: str,
        orientation: str = "standard",
    ) -> None:
        if placement not in ("top", "bottom"):
            raise ValueError("placement must be 'top' or 'bottom'")
        if orientation not in ("standard", "transpose"):
            raise ValueError("orientation must be 'standard' or 'transpose'")
        self.bridge = bridge
        self.d = bridge.d
        self.placement = placement
        self.orientation = orientation
        self.include_polygons = False
        self.coords = bridge.coords
        self.integer: dict[tuple[int, int], int] = {}
        self.half: dict[tuple[int, int], int] = {}
        self.qubit_to_local_key: dict[int, CoordKey] = {}
        self._build_view()
        self._validate_h_schedule()

    def _oriented_coord(self, x: int, y: int) -> tuple[int, int]:
        if self.orientation == "transpose":
            return (y, x)
        return (x, y)

    def _global_integer_coord(self, x: int, y: int) -> tuple[int, int]:
        ox, oy = self._oriented_coord(x, y)
        if self.placement == "top":
            return (ox, oy)
        # The lower standard block uses the same physical trivalent coupling
        # orientation as the upper block, translated down by d.  Reflecting
        # the local y coordinate would swap the omitted diagonal and make the
        # union of the bridge and lower-standard stages four-valent.
        return (ox, self.d + oy)

    def _global_half_coord(self, x: int, y: int) -> tuple[int, int]:
        ox, oy = self._oriented_coord(x, y)
        if self.placement == "top":
            return (ox, oy)
        return (ox, self.d + oy)

    def _build_view(self) -> None:
        for x in range(self.d + 1):
            y_values = range(1, self.d + 1) if x == 0 else range(self.d + 1)
            for y in y_values:
                q = self.bridge.integer[self._global_integer_coord(x, y)]
                self.integer[(x, y)] = q
                self.qubit_to_local_key[q] = ikey(x, y)
        for x in range(self.d):
            for y in range(self.d):
                q = self.bridge.half[self._global_half_coord(x, y)]
                self.half[(x, y)] = q
                self.qubit_to_local_key[q] = hkey(x, y)

    def to_global_key(self, key: CoordKey) -> CoordKey:
        if self.orientation == "transpose":
            key = (key[1], key[0])
        if self.placement == "top":
            return key
        return (key[0], key[1] + 2 * self.d)

    def int_q(self, x: int, y: int) -> int | None:
        return self.integer.get((x, y))

    def half_q(self, x: int, y: int) -> int | None:
        return self.half.get((x, y))

    def active_qubits(self) -> list[int]:
        return sorted(self.integer.values())

    def h1_prepare(self) -> list[int]:
        qs: set[int] = set()
        for x in range(1, self.d + 1):
            qs.add(self.integer[(x, 0)])
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
            raise AssertionError("standard-view corner ancilla is missing from h1_measure")
        if corner not in self.h2_prepare():
            raise AssertionError("standard-view corner ancilla is missing from h2_prepare")

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

        for y in range(self.d):
            for x in range(1, self.d + 1):
                iq = self.integer[(x, y)]
                hq = self.half[(x - 1, y)]
                pair = (iq, hq)
                if (x + y) % 2 == 0:
                    layer3.append(pair)
                else:
                    layer2.append(pair)

        for y in range(1, self.d + 1):
            for x in range(self.d):
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
            self._check_matching(layer, f"standard-on-bridge layer {index}")
        return layers

    def second_cx_layers(self, forward: list[list[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
        layers = [list(layer) for layer in reversed(forward)]
        for index, layer in enumerate(layers, start=1):
            self._check_matching(layer, f"standard-on-bridge second layer {index}")
        return layers


def _validate_basis(basis: str) -> str:
    basis = basis.upper()
    if basis not in ("X", "Z"):
        raise ValueError("basis must be X or Z")
    return basis


def _op_for_basis(basis: str, *, measurement: bool) -> str:
    return ("MX" if measurement else "RX") if basis == "X" else ("M" if measurement else "R")


def _append_op(lines: list[str], name: str, qubits: Iterable[int]) -> None:
    qs = list(qubits)
    if qs:
        lines.append(f"{name} {' '.join(map(str, qs))}")


def _record_standard_view_round(
    tracker: MeasurementTracker,
    view: StandardOnBridgeAdapter,
    time: int,
    qubits: Iterable[int],
) -> None:
    mapping: dict[CoordKey, int] = {}
    for q in qubits:
        mapping[view.qubit_to_local_key[q]] = tracker.count
        tracker.count += 1
    tracker.rounds[time] = mapping


def append_standard_view_measurement_round(
    view: StandardOnBridgeAdapter,
    lines: list[str],
    tracker: MeasurementTracker,
    *,
    time: int,
    second_half: bool,
) -> None:
    active = view.active_qubits()
    forward = view.forward_cx_layers()
    layers = view.second_cx_layers(forward) if second_half else forward
    prepare_h = view.h2_prepare() if second_half else view.h1_prepare()
    measure_h = view.h2_measure() if second_half else view.h1_measure()

    lines.append("TICK")
    _append_op(lines, "R", active)
    lines.append("TICK")
    _append_op(lines, "H", prepare_h)
    for layer in layers:
        lines.append("TICK")
        cx = view.cx_line(layer)
        if cx is not None:
            lines.append(cx)
    lines.append("TICK")
    _append_op(lines, "H", measure_h)
    lines.append("TICK")
    _append_op(lines, "M", active)
    _record_standard_view_round(tracker, view, time, active)


def append_measurement_index_detector(
    lines: list[str],
    tracker: MeasurementTracker,
    indices: Iterable[int],
    *,
    coord_keys: Iterable[CoordKey],
    time: int,
) -> None:
    index_counts: dict[int, int] = {}
    for index in indices:
        index_counts[index] = index_counts.get(index, 0) ^ 1
    kept = [index for index, keep in index_counts.items() if keep]
    if not kept:
        return
    coords = list(coord_keys)
    x = sum(key[0] / 2 for key in coords) / len(coords)
    y = sum(key[1] / 2 for key in coords) / len(coords)
    terms = [f"rec[{index - tracker.count}]" for index in kept]
    lines.append(f"DETECTOR({stim_num_text(x)}, {stim_num_text(y)}, {time}) {' '.join(terms)}")


def append_standard_view_detector(
    view: StandardOnBridgeAdapter,
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
    indices = [tracker.rounds[ref_time][key] for (ref_time, key), keep in ref_counts.items() if keep]
    append_measurement_index_detector(
        lines,
        tracker,
        indices,
        coord_keys=[view.to_global_key(key) for key in coord_keys],
        time=time,
    )


def _has_round_key(tracker: MeasurementTracker, time: int, key: CoordKey) -> bool:
    return key in tracker.rounds.get(time, {})


def _append_standard_view_if_present(
    view: StandardOnBridgeAdapter,
    lines: list[str],
    tracker: MeasurementTracker,
    refs: Iterable[tuple[int, CoordKey]],
    *,
    coord_keys: Iterable[CoordKey],
    time: int,
) -> None:
    refs = tuple(refs)
    if all(_has_round_key(tracker, t, key) for t, key in refs):
        append_standard_view_detector(view, lines, tracker, refs, coord_keys=coord_keys, time=time)


def append_standard_view_odd_to_even_detectors(
    view: StandardOnBridgeAdapter,
    lines: list[str],
    tracker: MeasurementTracker,
    odd_time: int,
    even_time: int,
) -> None:
    d = view.d
    for x in range(1, d):
        for y in range(1, d):
            current = ikey(x + 1, y) if (x + y) % 2 == 0 else ikey(x, y + 1)
            _append_standard_view_if_present(
                view,
                lines,
                tracker,
                [(odd_time, ikey(x, y)), (even_time, current)],
                coord_keys=[current],
                time=even_time,
            )

    for n in range(1, d + 1):
        a = ikey(2 * n, 0)
        b = ikey(2 * n + 1, 0)
        _append_standard_view_if_present(
            view,
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
            _append_standard_view_if_present(
                view,
                lines,
                tracker,
                [(odd_time, ikey(d, y0)), (even_time, ikey(d, y1))],
                coord_keys=[ikey(d, y1)],
                time=even_time,
            )

    for n in range(1, d):
        a = ikey(0, 2 * n - 1)
        b = ikey(0, 2 * n)
        _append_standard_view_if_present(
            view,
            lines,
            tracker,
            [(odd_time, a), (odd_time, b), (even_time, a), (even_time, b)],
            coord_keys=[a, b],
            time=even_time,
        )

    for x0 in range(1, d, 2):
        x1 = x0 + 1
        _append_standard_view_if_present(
            view,
            lines,
            tracker,
            [(odd_time, ikey(x0, d)), (even_time, ikey(x1, d))],
            coord_keys=[ikey(x1, d)],
            time=even_time,
        )


def append_standard_view_even_to_odd_detectors(
    view: StandardOnBridgeAdapter,
    lines: list[str],
    tracker: MeasurementTracker,
    even_time: int,
    odd_time: int,
) -> None:
    d = view.d
    for x in range(1, d):
        for y in range(1, d):
            current = ikey(x + 1, y) if (x + y) % 2 == 1 else ikey(x, y + 1)
            _append_standard_view_if_present(
                view,
                lines,
                tracker,
                [(even_time, ikey(x, y)), (odd_time, current)],
                coord_keys=[current],
                time=odd_time,
            )

    for n in range(1, d):
        a = ikey(2 * n - 1, 0)
        b = ikey(2 * n, 0)
        _append_standard_view_if_present(
            view,
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
            _append_standard_view_if_present(
                view,
                lines,
                tracker,
                [(even_time, ikey(d, y0)), (odd_time, ikey(d, y1))],
                coord_keys=[ikey(d, y1)],
                time=odd_time,
            )

    for n in range(1, d):
        a = ikey(0, 2 * n)
        b = ikey(0, 2 * n + 1)
        _append_standard_view_if_present(
            view,
            lines,
            tracker,
            [(even_time, a), (even_time, b), (odd_time, a), (odd_time, b)],
            coord_keys=[a, b],
            time=odd_time,
        )

    for x0 in range(2, d, 2):
        x1 = x0 + 1
        _append_standard_view_if_present(
            view,
            lines,
            tracker,
            [(even_time, ikey(x0, d)), (odd_time, ikey(x1, d))],
            coord_keys=[ikey(x1, d)],
            time=odd_time,
        )


def append_standard_view_detectors_between(
    view: StandardOnBridgeAdapter,
    lines: list[str],
    tracker: MeasurementTracker,
    *,
    previous_time: int,
    current_time: int,
) -> None:
    if previous_time % 2 == 1 and current_time % 2 == 0:
        append_standard_view_odd_to_even_detectors(view, lines, tracker, previous_time, current_time)
    elif previous_time % 2 == 0 and current_time % 2 == 1:
        append_standard_view_even_to_odd_detectors(view, lines, tracker, previous_time, current_time)
    else:
        raise ValueError("detectors must connect adjacent odd/even half-cycles")


def append_standard_view_memory_detector(
    view: StandardOnBridgeAdapter,
    lines: list[str],
    tracker: MeasurementTracker,
    refs: Iterable[tuple[int, CoordKey]],
    coord_key: CoordKey,
    time: int,
) -> None:
    append_standard_view_detector(view, lines, tracker, refs, coord_keys=[coord_key], time=time)


def append_wide_entry_detectors(
    wide: WideOnBridgeAdapter,
    lines: list[str],
    tracker: MeasurementTracker,
    *,
    basis: str,
    time: int,
    known_data_qubits: Iterable[int],
) -> None:
    known_data = set(known_data_qubits)
    if time % 2 == 0:
        specs = wide_memory.odd_to_even_boundary_specs(wide)
        refs_attr = "even_refs"
    else:
        specs = wide_memory.even_to_odd_boundary_specs(wide)
        refs_attr = "odd_refs"
    for spec in specs:
        if spec.basis != basis:
            continue
        if any(
            wide.half[((key[0] - 1) // 2, (key[1] - 1) // 2)] not in known_data
            for key in spec.data_refs
        ):
            continue
        wide_memory.append_memory_detector(
            wide,
            lines,
            tracker,
            [(time, key) for key in getattr(spec, refs_attr)],
            spec.coord_key,
            time=time,
        )


def append_standard_to_wide_right_x_transition_detectors(
    standard: StandardOnBridgeAdapter,
    wide: WideOnBridgeAdapter,
    lines: list[str],
    tracker: MeasurementTracker,
    *,
    standard_time: int,
    wide_time: int,
) -> None:
    """Connect the transposed top standard right-side X history into wide-H."""
    d = wide.d
    if standard.placement != "top" or standard.orientation != "transpose":
        raise ValueError("standard-to-wide transition expects the transposed top standard block")

    def standard_local_key_for_global(global_key: CoordKey) -> CoordKey:
        return (global_key[1], global_key[0])

    for y in range(0, d, 2):
        if y == 0:
            indices = [
                tracker.rounds[standard_time][standard_local_key_for_global(ikey(d - 1, 0))],
                tracker.rounds[standard_time][standard_local_key_for_global(ikey(d, 0))],
                tracker.rounds[wide_time][ikey(d - 1, 0)],
                tracker.rounds[wide_time][ikey(d, 0)],
            ]
        else:
            indices = [
                tracker.rounds[standard_time][standard_local_key_for_global(ikey(d - 1, y))],
                tracker.rounds[wide_time][ikey(d, y)],
            ]
        append_measurement_index_detector(
            lines,
            tracker,
            indices,
            coord_keys=[ikey(d, y)],
            time=wide_time,
        )


def append_standard_to_wide_transition_detectors(
    standard: StandardOnBridgeAdapter,
    wide: WideOnBridgeAdapter,
    lines: list[str],
    tracker: MeasurementTracker,
    *,
    standard_time: int,
    wide_time: int,
) -> None:
    """Connect the last standard-code symptoms into the first wide round."""
    if standard.placement != "top" or standard.orientation != "transpose":
        raise ValueError("standard-to-wide transition expects the transposed top standard block")

    def standard_local_key_for_global(global_key: CoordKey) -> CoordKey:
        return (global_key[1], global_key[0])

    if wide_time % 2 == 0:
        specs = wide_memory.odd_to_even_boundary_specs(wide)
        previous_refs_attr = "odd_refs"
        current_refs_attr = "even_refs"
    else:
        specs = wide_memory.even_to_odd_boundary_specs(wide)
        previous_refs_attr = "even_refs"
        current_refs_attr = "odd_refs"

    for spec in specs:
        if spec.basis == "Z" and any(key[1] > 2 * wide.d for key in spec.data_refs):
            continue
        previous_refs = tuple(standard_local_key_for_global(key) for key in getattr(spec, previous_refs_attr))
        current_refs = tuple(getattr(spec, current_refs_attr))
        if not all(key in tracker.rounds.get(standard_time, {}) for key in previous_refs):
            continue
        if not all(key in tracker.rounds.get(wide_time, {}) for key in current_refs):
            continue
        indices = [tracker.rounds[standard_time][key] for key in previous_refs]
        indices.extend(tracker.rounds[wide_time][key] for key in current_refs)
        append_measurement_index_detector(
            lines,
            tracker,
            indices,
            coord_keys=[spec.coord_key],
            time=wide_time,
        )


def standard_view_local_key_for_global(
    view: StandardOnBridgeAdapter,
    global_key: CoordKey,
) -> CoordKey:
    """Invert ``StandardOnBridgeAdapter.to_global_key`` for detector refs."""
    x, y = global_key
    if view.placement == "bottom":
        y -= 2 * view.d
    if view.orientation == "transpose":
        local_key = (y, x)
    else:
        local_key = (x, y)
    if view.to_global_key(local_key) != global_key:
        raise ValueError(
            f"global key ({global_key[0] / 2:g}, {global_key[1] / 2:g}) "
            "is not in the requested standard view"
        )
    return local_key


def append_wide_to_bridge_transition_detectors(
    bridge: bridge_framework.BridgeWideGenerator,
    lines: list[str],
    tracker: MeasurementTracker,
    *,
    wide_time: int,
    bridge_time: int,
) -> None:
    """Connect the ordinary wide round into the bridge round where possible."""
    for relation in bridge_framework.specified_repeat_detector_relations(
        bridge,
        previous_time=wide_time,
        current_time=bridge_time,
    ):
        if (
            relation.family.startswith("internal-Z")
            and relation.current_keys == (ikey(1, 2 * bridge.d),)
        ):
            corner = hkey(0, 2 * bridge.d - 1)
            if corner not in tracker.rounds.get(wide_time, {}):
                continue
            previous_keys = tuple(relation.previous_keys) + (corner,)
            if not all(key in tracker.rounds.get(wide_time, {}) for key in previous_keys):
                continue
            if not all(key in tracker.rounds.get(bridge_time, {}) for key in relation.current_keys):
                continue
            indices = [tracker.rounds[wide_time][key] for key in previous_keys]
            indices.extend(tracker.rounds[bridge_time][key] for key in relation.current_keys)
            append_measurement_index_detector(
                lines,
                tracker,
                indices,
                coord_keys=relation.current_keys,
                time=bridge_time,
            )
            continue
        if any(key[0] == 0 for key in relation.previous_keys):
            continue
        if not all(key in tracker.rounds.get(wide_time, {}) for key in relation.previous_keys):
            continue
        if not all(key in tracker.rounds.get(bridge_time, {}) for key in relation.current_keys):
            continue
        indices = [tracker.rounds[wide_time][key] for key in relation.previous_keys]
        indices.extend(tracker.rounds[bridge_time][key] for key in relation.current_keys)
        append_measurement_index_detector(
            lines,
            tracker,
            indices,
            coord_keys=relation.current_keys,
            time=bridge_time,
        )


def append_bridge_to_standard_transition_detectors(
    bridge: bridge_framework.BridgeWideGenerator,
    standard: StandardOnBridgeAdapter,
    lines: list[str],
    tracker: MeasurementTracker,
    *,
    bridge_time: int,
    shrink_time: int,
    standard_time: int,
) -> None:
    """Connect the bridge exit into the first lower-standard syndrome round."""
    if standard.placement != "bottom" or standard.orientation != "transpose":
        raise ValueError("bridge-to-standard transition expects the transposed bottom standard block")
    d = bridge.d

    measured_upper_data = set(tracker.rounds.get(shrink_time, {}))
    for stab in bridge_framework.specified_first_half_stabilizers(bridge):
        if stab.color != "Z":
            continue
        if not set(stab.final_data_keys).issubset(measured_upper_data):
            continue
        if not all(key in tracker.rounds.get(bridge_time, {}) for key in stab.auxiliary_keys):
            continue
        indices = [tracker.rounds[bridge_time][key] for key in stab.auxiliary_keys]
        indices.extend(tracker.rounds[shrink_time][key] for key in stab.final_data_keys)
        append_measurement_index_detector(
            lines,
            tracker,
            indices,
            coord_keys=[(round(2 * stab.center[0]), round(2 * stab.center[1]))],
            time=shrink_time,
        )

    # The lower block starts in its local first half-cycle.  Because d is odd,
    # this is the physical translation of the required second-half pattern.
    # Interior and left-boundary checks therefore use the ordinary even-to-odd
    # relation.  The other three boundaries contract at the code switch.
    for spec in standard_memory.even_to_odd_boundary_specs(standard):
        previous_global_keys = tuple(standard.to_global_key(key) for key in spec.even_refs)
        current_global_keys = tuple(standard.to_global_key(key) for key in spec.odd_refs)
        is_right_x = spec.basis == "X" and all(key[0] == 2 * d for key in current_global_keys)
        is_top_z = spec.basis == "Z" and all(key[1] == 2 * d for key in current_global_keys)
        is_bottom_z = spec.basis == "Z" and all(key[1] == 4 * d for key in current_global_keys)

        refs: list[tuple[int, CoordKey]] = []
        if is_right_x or is_bottom_z:
            refs.extend((bridge_time, key) for key in previous_global_keys)
            refs.extend((bridge_time, key) for key in current_global_keys)
        elif is_top_z:
            refs.append((bridge_time, min(current_global_keys)))
            left_x = min(key[0] for key in current_global_keys) // 2
            refs.extend(
                [
                    (shrink_time, hkey(left_x - 1, d - 1)),
                    (shrink_time, hkey(left_x, d - 1)),
                ]
            )
        else:
            refs.extend((bridge_time, key) for key in previous_global_keys)
        refs.extend((standard_time, key) for key in spec.odd_refs)

        if not all(key in tracker.rounds.get(time, {}) for time, key in refs):
            continue
        append_measurement_index_detector(
            lines,
            tracker,
            [tracker.rounds[time][key] for time, key in refs],
            coord_keys=[standard.to_global_key(spec.coord_key)],
            time=standard_time,
        )


def append_bridge_repeat_detectors(
    bridge: bridge_framework.BridgeWideGenerator,
    lines: list[str],
    tracker: MeasurementTracker,
    *,
    previous_time: int,
    current_time: int,
) -> None:
    for relation in bridge_framework.specified_repeat_detector_relations(
        bridge,
        previous_time=previous_time,
        current_time=current_time,
    ):
        indices = []
        for key in relation.previous_keys:
            indices.append(tracker.rounds[previous_time][key])
        for key in relation.current_keys:
            indices.append(tracker.rounds[current_time][key])
        append_measurement_index_detector(
            lines,
            tracker,
            indices,
            coord_keys=relation.current_keys,
            time=current_time,
        )

    if current_time % 2 == 0:
        key = ikey(1, 0)
        if (
            key in tracker.rounds.get(previous_time, {})
            and key in tracker.rounds.get(current_time, {})
        ):
            append_measurement_index_detector(
                lines,
                tracker,
                [
                    tracker.rounds[previous_time][key],
                    tracker.rounds[current_time][key],
                ],
                coord_keys=[key],
                time=current_time,
            )


def append_bridge_column0_singleton_detectors(
    bridge: bridge_framework.BridgeWideGenerator,
    lines: list[str],
    tracker: MeasurementTracker,
    *,
    time: int,
) -> None:
    for y in range(1, 2 * bridge.d + 1):
        key = ikey(0, y)
        if y == 1:
            if time % 2 == 0 and key in tracker.rounds.get(time - 1, {}):
                append_measurement_index_detector(
                    lines,
                    tracker,
                    [tracker.rounds[time - 1][key]],
                    coord_keys=[key],
                    time=time,
                )
            continue
        if key not in tracker.rounds.get(time, {}):
            continue
        append_measurement_index_detector(
            lines,
            tracker,
            [tracker.rounds[time][key]],
            coord_keys=[key],
            time=time,
        )


def append_final_standard_closure_detectors(
    view: StandardOnBridgeAdapter,
    lines: list[str],
    tracker: MeasurementTracker,
    *,
    basis: str,
    syndrome_time: int,
    syndrome_second_half: bool,
    final_time: int,
) -> None:
    if syndrome_second_half:
        specs = standard_memory.even_to_odd_boundary_specs(view)
        refs_attr = "even_refs"
    else:
        specs = standard_memory.odd_to_even_boundary_specs(view)
        refs_attr = "odd_refs"
    for spec in specs:
        if spec.basis != basis:
            continue
        append_standard_view_memory_detector(
            view,
            lines,
            tracker,
            [(syndrome_time, key) for key in getattr(spec, refs_attr)]
            + [(final_time, key) for key in spec.data_refs],
            spec.coord_key,
            time=final_time,
        )


def _xor_indices(indices: Iterable[int]) -> list[int]:
    counts: dict[int, int] = {}
    for index in indices:
        counts[index] = counts.get(index, 0) ^ 1
    return sorted(index for index, keep in counts.items() if keep)


def standard_data_qubits(view: StandardOnBridgeAdapter) -> list[int]:
    return sorted(view.half.values())


def wide_data_qubits(wide: WideOnBridgeAdapter) -> list[int]:
    return sorted(q for key, q in wide.half.items() if key != (0, 2 * wide.d - 1))


def final_standard_logical_keys(distance: int, basis: str) -> list[CoordKey]:
    if basis == "X":
        return [hkey(distance - 1, y) for y in range(distance)]
    if basis == "Z":
        return [hkey(x, distance - 1) for x in range(distance)]
    raise ValueError("basis must be X or Z")


def final_standard_x_observable_aux_keys(distance: int) -> list[CoordKey]:
    """Measured auxiliaries completing the translated lower-code logical X."""
    if distance < 3 or distance % 2 == 0:
        raise ValueError("distance must be an odd integer at least 3")
    return [ikey(distance, 0)] + [ikey(x, distance) for x in range(3, distance, 2)]


def _standard_measure_h(view: StandardOnBridgeAdapter, *, second_half: bool) -> set[int]:
    return set(view.h2_measure() if second_half else view.h1_measure())


def _wide_prepare_h(wide: WideOnBridgeAdapter, *, second_half: bool) -> set[int]:
    return set(wide.h2_prepare() if second_half else wide.h1_prepare())


def _wide_measure_h(wide: WideOnBridgeAdapter, *, second_half: bool) -> set[int]:
    return set(wide.h2_measure() if second_half else wide.h1_measure())


def _bridge_measure_h(
    bridge: bridge_framework.BridgeWideGenerator,
    *,
    second_half: bool,
) -> set[int]:
    return set(bridge.h2_measure() if second_half else bridge.h1_measure())


def _standard_aux_indices_by_global_predicate(
    view: StandardOnBridgeAdapter,
    tracker: MeasurementTracker,
    *,
    time: int,
    h_set: set[int],
    use_h: bool,
    predicate: callable,
) -> list[int]:
    out: list[int] = []
    for (x, y), q in view.integer.items():
        local_key = ikey(x, y)
        if local_key not in tracker.rounds.get(time, {}):
            continue
        global_key = view.to_global_key(local_key)
        if predicate(global_key) and ((q in h_set) == use_h):
            out.append(tracker.rounds[time][local_key])
    return out


def _active_aux_indices(
    generator: object,
    tracker: MeasurementTracker,
    *,
    time: int,
    h_set: set[int] | None = None,
    use_h: bool | None = None,
    predicate: callable,
) -> list[int]:
    out: list[int] = []
    for (x, y), q in generator.integer.items():
        key = ikey(x, y)
        if key not in tracker.rounds.get(time, {}):
            continue
        if h_set is not None and use_h is not None and ((q in h_set) != use_h):
            continue
        if predicate(key):
            out.append(tracker.rounds[time][key])
    return out


def _indices_for_aux_keys(
    tracker: MeasurementTracker,
    *,
    time: int,
    keys: Iterable[CoordKey],
    label: str,
) -> list[int]:
    out: list[int] = []
    for key in keys:
        if key not in tracker.rounds.get(time, {}):
            raise KeyError(f"{label}: missing auxiliary measurement at ({key[0] / 2:g},{key[1] / 2:g},{time})")
        out.append(tracker.rounds[time][key])
    return out


def specified_h_flow_observable_indices(
    *,
    distance: int,
    tracker: MeasurementTracker,
    initial_standard: StandardOnBridgeAdapter,
    wide: WideOnBridgeAdapter,
    bridge: bridge_framework.BridgeWideGenerator,
    final_standard: StandardOnBridgeAdapter,
    first_standard_time: int,
    standard_last_time: int,
    first_wide_time: int | None,
    wide_last_time: int | None,
    first_bridge_time: int | None,
    bridge_last_time: int | None,
    first_post_time: int | None,
    post_standard_last_time: int | None,
    final_time: int | None,
) -> dict[str, list[int]]:
    """Return post-H output X/Z observable rows from the user's prescription."""
    d = distance
    output_x: list[int] = []
    output_z: list[int] = []

    # Initial transposed standard memory.  Before the transversal H, the
    # initial Z logical becomes output X and the initial X logical becomes
    # output Z.
    for time in range(first_standard_time, standard_last_time + 1):
        measure_h = _standard_measure_h(initial_standard, second_half=(time % 2 == 0))
        output_z.extend(
            _standard_aux_indices_by_global_predicate(
                initial_standard,
                tracker,
                time=time,
                h_set=measure_h,
                use_h=True,
                predicate=lambda key, d=d: key[1] == 2 * d,
            )
        )
        output_x.extend(
            _standard_aux_indices_by_global_predicate(
                initial_standard,
                tracker,
                time=time,
                h_set=measure_h,
                use_h=False,
                predicate=lambda key, d=d: key[0] == 2 * d,
            )
        )

    # Ordinary wide-H stage.  The first wide round has a special Z switch.
    # Its X side still uses the odd-round terminal spread from the user's
    # prescription; MARK pragmas in the reference circuit mark the init side.
    if first_wide_time is not None and wide_last_time is not None:
        prepare_h = _wide_prepare_h(wide, second_half=(first_wide_time % 2 == 0))
        # The oracle circuit labels these first wide-round initial-X-flow
        # measurement terms with MX. They are not the MARK support itself.
        output_z.extend(
            _active_aux_indices(
                wide,
                tracker,
                time=first_wide_time,
                predicate=lambda key, d=d: key[0] == 0 and 2 * d <= key[1] < 4 * d,
            )
        )
        output_z.extend(
            _indices_for_aux_keys(
                tracker,
                time=first_wide_time,
                keys=[hkey(0, 2 * d - 1)],
                label="wide entry Z corner observable",
            )
        )
        output_z.extend(
            _active_aux_indices(
                wide,
                tracker,
                time=first_wide_time,
                h_set=prepare_h,
                use_h=True,
                predicate=lambda key, d=d: 0 < key[0] < 2 * d and 2 * d < key[1] < 4 * d,
            )
        )
        output_z.extend(
            _active_aux_indices(
                wide,
                tracker,
                time=first_wide_time,
                predicate=lambda key, d=d: 4 <= key[0] <= 2 * d and key[1] == 4 * d,
            )
        )
        output_z.extend(
            _indices_for_aux_keys(
                tracker,
                time=first_wide_time,
                keys=[ikey(d, d)],
                label="wide entry Z right contraction observable",
            )
        )
        measure_h = _wide_measure_h(wide, second_half=(first_wide_time % 2 == 0))
        output_x.extend(
            _active_aux_indices(
                wide,
                tracker,
                time=first_wide_time,
                h_set=measure_h,
                use_h=True,
                predicate=lambda key, d=d: key[0] == 2 * d and key[1] <= 2 * d,
            )
        )

        for time in range(first_wide_time + 1, wide_last_time + 1):
            measure_h = _wide_measure_h(wide, second_half=(time % 2 == 0))
            if time % 2 == 1:
                output_x.extend(
                    _active_aux_indices(
                        wide,
                        tracker,
                        time=time,
                        h_set=measure_h,
                        use_h=True,
                        predicate=lambda key, d=d: key[0] == 2 * d and key[1] <= 2 * (d + 1),
                    )
                )
            else:
                output_x.extend(
                    _active_aux_indices(
                        wide,
                        tracker,
                        time=time,
                        h_set=measure_h,
                        use_h=True,
                        predicate=lambda key, d=d: key[0] == 2 * d and key[1] <= 2 * d,
                    )
                )
                output_z.extend(
                    _indices_for_aux_keys(
                        tracker,
                        time=time,
                        keys=[ikey(d, d), ikey(d, 2 * d)],
                        label="wide even Z observable",
                    )
                )

    # User-specified bridge-wide stage.  The first bridge round is the entry
    # switch; later rounds use the bridge odd/even observable flow.
    if first_bridge_time is not None and bridge_last_time is not None:
        measure_h = _bridge_measure_h(bridge, second_half=(first_bridge_time % 2 == 0))
        output_x.extend(
            _active_aux_indices(
                bridge,
                tracker,
                time=first_bridge_time,
                h_set=measure_h,
                use_h=True,
                predicate=lambda key, d=d: (
                    key[1] <= 2 * d and key[0] != 0 and key != ikey(d, 0)
                ),
            )
        )
        output_x.extend(
            _indices_for_aux_keys(
                tracker,
                time=first_bridge_time,
                keys=[ikey(d, d + 1)],
                label="bridge entry X observable",
            )
        )

        for time in range(first_bridge_time + 1, bridge_last_time + 1):
            if time % 2 == 1:
                output_x.extend(
                    _indices_for_aux_keys(
                        tracker,
                        time=time,
                        keys=[ikey(0, d + 1), ikey(d, d + 1)],
                        label="bridge odd X observable",
                    )
                )
            else:
                output_z.extend(
                    _indices_for_aux_keys(
                        tracker,
                        time=time,
                        keys=[ikey(d, d), ikey(d, 2 * d)],
                        label="bridge even Z observable",
                    )
                )

    # Final bottom standard stage, again in physical coordinates.
    if first_post_time is not None and post_standard_last_time is not None:
        for time in range(first_post_time, post_standard_last_time + 1):
            measure_h = _standard_measure_h(
                final_standard,
                second_half=((time - first_post_time) % 2 == 1),
            )
            output_x.extend(
                _standard_aux_indices_by_global_predicate(
                    final_standard,
                    tracker,
                    time=time,
                    h_set=measure_h,
                    use_h=True,
                    predicate=lambda key, d=d: key[1] == 2 * d,
                )
            )
            if time == first_post_time:
                output_x.extend(
                    _indices_for_aux_keys(
                        tracker,
                        time=time,
                        keys=final_standard_x_observable_aux_keys(d),
                        label="translated final-standard X observable",
                    )
                )
            output_z.extend(
                _standard_aux_indices_by_global_predicate(
                    final_standard,
                    tracker,
                    time=time,
                    h_set=measure_h,
                    use_h=False,
                    predicate=lambda key, d=d: key[0] == 2 * d,
                )
            )

    if final_time is not None:
        output_x.extend(tracker.rounds[final_time][key] for key in final_standard_logical_keys(d, "X"))
        output_z.extend(tracker.rounds[final_time][key] for key in final_standard_logical_keys(d, "Z"))

    return {
        "X": _xor_indices(output_x),
        "Z": _xor_indices(output_z),
    }


def build_h_gate_flow_without_observable(
    distance: int,
    basis: str,
    *,
    observable_basis: str | None = None,
    standard_rounds: int | None = None,
    wide_rounds: int | None = None,
    bridge_rounds: int | None = None,
    post_standard_rounds: int = 1,
    final_measure_output: bool = True,
    validate: bool = True,
) -> HGateUserSpecBuild:
    basis = _validate_basis(basis)
    output_basis = "Z" if basis == "X" else "X"
    observable_basis = output_basis if observable_basis is None else _validate_basis(observable_basis)
    standard_rounds = distance if standard_rounds is None else standard_rounds
    wide_rounds = distance if wide_rounds is None else wide_rounds
    bridge_rounds = distance if bridge_rounds is None else bridge_rounds
    if standard_rounds < 1 or wide_rounds < 0 or bridge_rounds < 0 or post_standard_rounds < 0:
        raise ValueError("round counts must be non-negative, with standard_rounds >= 1")
    if bridge_rounds and bridge_rounds % 2 != 1:
        raise ValueError("bridge_rounds must be odd so the user-specified final half is odd")

    bridge = bridge_spec.SpecBridgeWideGenerator(distance, include_polygons=False)
    wide = WideOnBridgeAdapter(bridge, include_polygons=False)
    initial_standard = StandardOnBridgeAdapter(bridge, placement="top", orientation="transpose")
    final_standard = StandardOnBridgeAdapter(bridge, placement="bottom", orientation="transpose")
    tracker = MeasurementTracker(bridge)
    lines = bridge.qubit_coord_lines()

    initial_standard_data = standard_data_qubits(initial_standard)
    final_standard_data = standard_data_qubits(final_standard)
    wide_data = set(wide_data_qubits(wide))
    wide_initialized_data = sorted(wide_data.difference(initial_standard_data))
    bridge_promoted_data = [bridge.half[(0, 2 * distance - 1)]]
    upper_measured_data = sorted(set(bridge.data_qubits()).difference(final_standard_data))

    lines.append("# Stage 1: transposed top standard block, physical left/right X boundaries.")
    lines.append("TICK")
    _append_op(lines, _op_for_basis(basis, measurement=False), initial_standard_data)

    first_standard_time = 2
    standard_last_time = first_standard_time + standard_rounds - 1
    standard_specs = standard_memory.odd_to_even_boundary_specs(initial_standard)
    for time in range(first_standard_time, standard_last_time + 1):
        append_standard_view_measurement_round(
            initial_standard,
            lines,
            tracker,
            time=time,
            second_half=(time % 2 == 0),
        )
        if time == first_standard_time:
            for spec in standard_specs:
                if spec.basis == basis:
                    append_standard_view_memory_detector(
                        initial_standard,
                        lines,
                        tracker,
                        [(time, key) for key in spec.even_refs],
                        spec.coord_key,
                        time=time,
                    )
        else:
            append_standard_view_detectors_between(
                initial_standard,
                lines,
                tracker,
                previous_time=time - 1,
                current_time=time,
            )

    lines.append("# Stage 2: transversal H maps physical left/right X to top/bottom X.")
    lines.append("TICK")
    _append_op(lines, "H", initial_standard_data)

    lines.append("# Stage 3: initialize lower wide-H data in the X basis.")
    lines.append("TICK")
    _append_op(lines, "RX", wide_initialized_data)

    wide_last_time: int | None = None
    first_wide_time: int | None = None
    if wide_rounds:
        first_wide_time = standard_last_time + 1
        if first_wide_time % 2 == 1:
            first_wide_time += 1
        wide_last_time = first_wide_time + wide_rounds - 1
        lines.append("# Stage 4: ordinary wide-H memory.")
        for index, time in enumerate(range(first_wide_time, wide_last_time + 1)):
            wide.append_measurement_round(
                lines,
                tracker,
                time=time,
                second_half=(time % 2 == 0),
            )
            if index == 0:
                append_wide_entry_detectors(
                    wide,
                    lines,
                    tracker,
                    basis="X",
                    time=time,
                    known_data_qubits=wide_initialized_data,
                )
                append_standard_to_wide_transition_detectors(
                    initial_standard,
                    wide,
                    lines,
                    tracker,
                    standard_time=standard_last_time,
                    wide_time=time,
                )
            else:
                wide.append_detectors_between(
                    lines,
                    tracker,
                    previous_time=time - 1,
                    current_time=time,
                )

    bridge_last_time: int | None = None
    first_bridge_time: int | None = None
    if bridge_rounds:
        base_time = wide_last_time if wide_last_time is not None else standard_last_time
        lines.append("# Stage 5: promote the lower-left wide auxiliary half-qubit into bridge data.")
        lines.append("TICK")
        _append_op(lines, "R", bridge_promoted_data)

        first_bridge_time = base_time + 1
        if first_bridge_time % 2 == 0:
            first_bridge_time += 1
        bridge_last_time = first_bridge_time + bridge_rounds - 1
        lines.append("# Stage 6: user-specified bridge-wide memory.")
        for index, time in enumerate(range(first_bridge_time, bridge_last_time + 1)):
            bridge.append_measurement_round(
                lines,
                tracker,
                time=time,
                second_half=(time % 2 == 0),
            )
            append_bridge_column0_singleton_detectors(bridge, lines, tracker, time=time)
            if index == 0 and wide_last_time is not None:
                append_wide_to_bridge_transition_detectors(
                    bridge,
                    lines,
                    tracker,
                    wide_time=wide_last_time,
                    bridge_time=time,
                )
            elif index > 0:
                append_bridge_repeat_detectors(
                    bridge,
                    lines,
                    tracker,
                    previous_time=time - 1,
                    current_time=time,
                )

    shrink_time: int | None = None
    if bridge_last_time is not None:
        shrink_time = bridge_last_time + 1
        lines.append("# Stage 7: measure upper bridge data; keep rows d+0.5 through 2d-0.5.")
        lines.append("TICK")
        _append_op(lines, "M", upper_measured_data)
        tracker.record_round(shrink_time, upper_measured_data)

    post_standard_last_time: int | None = None
    first_post_time: int | None = None
    if post_standard_rounds:
        base_time = shrink_time if shrink_time is not None else (
            bridge_last_time if bridge_last_time is not None else (
                wide_last_time if wide_last_time is not None else standard_last_time
            )
        )
        first_post_time = base_time + 1
        if first_post_time % 2 == 1:
            first_post_time += 1
        post_standard_last_time = first_post_time + post_standard_rounds - 1
        lines.append("# Stage 8: transposed bottom standard block, physical left/right X boundaries.")
        for index, time in enumerate(range(first_post_time, post_standard_last_time + 1)):
            append_standard_view_measurement_round(
                final_standard,
                lines,
                tracker,
                time=time,
                second_half=(index % 2 == 1),
            )
            if (
                index == 0
                and bridge_last_time is not None
                and shrink_time is not None
            ):
                append_bridge_to_standard_transition_detectors(
                    bridge,
                    final_standard,
                    lines,
                    tracker,
                    bridge_time=bridge_last_time,
                    shrink_time=shrink_time,
                    standard_time=time,
                )

    final_time: int | None = None
    before_final_data = tracker.count
    final_logical_indices: list[int] = []
    observable_indices: list[int] = []
    observable_indices_by_basis: dict[str, list[int]] = {"X": [], "Z": []}
    if final_measure_output:
        base_time = post_standard_last_time or shrink_time or bridge_last_time or wide_last_time or standard_last_time
        final_time = base_time + 1
        lines.append("# Stage 9: destructively measure the restored lower standard data.")
        lines.append("TICK")
        _append_op(lines, _op_for_basis(output_basis, measurement=True), final_standard_data)
        before_final_data = tracker.count
        _record_standard_view_round(tracker, final_standard, final_time, final_standard_data)
        if post_standard_last_time is not None:
            append_final_standard_closure_detectors(
                final_standard,
                lines,
                tracker,
                basis=output_basis,
                syndrome_time=post_standard_last_time,
                syndrome_second_half=(post_standard_rounds % 2 == 0),
                final_time=final_time,
            )
        observable_indices_by_basis = specified_h_flow_observable_indices(
            distance=distance,
            tracker=tracker,
            initial_standard=initial_standard,
            wide=wide,
            bridge=bridge,
            final_standard=final_standard,
            first_standard_time=first_standard_time,
            standard_last_time=standard_last_time,
            first_wide_time=first_wide_time,
            wide_last_time=wide_last_time,
            first_bridge_time=first_bridge_time,
            bridge_last_time=bridge_last_time,
            first_post_time=first_post_time,
            post_standard_last_time=post_standard_last_time,
            final_time=final_time,
        )
        final_logical_indices = [
            tracker.rounds[final_time][key]
            for key in final_standard_logical_keys(distance, observable_basis)
        ]
        observable_indices = list(observable_indices_by_basis[observable_basis])

    text_without_observable = "\n".join(lines) + "\n"
    if validate:
        stim.Circuit(text_without_observable).detector_error_model(decompose_errors=True)

    return HGateUserSpecBuild(
        text_without_observable=text_without_observable,
        tracker=tracker,
        input_basis=basis,
        output_basis=output_basis,
        standard_last_time=standard_last_time,
        wide_last_time=wide_last_time,
        bridge_last_time=bridge_last_time,
        shrink_time=shrink_time,
        post_standard_last_time=post_standard_last_time,
        final_time=final_time,
        before_final_data_measurements=before_final_data,
        final_logical_indices=final_logical_indices,
        observable_indices=observable_indices,
        observable_indices_by_basis=observable_indices_by_basis,
        observable_basis=observable_basis,
        initial_standard_data=initial_standard_data,
        wide_initialized_data=wide_initialized_data,
        bridge_promoted_data=bridge_promoted_data,
        upper_measured_data=upper_measured_data,
        final_standard_data=final_standard_data,
        standard_rounds=standard_rounds,
        wide_rounds=wide_rounds,
        bridge_rounds=bridge_rounds,
        post_standard_rounds=post_standard_rounds,
    )


def measurement_record_terms_from_indices(
    total_measurements: int,
    indices: Iterable[int],
) -> list[str]:
    return [f"rec[{index - total_measurements}]" for index in sorted(indices)]


def append_observable(build: HGateUserSpecBuild) -> str:
    if not build.observable_indices:
        return build.text_without_observable
    total = stim.Circuit(build.text_without_observable).num_measurements
    terms = measurement_record_terms_from_indices(total, build.observable_indices)
    return build.text_without_observable + "OBSERVABLE_INCLUDE(0) " + " ".join(terms) + "\n"


def generate_h_gate_flow_circuit(
    distance: int,
    basis: str,
    *,
    p1: float = 0.0,
    p2: float = 0.0,
    standard_rounds: int | None = None,
    wide_rounds: int | None = None,
    bridge_rounds: int | None = None,
    post_standard_rounds: int = 1,
    include_observable: bool = False,
    validate: bool = True,
) -> str:
    build = build_h_gate_flow_without_observable(
        distance,
        basis,
        standard_rounds=standard_rounds,
        wide_rounds=wide_rounds,
        bridge_rounds=bridge_rounds,
        post_standard_rounds=post_standard_rounds,
        validate=validate,
    )
    text = append_observable(build) if include_observable else build.text_without_observable
    if validate:
        stim.Circuit(text).detector_error_model(decompose_errors=True)
    text = add_gate_noise(text, p1=p1, p2=p2)
    if validate:
        stim.Circuit(text).detector_error_model(decompose_errors=True)
    return text

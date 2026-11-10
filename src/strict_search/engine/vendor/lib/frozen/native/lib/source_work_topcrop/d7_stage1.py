"""2d+1-row top-cropped reconstruction with re-indexed physical y coordinates.

Start from the original 2d+2-row construction, delete the old y=0 data/ancilla
row, and keep the already-validated top-boundary closure.  Internal qubit IDs
continue to use the old y indices (1..2d+1), but physical coordinates are
re-indexed by y_new = y_old - 1.  Therefore the domain-wall seam stored at
internal row d+1 is physically located at y=d+0.5, immediately below the
final/readout d-row block at y=0.5..d-0.5.

This is an experimental stage-1 constructor. It must pass the d=5 regression
before any d=7 continuation or distance claim is made.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import stim




class Stage1:
    def __init__(self, distance: int):
        if distance < 5 or distance % 2 != 1:
            raise ValueError("distance must be odd and >=5")
        self.d = distance
        self.h = 2 * distance + 2
        self.stride = 2 * self.h + 1
        self.base = self.h + 1
        self.right_seam_width = (distance - 1) // 2
        self.circuit = stim.Circuit()
        self.segments: dict[str, stim.Circuit] = {}

    def anc(self, x: int, y: int) -> int:
        assert 0 <= x <= self.d and 0 <= y <= self.h
        return self.stride * x + y

    def data(self, x: int, y: int) -> int:
        assert 0 <= x < self.d and 0 <= y < self.h
        return self.stride * x + self.base + y

    def coords(self) -> dict[int, tuple[float, float]]:
        # Old row y=0 is physically deleted.  Surviving old row y>=1 is
        # re-indexed to new row y-1; this is the actual 2d+1-row geometry.
        # Omitting y=0 from this map intentionally makes any accidental use
        # of a deleted-row qubit fail during coordinate annotation.
        out = {}
        for x in range(self.d + 1):
            for y in range(1, self.h + 1):
                out[self.anc(x, y)] = (float(x), float(y - 1))
        for x in range(self.d):
            for y in range(1, self.h):
                out[self.data(x, y)] = (x + 0.5, y - 0.5)
        return out

    def append_segment(self, name: str, c: stim.Circuit) -> None:
        self.segments[name] = c
        self.circuit += c

    @staticmethod
    def append_each(c: stim.Circuit, name: str, qs: list[int]) -> None:
        for q in qs:
            c.append(name, [q])

    @staticmethod
    def append_layer(c: stim.Circuit, pairs: list[tuple[int, int]]) -> None:
        flat = [q for p in pairs for q in p]
        assert len(flat) == len(set(flat)), "non-disjoint CX layer"
        c.append("CX", flat)
        c.append("TICK")

    def data_init(self, basis: str) -> None:
        c = stim.Circuit()
        data = [self.data(x, y) for x in range(self.d) for y in range(1, self.d + 1)]
        self.append_each(c, "R", data)
        c.append("TICK")
        if basis == "X":
            self.append_each(c, "H", data)
            c.append("TICK")
        elif basis != "Z":
            raise ValueError(basis)
        self.append_segment("data_init", c)

    def standard_h(self) -> tuple[list[int], list[int]]:
        before = []
        after = []
        for y in range(self.d + 1, 0, -1):
            for x in range(self.d + 1):
                if x == 0 and y == self.d + 1:
                    continue
                q = self.anc(x, y)
                if x == 0:
                    before.append(q)
                elif y == 1:
                    before.append(q)
                    if x < self.d:
                        after.append(q)
                elif y == self.d + 1:
                    after.append(q)
                elif x < self.d:
                    (before if (x + y) % 2 else after).append(q)
        return before, after

    def standard_round(self, phase: str, index: int) -> None:
        c = stim.Circuit()
        active = [self.anc(x, y)
                  for y in range(self.d + 1, 0, -1)
                  for x in range(self.d + 1) if not (x == 0 and y == self.d + 1)]
        before, after = self.standard_h()
        if phase == "A":
            before, after = after, before
        self.append_each(c, "R", active)
        c.append("TICK")
        self.append_each(c, "H", before)
        c.append("TICK")
        layers = [[], [], [], []]
        for i in range(self.d):
            for j in range(1, self.d + 1):
                q = self.data(i, j)
                up_right = self.anc(i + 1, j)
                up_left = self.anc(i, j)
                down_right = self.anc(i + 1, j + 1)
                odd = (i + j) % 2 == 1
                pair = (q, up_right) if odd else (up_right, q)
                layers[0].append(pair)
                layers[3].append((pair[1], pair[0]))
                left_pair = (up_left, q)
                down_pair = (q, down_right)
                layers[1].append(left_pair if odd else down_pair)
                layers[2].append(down_pair if odd else left_pair)
        if phase == "A":
            layers.reverse()
        for layer in layers:
            self.append_layer(c, layer)
        self.append_each(c, "H", after)
        c.append("TICK")
        self.append_each(c, "M", active)
        c.append("TICK")
        self.append_segment(f"standard_{index}_{phase}", c)

    def expansion(self) -> None:
        c = stim.Circuit()
        # Top-cropped geometry: y=0 data are deleted. Preserve y=1..d and
        # initialize only the lower extension y=d+1..h-1.
        data = [self.data(x, y) for x in range(self.d)
                for y in range(self.d + 1, self.h)]
        self.append_each(c, "R", data)
        c.append("TICK")
        self.append_each(c, "H", data)
        c.append("TICK")
        self.append_segment("expansion", c)

    def wide_h(self) -> tuple[list[int], list[int]]:
        before = []
        after = []
        for x in range(1, self.d + 1):
            # y=0 ancillas belonged to the deleted row.  On the new top
            # boundary y=1, use the standard-code rough-boundary convention:
            # x=1..d-1 are H'd both before and after so their alternating
            # gauge products give the two-body X boundary checks.  The old
            # top-right corner ancilla (d,1) is omitted; the right Z boundary
            # starts at (d,2), avoiding a spurious weight-1 corner check.
            for y in range(1, self.h + 1):
                if x == self.d and y == 1:
                    continue
                q = self.anc(x, y)
                if y == 1 and x < self.d:
                    before.append(q)
                    after.append(q)
                elif x < self.d:
                    if y < self.h:
                        (before if (x + y) % 2 == 0 else after).append(q)
                elif self.d + 2 <= y < self.h:
                    (before if (x + y) % 2 == 0 else after).append(q)
        return before, after

    def wide_round_circuit(self, phase: str) -> stim.Circuit:
        c = stim.Circuit()
        active = [self.anc(x, y) for x in range(1, self.d + 1)
                  for y in range(1, self.h + 1)
                  if not (x == self.d and y == 1)]
        a_before, a_after = self.wide_h()
        before, after = (a_before, a_after) if phase == "A" else (a_after, a_before)
        before_set = set(before)
        after_set = set(after)
        # The old corner ancilla (d,0) is outside the cropped geometry.
        reset_first = before_set
        measure_first = after_set
        self.append_each(c, "R", sorted(reset_first) + sorted(set(active) - reset_first))
        c.append("TICK")
        self.append_each(c, "H", sorted(before))
        c.append("TICK")
        layers = [[], [], [], []]
        for i in range(self.d):
            # data row j=0 (physical y=0.5) is deleted. Internal rows retain
            # exactly the original four-layer interaction pattern.
            for j in range(1, self.h):
                q = self.data(i, j)
                up_right = self.anc(i + 1, j)
                down_right = self.anc(i + 1, j + 1)
                even = (i + j) % 2 == 0
                pair = (q, up_right) if even else (up_right, q)
                if up_right in active:
                    layers[0].append(pair)
                    layers[3].append((pair[1], pair[0]))
                left_anc = self.anc(i, j) if i else None
                left_pair = (left_anc, q) if left_anc in active else None
                down_pair = (q, down_right) if down_right in active else None
                mid1 = left_pair if even else down_pair
                mid2 = down_pair if even else left_pair
                if mid1:
                    layers[1].append(mid1)
                if mid2:
                    layers[2].append(mid2)
        if phase == "B":
            layers.reverse()
        for layer in layers:
            self.append_layer(c, layer)
        self.append_each(c, "H", sorted(after))
        c.append("TICK")
        self.append_each(c, "M", sorted(measure_first) + sorted(set(active) - measure_first))
        c.append("TICK")
        return c

    def wide_round(self, phase: str, index: int) -> None:
        self.append_segment(f"wide_{index}_{phase}", self.wide_round_circuit(phase))

    def my_seam(self, side: str) -> None:
        c = stim.Circuit()
        k = self.right_seam_width
        indices = range(self.d - k, self.d) if side == "right" else range(self.d - k)
        self.append_each(c, "MY", [self.data(i, self.d + 1) for i in indices])
        c.append("TICK")
        self.append_segment(f"wall_MY_{k}" if side == "right" else f"wall_MY_{self.d-k}", c)

    def twist_round(self, phase: str, index: int) -> None:
        base = self.wide_round_circuit(phase)
        k = self.right_seam_width
        seam = [self.data(i, self.d + 1) for i in range(self.d - k, self.d)]
        seam_set = set(seam)
        special = [self.anc(i, self.d + 1) for i in range(self.d - k + 1, self.d + 1)]
        active = {self.anc(x, y) for x in range(1, self.d + 1)
                  for y in range(1, self.h + 1)
                  if not (x == self.d and y == 1)} | seam_set
        out = stim.Circuit()
        moment = 0
        for op in base:
            if op.name == "TICK":
                if not (moment == 7 and index == 2 * self.d + 5):
                    out.append("TICK")
                moment += 1
                continue
            ts = [t.value for t in op.targets_copy()]
            if op.name in ("R", "M"):
                if phase == "A":
                    ts = sorted(active)
                else:
                    h_before, h_after = self.wide_h()
                    h_group = h_after if op.name == "R" else h_before
                    first = set(h_group) | seam_set | set(special)
                    ts = sorted(first) + sorted(active - first)
            elif op.name == "H":
                ts = sorted(set(ts) | seam_set | set(special))
            elif op.name == "CX":
                replaced = [(a, b) for a, b in zip(ts[::2], ts[1::2])
                            if a in seam_set or b in seam_set]
                ts = [q for a, b in zip(ts[::2], ts[1::2])
                      if (a not in seam_set and b not in seam_set)
                      or (moment in (3, 4) and (b if a in seam_set else a) % self.stride != self.d + 1)
                      for q in (a, b)]
                out.append("CX", ts)
                if moment in (2, 5):
                    assert len(replaced) == k
                    out.append("CZ", [q for a, b in replaced
                                      for q in ((a, b) if a not in seam_set else (b, a))])
                elif moment == (3 if phase == "B" else 4):
                    out.append("S_DAG" if phase == "B" else "S",
                               [self.anc(self.d - k, self.d + 1)])
                continue
            out.append(op.name, ts)
        self.append_segment(f"twist_memory_{index}_{phase}", out)

    def build(self, basis: str) -> stim.Circuit:
        self.data_init(basis)
        for k in range(self.d):
            self.standard_round("B" if k % 2 == 0 else "A", k + 1)
        self.expansion()
        for k, phase in enumerate("ABA", self.d + 1):
            self.wide_round(phase, k)
        return self.circuit

    def build_through_twist(self, basis: str) -> stim.Circuit:
        self.build(basis)
        self.my_seam("right")
        for k in range(self.d):
            self.twist_round("B" if k % 2 == 0 else "A", self.d + 6 + k)
        return self.circuit

    def fullwall_round(self, phase: str, index: int) -> None:
        base = self.wide_round_circuit(phase)
        seam_row = self.d + 1
        seam = {self.data(i, seam_row) for i in range(self.d)}
        left = {self.anc(0, j) for j in range(seam_row, self.h)}
        active = {self.anc(x, y) for x in range(1, self.d + 1)
                  for y in range(1, self.h + 1)
                  if not (x == self.d and y == 1)} | seam | left
        out = stim.Circuit()
        moment = 0
        for op in base:
            if op.name == "TICK":
                if moment != 7:
                    out.append("TICK")
                moment += 1
                continue
            ts = [t.value for t in op.targets_copy()]
            if op.name in ("R", "M"):
                ts = (sorted(active - left) + sorted(left)) if phase == "A" else sorted(active)
            elif op.name == "H":
                extra = seam | {self.anc(x, seam_row) for x in range(1, self.d + 1)}
                extra |= left - ({self.anc(0, seam_row)} if (phase == "A") == (moment == 1) else set())
                ts = sorted(set(ts) | extra)
            elif op.name == "CX":
                pairs = list(zip(ts[::2], ts[1::2]))
                replaced = [(a, b) for a, b in pairs if a in seam or b in seam] if moment in (2, 5) else []
                pairs = [(a, b) for a, b in pairs
                         if (a not in seam and b not in seam)
                         or (moment in (3, 4) and (b if a in seam else a) % self.stride != seam_row)]
                if moment in (3, 4):
                    for j in range(seam_row, self.h):
                        layer = (3 if j % 2 == 0 else 4) if phase == "A" else (4 if j % 2 == 0 else 3)
                        if moment == layer:
                            pairs.append((self.anc(0, j), self.data(0, j)))
                out.append("CX", [q for a, b in pairs for q in (a, b)])
                if moment in (2, 5):
                    assert len(replaced) == self.d
                    out.append("CZ", [q for a, b in replaced
                                      for q in ((a, b) if a not in seam else (b, a))])
                continue
            out.append(op.name, ts)
        self.append_segment(f"fullwall_{index}_{phase}", out)

    def h_lower(self) -> None:
        c = stim.Circuit()
        c.append("H", [self.data(i, j) for i in range(self.d)
                       for j in range(self.d + 2, self.h)])
        c.append("TICK")
        self.append_segment("H_lower", c)

    def shrink(self) -> None:
        c = stim.Circuit()
        # Keep old y=1..d as the new first d rows for terminal readout.
        # The seam row y=d+1 is already handled by the wall measurements.
        removed = [self.data(i, j) for i in range(self.d)
                   for j in range(self.d + 2, self.h)]
        c.append("H", removed)
        c.append("TICK")
        c.append("M", removed)
        c.append("TICK")
        self.append_segment("shrink", c)

    def build_all(self, basis: str) -> stim.Circuit:
        self.build_through_twist(basis)
        self.my_seam("left")
        for k in range(self.d):
            self.fullwall_round("A" if k % 2 == 0 else "B", k + 1)
        self.h_lower()
        self.shrink()
        return self.circuit


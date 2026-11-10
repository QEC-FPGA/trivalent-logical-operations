"""Cumulative GF(2) logical-flow audit of the frozen-compatible stage 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import stim

from d7_stage1 import Stage1


def logicals(builder: Stage1, stage: str) -> dict[str, stim.PauliString]:
    n = builder.stride * builder.d + builder.h + 1
    x = stim.PauliString(n)
    z = stim.PauliString(n)
    if stage == "shrink":
        out_x = stim.PauliString(n)
        out_z = stim.PauliString(n)
        for j in range(1, builder.d + 1):
            out_x[builder.data(0, j)] = "X"
        for i in range(builder.d):
            out_z[builder.data(i, 1)] = "Z"
        out_y = out_x * out_z
        out_y *= 1j
        return {"X": out_y, "Z": out_z, "Y": out_x}
    if stage == "H_lower":
        for i in range(builder.d):
            z[builder.data(i, 1)] = "Z"
        h_y = stim.PauliString(n)
        for j in list(range(1, builder.d + 1)) + list(range(builder.d + 2, builder.h)):
            h_y[builder.data(builder.d - 1, j)] = "X"
        x = z * h_y
        x *= 1j
        return {"X": x, "Z": z, "Y": h_y}
    if stage.startswith("wall_MY_") or stage.startswith("twist_memory_") or stage.startswith("fullwall_"):
        for i in range(builder.d):
            z[builder.data(i, 1)] = "Z"
        wall_y = stim.PauliString(n)
        for j in range(1, builder.d + 1):
            wall_y[builder.data(builder.d - 1, j)] = "X"
        for j in range(builder.d + 2, builder.h):
            wall_y[builder.data(builder.d - 1, j)] = "Z"
        x = z * wall_y
        x *= 1j
        return {"X": x, "Z": z, "Y": wall_y}
    if stage.startswith("standard_") or stage == "input":
        for j in range(1, builder.d + 1):
            x[builder.data(0, j)] = "X"
    else:
        row = builder.d + 2 if stage.endswith("_B") else builder.d + 1
        for j in range(1, row + 1):
            x[builder.data(0, j)] = "X"
        for i in range(1, builder.d):
            x[builder.data(i, row)] = "X"
    for i in range(builder.d):
        z[builder.data(i, 1)] = "Z"
    y = x * z
    y *= 1j
    return {"X": x, "Z": z, "Y": y}


def flow(circuit: stim.Circuit, pin: stim.PauliString, pout: stim.PauliString) -> dict:
    bare = stim.Flow(input=pin, output=pout)
    solution = circuit.solve_flow_measurements([bare])[0]
    if solution is None:
        return {"exists": False}
    plus = circuit.has_flow(stim.Flow(input=pin, output=pout, measurements=solution))
    minus = circuit.has_flow(stim.Flow(input=pin, output=-pout, measurements=solution))
    assert plus ^ minus
    return {"exists": True, "sign": 1 if plus else -1,
            "measurement_count": len(solution), "measurements": list(solution)}


#!/usr/bin/env python3
"""Generate one current-mainline H-flow or memory .stim circuit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_ROOT
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import bridge_wide_memory_user_spec as bridge_memory
import h_gate_memory_flow_user_spec as h_flow
import standard_trivalent_memory as standard_memory
import wide_h_memory as wide_memory


SAMPLE_DIR_BY_KIND = {
    "standard": PACKAGE_ROOT / "samples" / "standard_memory",
    "wide": PACKAGE_ROOT / "samples" / "wide_memory",
    "bridge": PACKAGE_ROOT / "samples" / "bridge_wide_memory",
    "h_flow": PACKAGE_ROOT / "samples" / "h_gate_flow",
}


def prob_token(value: float) -> str:
    if value == 0:
        return "0"
    return f"{value:g}".replace(".", "p").replace("-", "m")


def default_output_path(
    *,
    kind: str,
    distance: int,
    basis: str,
    rounds: int | None,
    p1: float,
    p2: float,
    standard_rounds: int | None,
    wide_rounds: int | None,
    bridge_rounds: int | None,
    post_standard_rounds: int,
) -> Path:
    p1_text = prob_token(p1)
    p2_text = prob_token(p2)
    if kind == "standard":
        r = distance if rounds is None else rounds
        name = f"standard_memory_d{distance:02d}_r{r:02d}_basis_{basis}_p1_{p1_text}_p2_{p2_text}.stim"
    elif kind == "wide":
        r = distance if rounds is None else rounds
        name = f"wide_memory_d{distance:02d}_r{r:02d}_basis_{basis}_p1_{p1_text}_p2_{p2_text}.stim"
    elif kind == "bridge":
        r = distance if rounds is None else rounds
        name = (
            f"bridge_wide_user_spec_memory_d{distance:02d}_r{r:02d}_basis_{basis}"
            f"_p1_{p1_text}_p2_{p2_text}.stim"
        )
    elif kind == "h_flow":
        std = distance if standard_rounds is None else standard_rounds
        wide = distance if wide_rounds is None else wide_rounds
        bridge = distance if bridge_rounds is None else bridge_rounds
        name = (
            f"h_gate_user_spec_d{distance:02d}_std{std:02d}_wide{wide:02d}"
            f"_bridge{bridge:02d}_poststd{post_standard_rounds:02d}_input_{basis}"
            f"_p1_{p1_text}_p2_{p2_text}.stim"
        )
    else:
        raise ValueError(f"unknown kind: {kind}")
    return SAMPLE_DIR_BY_KIND[kind] / name


def generate_circuit_text(args: argparse.Namespace) -> str:
    if args.kind == "standard":
        return standard_memory.generate_memory_circuit(
            args.distance,
            args.basis,
            rounds=args.rounds,
            p1=args.p1,
            p2=args.p2,
        )
    if args.kind == "wide":
        return wide_memory.generate_memory_circuit(
            args.distance,
            args.basis,
            rounds=args.rounds,
            p1=args.p1,
            p2=args.p2,
        )
    if args.kind == "bridge":
        return bridge_memory.generate_memory_circuit(
            args.distance,
            args.basis,
            rounds=args.rounds,
            p1=args.p1,
            p2=args.p2,
        )

    standard_rounds = args.standard_rounds
    wide_rounds = args.wide_rounds
    bridge_rounds = args.bridge_rounds
    if args.rounds is not None:
        standard_rounds = args.rounds if standard_rounds is None else standard_rounds
        wide_rounds = args.rounds if wide_rounds is None else wide_rounds
        bridge_rounds = args.rounds if bridge_rounds is None else bridge_rounds
    return h_flow.generate_h_gate_flow_circuit(
        args.distance,
        args.basis,
        p1=args.p1,
        p2=args.p2,
        standard_rounds=standard_rounds,
        wide_rounds=wide_rounds,
        bridge_rounds=bridge_rounds,
        post_standard_rounds=args.post_standard_rounds,
        include_observable=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["standard", "wide", "bridge", "h_flow"], required=True)
    parser.add_argument("-d", "--distance", type=int, default=5)
    parser.add_argument("--basis", choices=["X", "Z"], default="Z")
    parser.add_argument("--rounds", type=int, default=None, help="Memory rounds; for h_flow, shorthand for std/wide/bridge.")
    parser.add_argument("--p1", type=float, default=0.0, help="Single-qubit/reset/measurement noise probability.")
    parser.add_argument("--p2", type=float, default=0.0, help="Two-qubit CX noise probability.")
    parser.add_argument("--standard-rounds", type=int, default=None)
    parser.add_argument("--wide-rounds", type=int, default=None)
    parser.add_argument("--bridge-rounds", type=int, default=None)
    parser.add_argument("--post-standard-rounds", type=int, default=1)
    parser.add_argument("-o", "--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or default_output_path(
        kind=args.kind,
        distance=args.distance,
        basis=args.basis,
        rounds=args.rounds,
        p1=args.p1,
        p2=args.p2,
        standard_rounds=args.standard_rounds,
        wide_rounds=args.wide_rounds,
        bridge_rounds=args.bridge_rounds,
        post_standard_rounds=args.post_standard_rounds,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    text = generate_circuit_text(args)
    output.write_text(text, encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

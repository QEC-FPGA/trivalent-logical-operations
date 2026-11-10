"""Compile one independently validated full-boundary S-gate condition.

The physics construction is copied from the frozen 2026-10-02 XY/ZZ release.
Only (d, W) selection and the output location are parameterized. A body-only
reference boundary is deliberately never generated here.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import stim
from sample_shard import scaled_circuit

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "lib"))
from common import moving, base, wc  # noqa: E402
from attempt import extract, basis, solve, annotate, bits  # noqa: E402

OLD_PHYSICAL_SHA256 = {
    # SHA-256 of the canonical Stim circuit text, independent of CRLF/LF files.
    (5, 3, "XY"): "a7898a11dd4b19782a5fe2bd59d29e8d10da1dd1f97ec73f6cd615289fb63b31",
    (5, 3, "ZZ"): "385bbfc7711711e274aad9acb00be239cd94db76f2256c0267401f9e2f6880b9",
    (7, 5, "XY"): "2bd76b5f693ed1842e3ca83a6aa9bc19b1e8712f1dac65a0569bde5cc861df2e",
    (7, 5, "ZZ"): "4c3631e570d569bd4c7011e5c55d9f1de2fd687ef1d6fa24f89e475e58ddc2bc",
    (9, 5, "XY"): "439aecfc3447cbb7b433e7959daad5e58e3681905a58ba959e560ebe5f450884",
    (9, 5, "ZZ"): "11586129f13083aa3c0f9d051b3dbb37adf85bb3337b734d6da1ce68fa8021ff",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def mpp(circuit: stim.Circuit, pauli: stim.PauliString) -> None:
    targets = []
    for qubit, value in enumerate(pauli):
        if not value:
            continue
        if targets:
            targets.append(stim.target_combiner())
        targets.append(getattr(stim, "target_" + "IXYZ"[value].lower())(qubit))
    circuit.append("MPP", targets)


def code_stabilizers(builder, body: stim.Circuit):
    # Keep the Bell-reference quotient and signed flow check from generate_xy.py.
    n, d = builder.circuit.num_qubits, builder.d
    data = [builder.data(i, j) for i in range(d) for j in range(1, d + 1)]
    data_set = set(data)
    st = []
    for q in range(n):
        if q not in data_set:
            p = stim.PauliString(n + 1)
            p[q] = "Z"
            st.append(p)
    for check in base.standard_stabilizers(d):
        p = stim.PauliString(n + 1)
        for q, value in zip(data, check):
            p[q] = value
        st.append(p)
    for label in "XZ":
        p = base.logicals(builder, "input")[label].copy()
        p += stim.PauliString(n + 1 - len(p))
        p[n] = label
        st.append(p)
    sim = stim.TableauSimulator(seed=279)
    sim.set_inverse_tableau(stim.Tableau.from_stabilizers(st).inverse())
    sim.do(body)
    N = sim.num_qubits
    outside = ((1 << (2 * N)) - 1) ^ sum((1 << q) | (1 << (N + q)) for q in data)
    pivots, output = {}, []
    for p in sim.canonical_stabilizers():
        vector = sum((int(k in (1, 2)) << q) | (int(k in (2, 3)) << (N + q)) for q, k in enumerate(p))
        while vector & outside:
            k = (vector & outside).bit_length() - 1
            if k not in pivots:
                pivots[k] = vector
                break
            vector ^= pivots[k]
        if not vector & outside and vector:
            row = stim.PauliString(n)
            for q in data:
                row[q] = "IXZY"[(vector >> q & 1) + 2 * (vector >> (N + q) & 1)]
            output.append(row)
    if len(output) != d * d - 1:
        raise ValueError(f"Output stabilizer rank {len(output)} != {d*d-1}")
    probe = stim.Circuit()
    probe.append("R", [q for q in range(n) if q not in data_set])
    for check in base.standard_stabilizers(d):
        row = stim.PauliString(n)
        for q, value in zip(data, check):
            row[q] = value
        mpp(probe, row)
    probe += body
    from strict_repair.batched_codegen import batch_terminal_flows
    flows = batch_terminal_flows(probe, stim.PauliString(n), output)
    if not all(f["exists"] for f in flows):
        raise ValueError("At least one terminal code stabilizer has no signed flow")
    return output, flows


def physical_circuit(d: int, w: int, channel: str):
    builder = moving.builder(d, w, 3)
    body = base.engine.body_after_init(builder)
    prep_builder = base.Stage1(d)
    prep_builder.data_init("X" if channel == "XY" else "Z")
    prep = prep_builder.circuit
    checks, flows = code_stabilizers(builder, body)
    logical = base.logicals(builder, "shrink")["X" if channel == "XY" else "Z"]
    if not all(p.commutes(logical) for p in checks):
        raise ValueError("Terminal checks do not commute with logical MPP")
    terminal = stim.Circuit()
    for p in checks:
        mpp(terminal, p)
    mpp(terminal, logical)
    physical = base.engine.with_coordinates(prep + body + terminal, builder)
    prefix = prep + body
    sol = base.flow(prefix, stim.PauliString(physical.num_qubits), logical)
    if not sol["exists"]:
        raise ValueError("Missing preparation-to-terminal logical flow")
    obs = sorted(set(sol["measurements"] + [physical.num_measurements - 1]))
    dets, rank = base.complete_detectors(physical, obs, [physical.num_measurements - 1], None)
    if len(dets) + 1 != rank:
        raise ValueError("Incomplete detector parity rank")
    ideal = base.annotate(physical, dets, obs)
    syndrome, observable = ideal.compile_detector_sampler(seed=731).sample(64, separate_observables=True)
    if syndrome.any() or observable.any():
        raise ValueError("Noiseless full circuit has a nonzero detector or observable")
    full_noise = base.engine.with_coordinates(base.engine.si1000_with_ideal_mpp(prep + body, .001) + terminal, builder)
    full = base.annotate(full_noise, dets, obs)
    if full.without_noise() != ideal:
        raise ValueError("Noise transform changed physical operations or reference boundary")
    metadata = {
        "d": d, "W": w, "T": 3, "channel": channel,
        "boundary": "full", "faults_start_tick": 0,
        "noise": "SI1000 p=0.001 template, rescaled per sample condition",
        "terminal": "ideal output-code checks and logical MPP",
        "detectors": len(dets), "parity_rank": rank,
        "terminal_code_rank": len(checks), "observable_records": obs,
        "expected_raw_observable": int(sol["sign"] == -1),
        "body_sha256": sha(str(body).encode()),
        "physical_sha256": sha(str(physical).encode()),
        "flow_checks": len(flows),
    }
    return physical, full, metadata


def strict_dem(circuit: stim.Circuit) -> tuple[bool, str]:
    try:
        circuit.detector_error_model(
            decompose_errors=True,
            ignore_decomposition_failures=False,
            block_decomposition_from_introducing_remnant_edges=True,
        )
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def audited_d13_w1_xy(original: stim.Circuit) -> tuple[stim.Circuit, dict]:
    """Apply the independently audited, full-rank native-strict detector basis.

    This is deliberately exact-case only. It must never be silently applied to
    another distance, waiting time, channel, or physical circuit revision.
    """
    folder = ROOT / "strict_repair" / "d13_W1_XY"
    reference_original = stim.Circuit.from_file(str(folder / "reference_original.stim"))
    if original != reference_original:
        raise ValueError("d13/W1/XY original circuit differs from audited reference")
    rows, _ = extract(original)
    transform = json.loads((folder / "detector_transform.json").read_text(encoding="utf-8"))
    if len(rows) != 10992 or len(transform) != 10992:
        raise ValueError("Audited detector transform has the wrong dimension")
    transformed = []
    for mask in transform:
        row = 0
        for index in mask:
            if not isinstance(index, int) or not 0 <= index < len(rows):
                raise ValueError("Invalid source detector index in audited transform")
            row ^= rows[index]
        transformed.append(row)
    candidate = annotate(original, transformed)
    reference_strict = stim.Circuit.from_file(str(folder / "reference_strict.stim"))
    if candidate != reference_strict:
        raise ValueError("Audited detector transform did not reproduce the frozen strict circuit")
    return candidate, {"source": "audited_d13_W1_XY_native_strict_20261006",
                       "reference_original_sha256": sha((folder / "reference_original.stim").read_bytes()),
                       "reference_strict_sha256": sha((folder / "reference_strict.stim").read_bytes()),
                       "detector_transform_sha256": sha((folder / "detector_transform.json").read_bytes())}


def sparse_reduce(circuit: stim.Circuit, *, seconds: float = 300) -> tuple[stim.Circuit, list]:
    rows, _ = extract(circuit)
    dem = circuit.detector_error_model(decompose_errors=False)
    columns = [set(t.val for t in op.targets_copy() if t.is_relative_detector_id())
               for op in dem if op.type == "error"]
    incidence = [set() for _ in rows]
    for j, col in enumerate(columns):
        for i in col:
            incidence[i].add(j)
    start, logs = time.monotonic(), []
    for pass_index in range(50):
        changes = 0
        for i in sorted(range(len(rows)), key=lambda i: len(incidence[i]), reverse=True):
            overlaps = collections.Counter(j for col in incidence[i] for j in columns[col] if j != i)
            choices = [(2 * ov - len(incidence[j]), j) for j, ov in overlaps.items()
                       if 2 * ov > len(incidence[j])]
            if not choices:
                continue
            _, other = min(choices, key=lambda x: (-x[0], (rows[i] ^ rows[x[1]]).bit_count(), x[1]))
            for col in tuple(incidence[other]):
                if col in incidence[i]:
                    incidence[i].remove(col)
                    columns[col].remove(i)
                else:
                    incidence[i].add(col)
                    columns[col].add(i)
            rows[i] ^= rows[other]
            changes += 1
        logs.append({"pass": pass_index, "changes": changes, "seconds": time.monotonic() - start})
        if not changes or time.monotonic() - start > seconds:
            break
    return annotate(circuit, rows), logs


def prefix_local(circuit: stim.Circuit, *, seconds: float = 900) -> tuple[stim.Circuit, dict]:
    original, measurements = extract(circuit)
    original_basis = basis(original)
    if len(original_basis) != circuit.num_detectors:
        raise ValueError("Original detector rows are dependent")
    candidates = set(original)
    prefix = stim.Circuit()
    start = time.monotonic()
    cuts = 0
    from reverse_faults import nmeas, NOISE
    for op in circuit:
        if op.name in NOISE or op.name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            continue
        prefix.append(op)
        if not nmeas(op):
            continue
        # Ideal terminal MPP products commute. Keep the original complete
        # basis and inspect their whole boundary once below, instead of
        # re-solving almost the same long prefix after every product.
        if op.name == "MPP":
            continue
        cuts += 1
        if time.monotonic() - start > seconds:
            raise TimeoutError("Prefix-local detector search time budget exhausted")
        for flow in prefix.flow_generators():
            if any(flow.input_copy()) or any(flow.output_copy()):
                continue
            mask = sum(1 << i for i in flow.measurements_copy())
            if mask and solve(mask, original_basis) is not None:
                candidates.add(mask)

    for flow in prefix.flow_generators():
        if any(flow.input_copy()) or any(flow.output_copy()):
            continue
        mask = sum(1 << i for i in flow.measurements_copy())
        if mask and solve(mask, original_basis) is not None:
            candidates.add(mask)

    def score(row):
        ids = list(bits(row))
        ticks = [measurements[i]["tick"] for i in ids]
        xy = [measurements[i].get("coordinate") for i in ids]
        xy = [q for q in xy if q]
        spread = (max(q[0] for q in xy) - min(q[0] for q in xy)
                  + max(q[1] for q in xy) - min(q[1] for q in xy)) if xy else 0
        return row.bit_count(), max(ticks) - min(ticks), spread, max(ticks), row

    selected, pivots = [], {}
    for row in sorted(candidates, key=score):
        value = row
        while value:
            key = value.bit_length() - 1
            if key not in pivots:
                pivots[key] = value
                selected.append(row)
                break
            value ^= pivots[key]
    if len(selected) != len(original):
        raise ValueError("Prefix-local rows do not span the detector space")
    return annotate(circuit, selected), {"cuts": cuts, "candidates": len(candidates)}


def compile_one(d: int, w: int, channel: str, output_root: Path) -> dict:
    if d < 5 or d % 2 != 1 or w < 1 or w % 2 != 1 or channel not in ("XY", "ZZ"):
        raise ValueError("Need odd d>=5, positive odd W, and XY or ZZ")
    dest = output_root / f"d{d}_W{w}_{channel}"
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite existing compile directory: {dest}")
    dest.mkdir(parents=True)
    start = time.monotonic()
    attempts = []
    try:
        physical, original, meta = physical_circuit(d, w, channel)
        old_hash = OLD_PHYSICAL_SHA256.get((d, w, channel))
        if old_hash and meta["physical_sha256"] != old_hash:
            raise ValueError("Frozen 2026-10-02 physical circuit hash differs")
        physical.to_file(str(dest / "physical.stim"))
        original.to_file(str(dest / "full_original.stim"))
        if (d, w, channel) == (13, 1, "XY"):
            repaired, info = audited_d13_w1_xy(original)
            repaired.to_file(str(dest / "audited_native_strict_candidate.stim"))
            primary_candidates = (("audited_native_strict", repaired, info),)
        else:
            primary_candidates = (("original", original, {}),)
        for name, candidate, details in primary_candidates:
            ok, error = strict_dem(candidate)
            attempts.append({"basis": name, "graphlike": ok, "error": error, **details})
            if ok:
                accepted, accepted_name = candidate, name
                break
            if name == "audited_native_strict":
                raise ValueError("Frozen audited d13/W1/XY circuit failed native strict DEM: " + error)
        else:
            reduced, logs = sparse_reduce(original)
            reduced.to_file(str(dest / "sparse_candidate.stim"))
            ok, error = strict_dem(reduced)
            attempts.append({"basis": "sparse", "graphlike": ok, "error": error, "passes": logs})
            if ok:
                accepted, accepted_name = reduced, "sparse"
            else:
                local, info = prefix_local(original)
                local.to_file(str(dest / "prefix_local_candidate.stim"))
                ok, error = strict_dem(local)
                attempts.append({"basis": "prefix_local", "graphlike": ok, "error": error, **info})
                if ok:
                    accepted, accepted_name = local, "prefix_local"
                else:
                    reduced, logs = sparse_reduce(local)
                    reduced.to_file(str(dest / "prefix_local_sparse_candidate.stim"))
                    ok, error = strict_dem(reduced)
                    attempts.append({"basis": "prefix_local_sparse", "graphlike": ok, "error": error, "passes": logs})
                    if not ok:
                        from strict_repair.general_search import repair
                        def repair_checkpoint(candidate, operations, event):
                            candidate.to_file(str(dest / "general_repair_checkpoint.stim"))
                            atomic_json(dest / "general_repair_checkpoint.json", {"operations": operations, "last_event": event})
                        candidate, repair_info = repair(
                            reduced, extract=extract, annotate=annotate,
                            strict_dem=strict_dem, seconds=900,
                            callback=repair_checkpoint,
                        )
                        atomic_json(dest / "general_repair.json", repair_info)
                        attempts.append({"basis": "general_pair_repair",
                                         "graphlike": candidate is not None,
                                         "seconds": repair_info["seconds"],
                                         "operations": len(repair_info["operations"]),
                                         "report": "general_repair.json"})
                        if candidate is None:
                            raise ValueError("No equivalent detector basis yielded strict graphlike DEM within bounded search")
                        candidate.to_file(str(dest / "general_repair_candidate.stim"))
                        accepted, accepted_name = candidate, "general_pair_repair"
                    else:
                        accepted, accepted_name = reduced, "prefix_local_sparse"
        # Detector bases differ; compare physical operations and the observable separately.
        def without_annotations(c):
            out = stim.Circuit()
            for op in c:
                if op.name not in ("DETECTOR", "OBSERVABLE_INCLUDE"):
                    out.append(op)
            return out

        def observable_rows(c):
            return [str(op) for op in c if op.name == "OBSERVABLE_INCLUDE"]

        if (without_annotations(accepted) != without_annotations(original)
                or observable_rows(accepted) != observable_rows(original)):
            raise ValueError("Detector basis transformation changed physical instructions or observable")
        original_rows, _ = extract(original)
        accepted_rows, _ = extract(accepted)
        original_basis = basis(original_rows)
        if (len(original_rows) != len(accepted_rows)
                or len(original_basis) != len(original_rows)
                or len(basis(accepted_rows)) != len(accepted_rows)
                or any(solve(row, original_basis) is None for row in accepted_rows)):
            raise ValueError("Accepted detector rows do not form an equivalent full-rank basis")
        s, o = accepted.without_noise().compile_detector_sampler(seed=739).sample(128, separate_observables=True)
        if s.any() or o.any():
            raise ValueError("Accepted noiseless detector basis is nondeterministic")
        accepted.to_file(str(dest / "full_matching.stim"))
        dem_hashes = {}
        for probability in (.001, .002):
            variant = scaled_circuit(accepted, probability)
            model = variant.detector_error_model(
                decompose_errors=True,
                ignore_decomposition_failures=False,
                block_decomposition_from_introducing_remnant_edges=True,
            )
            dem_path = dest / f"p{probability:.3f}.dem"
            model.to_file(str(dem_path))
            dem_hashes[f"{probability:.3f}"] = sha(dem_path.read_bytes())
        meta.update({"status": "ACCEPTED", "basis": accepted_name,
                     "full_original_sha256": sha((dest / "full_original.stim").read_bytes()),
                     "full_matching_sha256": sha((dest / "full_matching.stim").read_bytes()),
                     "dem_sha256": dem_hashes,
                     "basis_attempts": attempts, "seconds": time.monotonic() - start})
        atomic_json(dest / "compile.json", meta)
        return meta
    except Exception as exc:
        atomic_json(dest / "compile.json", {"status": "UNSUPPORTED", "d": d, "W": w,
                    "channel": channel, "error": f"{type(exc).__name__}: {exc}",
                    "seconds": time.monotonic() - start,
                    "basis_attempts": attempts})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--w", type=int, required=True)
    parser.add_argument("--channel", choices=["XY", "ZZ"], required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compile_one(args.d, args.w, args.channel, args.output_root), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

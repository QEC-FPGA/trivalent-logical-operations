"""One independently seeded, atomic, physical-circuit Monte Carlo shard."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
if os.environ.get("S_GATE_DECODER_RUNTIME"):
    sys.path.insert(0, os.environ["S_GATE_DECODER_RUNTIME"])

import numpy as np
import stim

NOISE = {"X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1", "DEPOLARIZE2"}


def scaled_circuit(template: stim.Circuit, p: float) -> stim.Circuit:
    if p not in (.001, .002):
        raise ValueError("Only frozen p=0.001 and p=0.002 are allowed")
    out = stim.Circuit()
    for op in template:
        if op.name in NOISE:
            out.append(op.name, op.targets_copy(), [x * p / .001 for x in op.gate_args_copy()])
        else:
            out.append(op)
    return out


def _atomic_json(path: Path, obj: dict) -> None:
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(obj, sort_keys=True, indent=2), encoding="utf-8")
    os.replace(temp, path)


def sample(*, circuit_path: Path, compile_meta: Path, p: float, shots: int,
           seed: int, output: Path, batch: int = 4096) -> dict:
    import pymatching

    if pymatching.__version__ != "2.3.1":
        raise RuntimeError(f"Pinned PyMatching 2.3.1 required; found {pymatching.__version__}")
    if shots < 1 or batch < 1:
        raise ValueError("shots and batch must be positive")
    meta = json.loads(compile_meta.read_text(encoding="utf-8"))
    if meta.get("status") != "ACCEPTED" or meta.get("boundary") != "full":
        raise ValueError("Compiler did not accept a full-boundary circuit")
    if hashlib.sha256(circuit_path.read_bytes()).hexdigest() != meta["full_matching_sha256"]:
        raise ValueError("Compiled circuit hash differs")
    probability_key = f"{p:.3f}"
    model_path = circuit_path.with_name(f"p{probability_key}.dem")
    expected_model_hash = meta.get("dem_sha256", {}).get(probability_key)
    if not expected_model_hash or not model_path.is_file():
        raise ValueError(f"Missing strictly compiled DEM for p={probability_key}")
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != expected_model_hash:
        raise ValueError(f"Compiled DEM hash differs for p={probability_key}")
    if output.exists():
        old = json.loads(output.read_text(encoding="utf-8"))
        if (old.get("status") != "COMPLETE" or old.get("shots") != shots
                or old.get("seed") != seed or old.get("p") != p
                or old.get("circuit_sha256") != meta["full_matching_sha256"]
                or old.get("dem_sha256") != expected_model_hash
                or old.get("decoder") != "pymatching 2.3.1 correlated"):
            raise ValueError(f"Existing shard is incomplete or mismatched: {output}")
        return old
    start = time.monotonic()
    cpu_start = time.process_time()
    circuit = scaled_circuit(stim.Circuit.from_file(str(circuit_path)), p)
    dem = stim.DetectorErrorModel.from_file(str(model_path))
    if dem.num_detectors != circuit.num_detectors or dem.num_observables != circuit.num_observables:
        raise ValueError("Compiled DEM and sampled circuit have different dimensions")
    decoder = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
    sampler = circuit.compile_detector_sampler(seed=seed)
    mistakes, done = 0, 0
    while done < shots:
        syndromes, actual = sampler.sample(min(batch, shots - done), separate_observables=True)
        predicted = decoder.decode_batch(syndromes, enable_correlations=True)
        if predicted.shape != actual.shape:
            raise ValueError("Decoder observable shape mismatch")
        mistakes += int(np.any(predicted != actual, axis=1).sum())
        done += len(syndromes)
    try:
        import resource
        maxrss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:
        maxrss_kib = None
    result = {
        "status": "COMPLETE", "d": meta["d"], "W": meta["W"], "T": 3,
        "channel": meta["channel"], "boundary": "full", "noise": "SI1000",
        "p": p, "shots": done, "mistakes": mistakes, "seed": seed,
        "decoder": "pymatching 2.3.1 correlated", "stim": stim.__version__,
        "numpy": np.__version__, "circuit_sha256": meta["full_matching_sha256"],
        "dem_sha256": expected_model_hash,
        "elapsed_seconds": time.monotonic() - start,
        "cpu_seconds": time.process_time() - cpu_start,
        "maxrss_kib": maxrss_kib,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuit", type=Path, required=True)
    parser.add_argument("--compile-meta", type=Path, required=True)
    parser.add_argument("--p", type=float, required=True)
    parser.add_argument("--shots", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(sample(circuit_path=args.circuit, compile_meta=args.compile_meta,
                            p=args.p, shots=args.shots, seed=args.seed, output=args.out),
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

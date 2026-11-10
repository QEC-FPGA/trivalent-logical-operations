"""Exact <=4-signature enumeration, including hyperedges.

64-bit linear fingerprints are ONLY a lookup filter. Every hit (including every
member of a collision bucket) is checked against the full original bit vectors.
All original physical realizations are subsequently checked by materialize().
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import threading
import time
import uuid

import numpy as np

VERSION = "full-pair-index-v1"
MASK = (1 << 64) - 1
DTYPE = np.dtype([("h", "<u8"), ("pair", "<u8")])


def row_hash(row):
    x = (row + 0x9E3779B97F4A7C15) & MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK
    return (x ^ (x >> 31)) & MASK


def vectors(columns, target, hash_bits=64):
    salts = [row_hash(i) & ((1 << hash_bits) - 1) for i in range(target + 1)]
    sigs, hashes = [], []
    for col in columns:
        v = h = 0
        for r in col:
            v ^= 1 << r
            h ^= salts[r]
        sigs.append(v)
        hashes.append(h)
    if len(set(sigs)) != len(sigs) or 0 in sigs:
        raise ValueError("Require unique nonzero complete signatures")
    return sigs, np.array(hashes + [0], dtype=np.uint64), salts[target]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_index(columns, target, path, hash_bits=64):
    """Generate every two-distinct-column set and every singleton exactly once."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sigs, h, _ = vectors(columns, target, hash_bits)
    n = len(sigs)
    count = n * (n + 1) // 2
    if count * DTYPE.itemsize > 32 * 1024**3:
        raise MemoryError("Pair-index safety limit 32 GiB exceeded; no exclusion proof produced")
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    pairs = np.lib.format.open_memmap(temp, mode="w+", dtype=DTYPE, shape=(count,))
    pos = 0
    for i in range(n):
        js = np.arange(i + 1, n + 1, dtype=np.uint64)
        size = len(js)
        pairs["h"][pos:pos+size] = h[i] ^ h[i+1:]
        pairs["pair"][pos:pos+size] = np.uint64(i * (n + 1)) + js
        pos += size
    if pos != count:
        raise AssertionError("Incomplete pair enumeration")
    pairs.sort(order=["h", "pair"], kind="quicksort")
    pairs.flush()
    del pairs
    temp.replace(path)
    return dict(version=VERSION, columns=n, pairs=count, target=target,
                hash_bits=hash_bits, index_sha256=sha(path))


def enumerate_partition(columns, target, roots, start, stop, index_path,
                        materialize, out, seconds, workers, hash_bits=64):
    if not (0 <= start < stop <= len(roots)) or seconds <= 0 or workers < 1:
        raise ValueError("Invalid finite search request")
    sigs, hashes, target_hash = vectors(columns, target, hash_bits)
    if roots != [j for j, s in enumerate(sigs) if (s >> target) & 1]:
        raise ValueError("Wrong logical root partition")
    n = len(sigs)
    pairs = np.load(index_path, mmap_mode="r", allow_pickle=False)
    if pairs.dtype != DTYPE or len(pairs) != n * (n + 1) // 2:
        raise ValueError("Incomplete pair index")
    # Copy only this column ONCE. NumPy may otherwise materialize the strided
    # structured-array view on every searchsorted call (hundreds of MB/root).
    keys = np.ascontiguousarray(pairs["h"])
    keys.flags.writeable = False
    began = time.monotonic()
    deadline = began + seconds
    stop_event = threading.Event()
    desired = 1 << target
    full_index = {s: j for j, s in enumerate(sigs)}

    def one_root(root_index):
        root = roots[root_index]
        found = set()
        if sigs[root] == desired:
            found.add((root,))
        # Explicit singleton+singleton also makes the low-weight coverage clear.
        partner = full_index.get(desired ^ sigs[root])
        if partner is not None and partner != root:
            choice = tuple(sorted((root, partner)))
            if min(j for j in choice if sigs[j] & desired) == root:
                found.add(choice)
        if stop_event.is_set() or time.monotonic() >= deadline:
            return root_index, False, list(found)
        wanted = np.uint64(target_hash) ^ hashes[root] ^ hashes
        lo = np.searchsorted(keys, wanted, side="left")
        valid = lo < len(keys)
        hit = np.zeros(n + 1, dtype=bool)
        hit[valid] = keys[lo[valid]] == wanted[valid]
        for j in np.flatnonzero(hit):
            j = int(j)
            if j == root or (j < n and sigs[j] & desired and j < root):
                continue
            hi = int(np.searchsorted(keys, wanted[j], side="right"))
            # All collisions are inspected; never keep just the first hash hit.
            for at in range(int(lo[j]), hi):
                if stop_event.is_set() or time.monotonic() >= deadline:
                    return root_index, False, list(found)
                a, b = divmod(int(pairs["pair"][at]), n + 1)
                choice = [root] + ([] if j == n else [j]) + [a] + ([] if b == n else [b])
                if len(set(choice)) != len(choice):
                    continue
                if min(k for k in choice if sigs[k] & desired) != root:
                    continue
                value = 0
                for k in choice:
                    value ^= sigs[k]
                if value == desired:
                    found.add(tuple(sorted(choice)))
                if len(found) > 100000:
                    raise MemoryError("Candidate buffer limit; no exclusion proof produced")
        return root_index, True, sorted(found)

    done, rejected = [], []
    witness = None
    seen = set()
    with ThreadPoolExecutor(max_workers=min(workers, stop-start)) as pool:
        futures = [pool.submit(one_root, i) for i in range(start, stop)]
        try:
            for f in as_completed(futures):
                i, complete, candidates = f.result()
                if complete:
                    done.append(i)
                for selected in candidates:
                    if selected in seen:
                        continue
                    seen.add(selected)
                    witness = materialize(list(selected))
                    if witness is not None:
                        if witness["status"] != "UNDETECTED_LOGICAL" or witness["physical_weight"] > 4:
                            raise ValueError("Invalid physical replay")
                        stop_event.set()
                        break
                    rejected.append(list(selected))
                if witness is not None:
                    break
        finally:
            stop_event.set()
            for f in futures:
                f.cancel()
    status = ("PHYSICAL_WITNESS" if witness else
              "INFEASIBLE" if sorted(done) == list(range(start, stop)) else "UNKNOWN")
    result = dict(status=status, method=VERSION, cap=4,
                  elapsed_seconds=time.monotonic()-began, solver_workers=workers,
                  time_limit_seconds=seconds, pair_count=len(pairs),
                  enumerated_roots=sorted(done), required_roots=[start, stop],
                  rejected_physical_assignments=rejected, candidate_sets=len(seen),
                  physical_weight=witness["physical_weight"] if witness else None)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "enumeration.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result

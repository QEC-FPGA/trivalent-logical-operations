"""Complete bounded-weight XOR search with durable deterministic DFS stacks.

Branches are partitioned by the first selected column incident on a currently
odd row. Previously explored choices are excluded from subsequent siblings.
No hyperedge truncation, probabilistic rejection, or solver-search restart.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
import uuid

VERSION = "resumable-full-xor-dfs-v1"
_PROBLEM = None
_STOP = None
PAUSE = object()


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
        f.write("\n")
        f.flush()
        import os
        os.fsync(f.fileno())
    temporary.replace(path)


def bits(v):
    while v:
        b = v & -v
        yield b.bit_length()-1
        v ^= b


class Problem:
    def __init__(self, columns, target_row, roots, cap, choices):
        self.n = len(columns)
        self.target = 1 << target_row
        self.cap = cap
        self.roots = list(roots)
        if cap < 1 or not roots or roots != sorted(set(roots)):
            raise ValueError("Invalid cap/ordered logical roots")
        self.values = [sum(1 << r for r in c) for c in columns]
        if any(len(set(c)) != len(c) or any(r < 0 or r > target_row for r in c) for c in columns):
            raise ValueError("Invalid exact column support")
        if len(set(self.values)) != self.n or 0 in self.values:
            raise ValueError("Columns must be distinct and nonzero")
        if any(j < 0 or j >= self.n for j in roots):
            raise ValueError("Invalid root column index")
        self.lookup = {v:j for j,v in enumerate(self.values)}
        self.rows = [0]*(target_row+1)
        for j,c in enumerate(columns):
            for r in c:
                self.rows[r] |= 1 << j
        self.max_weight = max(v.bit_count() for v in self.values)
        self.choices = [tuple(sorted(set(a))) for a in choices]
        if len(choices) != self.n or any(not a for a in self.choices):
            raise ValueError("All signatures must retain their physical realizations")
        self.prefix = []
        value = 0
        for j in self.roots:
            value |= 1 << j
            self.prefix.append(value)
        self.fingerprint = digest(dict(columns=columns, target=target_row,
            roots=roots, cap=cap, physical_site_choices=self.choices))

    def physical(self, selected):
        # Exact bipartite matching: one distinct original physical noise site
        # for every selected response. Failure is monotone under supersets.
        match = {}
        def visit(j, used):
            for site in self.choices[j]:
                if site in used:
                    continue
                used.add(site)
                if site not in match or visit(match[site], used):
                    match[site] = j
                    return True
            return False
        return all(visit(j, set()) for j in sorted(selected, key=lambda j:len(self.choices[j])))

    def pivot(self, residual, forbidden, remaining):
        if residual.bit_count() > remaining*self.max_weight:
            return None
        rows = []
        for r in bits(residual):
            candidates = self.rows[r] & ~forbidden
            if not candidates:
                return None
            rows.append((candidates.bit_count(), r, candidates))
        rows.sort()
        # Disjoint incidence sets each require at least one different fault.
        covered = 0
        packing = 0
        for _,r,candidates in rows:
            if not (candidates & covered):
                covered |= candidates
                packing += 1
                if packing > remaining:
                    return None
        return rows[0][1], rows[0][2]

    def leaf(self, residual, forbidden, selected, remaining, should_stop):
        one = self.lookup.get(residual)
        if one is not None and not (forbidden >> one & 1):
            if self.physical(selected+[one]):
                return selected+[one]
        if remaining < 2:
            return None
        pivot = self.pivot(residual, forbidden, remaining)
        if pivot is None:
            return None
        for number,j in enumerate(bits(pivot[1])):
            if number % 64 == 0 and should_stop():
                return PAUSE
            k = self.lookup.get(residual ^ self.values[j])
            if k is None or j == k or forbidden >> k & 1:
                continue
            # Do NOT impose j<k: only one of the pair touches the odd pivot.
            if self.physical(selected+[j,k]):
                return selected+[j,k]
        return None


def identity(problem, task_key, start, stop, lanes, lane):
    return dict(version=VERSION, problem=problem.fingerprint, task=task_key,
                start=start, stop=stop, lanes=lanes, lane=lane)


def encode_state(state):
    data = {k:v for k,v in state.items() if k != "stack"}
    data["stack"] = [
        dict(selected=f["selected"], residual=hex(f["residual"]),
             forbidden=hex(f["forbidden"]), pivot=f["pivot"],
             todo=None if f["todo"] is None else hex(f["todo"]))
        for f in state["stack"]
    ]
    return data


def save_state(path, state):
    data = encode_state(state)
    atomic(path, dict(payload=data, sha256=digest(data)))


def load_state(path, expected, problem):
    path = Path(path)
    owned = list(range(expected["start"]+expected["lane"], expected["stop"], expected["lanes"]))
    if not path.exists():
        return dict(identity=expected, cursor=0, nodes=0, stack=[], witness=None,
                    status="CHECKPOINT", generations=0)
    wrapped = json.loads(path.read_text(encoding="utf-8"))
    data = wrapped["payload"]
    if wrapped["sha256"] != digest(data) or data["identity"] != expected:
        raise ValueError("Corrupt checkpoint or changed search identity: "+str(path))
    if not 0 <= data["cursor"] <= len(owned):
        raise ValueError("Invalid checkpoint root position")
    data["stack"] = [
        dict(selected=f["selected"], residual=int(f["residual"],16),
             forbidden=int(f["forbidden"],16), pivot=f["pivot"],
             todo=None if f["todo"] is None else int(f["todo"],16))
        for f in data["stack"]
    ]
    if data["cursor"] == len(owned) and data["stack"]:
        raise ValueError("Completed root cursor still has active stack")
    for f in data["stack"]:
        selected = f["selected"]
        if not selected or len(set(selected))!=len(selected) or len(selected)>problem.cap:
            raise ValueError("Invalid checkpoint selection")
        root_index = owned[data["cursor"]]
        if selected[0] != problem.roots[root_index]:
            raise ValueError("Checkpoint belongs to another logical root")
        if f["forbidden"] < 0 or f["forbidden"] >> problem.n:
            raise ValueError("Invalid excluded column mask")
        if f["forbidden"] & problem.prefix[root_index] != problem.prefix[root_index]:
            raise ValueError("Checkpoint omitted minimum-root restriction")
        residual = problem.target
        for j in selected:
            if not 0 <= j < problem.n or not (f["forbidden"] >> j & 1):
                raise ValueError("Invalid selected/excluded column")
            residual ^= problem.values[j]
        if residual != f["residual"]:
            raise ValueError("Checkpoint residual disagrees with full syndrome")
        if f["todo"] is not None:
            if f["pivot"] is None or not (residual >> f["pivot"] & 1):
                raise ValueError("Checkpoint pivot is not odd")
            if f["todo"] != problem.rows[f["pivot"]] & ~f["forbidden"]:
                raise ValueError("Checkpoint sibling coverage mismatch")
    if data["status"] == "INFEASIBLE" and (
        data["cursor"] != len(owned) or data["stack"] or data["witness"] is not None
    ):
        raise ValueError("Incomplete checkpoint cannot certify exclusion")
    return data


def frame(selected, residual, forbidden):
    return dict(selected=selected,residual=residual,forbidden=forbidden,pivot=None,todo=None)


def run_lane(problem, checkpoint, task_key, start, stop, lanes, lane,
             deadline, stop_event=None, checkpoint_seconds=15, node_budget=None):
    expected = identity(problem,task_key,start,stop,lanes,lane)
    owned = list(range(start+lane,stop,lanes))
    state = load_state(checkpoint,expected,problem)
    nodes_before = state["nodes"]
    state["generations"] += 1
    began = last_saved = time.monotonic()
    def stopping():
        return (time.monotonic() >= deadline or
                (stop_event is not None and stop_event.is_set()))
    def pop():
        state["stack"].pop()
        if not state["stack"]:
            state["cursor"] += 1
    loops = 0
    while state["status"] != "INFEASIBLE" and state["witness"] is None:
        if loops % 64 == 0:
            if stopping():
                break
            now = time.monotonic()
            if now-last_saved >= checkpoint_seconds:
                save_state(checkpoint,state)
                last_saved = now
        loops += 1
        if not state["stack"]:
            if state["cursor"] == len(owned):
                state["status"] = "INFEASIBLE"
                break
            root_index = owned[state["cursor"]]
            j = problem.roots[root_index]
            state["stack"].append(frame([j],problem.target ^ problem.values[j],problem.prefix[root_index]))
        f = state["stack"][-1]
        remaining = problem.cap-len(f["selected"])
        if f["todo"] is None:
            # Node-budget hook only used by deterministic resume regression tests.
            if node_budget is not None and state["nodes"]-nodes_before >= node_budget:
                break
            state["nodes"] += 1
            if not problem.physical(f["selected"]):
                pop()
                continue
            if f["residual"] == 0:
                state["witness"] = list(f["selected"])
                break
            if remaining == 0:
                pop()
                continue
            if remaining <= 2:
                # Keep this unexpanded leaf if interrupted, so nothing is skipped.
                found = problem.leaf(f["residual"],f["forbidden"],f["selected"],remaining,stopping)
                if found is PAUSE:
                    break
                if found is not None:
                    state["witness"] = found
                    break
                pop()
                continue
            pivot = problem.pivot(f["residual"],f["forbidden"],remaining)
            if pivot is None:
                pop()
                continue
            f["pivot"],f["todo"] = pivot
        if not f["todo"]:
            pop()
            continue
        chosen_bit = f["todo"] & -f["todo"]
        j = chosen_bit.bit_length()-1
        f["todo"] ^= chosen_bit
        f["forbidden"] |= chosen_bit
        state["stack"].append(frame(f["selected"]+[j],
            f["residual"] ^ problem.values[j],f["forbidden"]))
    if state["witness"] is not None:
        state["status"] = "PHYSICAL_CANDIDATE"
    save_state(checkpoint,state)
    return dict(lane=lane,status=state["status"],root_count=len(owned),
                roots_completed=state["cursor"],nodes_total=state["nodes"],
                nodes_this_attempt=state["nodes"]-nodes_before,
                resumed=state["generations"]>1,selected=state["witness"],
                seconds=time.monotonic()-began)


def _init(problem, stop):
    global _PROBLEM, _STOP
    _PROBLEM, _STOP = problem, stop


def _run(args):
    return run_lane(_PROBLEM,*args,stop_event=_STOP)


def physical_choices(npz_path, column_count):
    import numpy as np
    data = np.load(npz_path,allow_pickle=False)
    choices = [set() for _ in range(column_count)]
    for column,site in zip(data["branch_columns"],data["branch_sites"]):
        column = int(column)
        if column >= 0:
            choices[column].add(int(site))
    return [tuple(sorted(c)) for c in choices]


def solve(columns,target,roots,cap,start,stop,choices,checkpoint_dir,task_key,
          out,materialize,seconds,workers):
    if not 0 <= start < stop <= len(roots) or seconds <= 0 or workers < 1:
        raise ValueError("Invalid partition/resource limit")
    began = time.monotonic()
    problem = Problem(columns,target,roots,cap,choices)
    checkpoint_dir,out = Path(checkpoint_dir),Path(out)
    checkpoint_dir.mkdir(parents=True,exist_ok=True)
    out.mkdir(parents=True,exist_ok=True)
    lanes = min(48,stop-start)  # Stable even when process allocation changes.
    deadline = time.monotonic()+seconds
    results = []
    witness = None
    if workers == 1:
        for lane in range(lanes):
            results.append(run_lane(problem,checkpoint_dir/f"lane_{lane:02d}.json",
                task_key,start,stop,lanes,lane,deadline))
            if results[-1]["selected"] is not None:
                witness = materialize(results[-1]["selected"])
                if witness is None:
                    raise ValueError("Exact site matching disagrees with physical materialization")
                break
    else:
        ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
        stop_event = ctx.Event()
        with ProcessPoolExecutor(max_workers=min(workers,lanes),mp_context=ctx,
                                 initializer=_init,initargs=(problem,stop_event)) as pool:
            futures = [pool.submit(_run,(
                checkpoint_dir/f"lane_{lane:02d}.json",task_key,start,stop,lanes,lane,deadline
            )) for lane in range(lanes)]
            try:
                for f in as_completed(futures):
                    r = f.result()
                    results.append(r)
                    if witness is None and r["selected"] is not None:
                        witness = materialize(r["selected"])
                        if witness is None:
                            raise ValueError("Exact site matching disagrees with physical materialization")
                        stop_event.set()
            finally:
                stop_event.set()
    completed = 0
    snapshots = []
    for lane in range(lanes):
        p = checkpoint_dir/f"lane_{lane:02d}.json"
        if not p.exists():
            continue
        state = load_state(p,identity(problem,task_key,start,stop,lanes,lane),problem)
        if state["status"]=="INFEASIBLE":
            completed += 1
        # Immutable snapshot belongs to this attempt's audited evidence.
        target_path = out/f"lane_{lane:02d}.json"
        target_path.write_bytes(p.read_bytes())
        snapshots.append(target_path.name)
    if witness is not None:
        if witness["status"]!="UNDETECTED_LOGICAL" or witness["physical_weight"]>cap:
            raise ValueError("Invalid original-circuit witness")
        status = "PHYSICAL_WITNESS"
    else:
        status = "INFEASIBLE" if completed==lanes else "CHECKPOINT"
    result = dict(status=status,method=VERSION,cap=cap,root_interval=[start,stop],
        lanes=lanes,lanes_complete=completed,problem_sha256=problem.fingerprint,
        elapsed_seconds=time.monotonic()-began,solver_workers=workers,
        time_limit_seconds=seconds,lane_results=results,checkpoint_snapshots=snapshots,
        physical_weight=None if witness is None else witness["physical_weight"])
    atomic(out/"enumeration.json",result)
    return result

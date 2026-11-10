"""Complete, disjoint minimum-logical-column partitions; no graph truncation."""
import gzip
import time
from ortools.sat.python import cp_model
from exact_search import build_model


def intervals(count, requested):
    if count < 0 or requested < 1:
        raise ValueError('Invalid partition size')
    n = min(count, requested)
    return [(count*i//n, count*(i+1)//n) for i in range(n)]


def validate_intervals(count, spans):
    end = 0
    for start, stop in spans:
        if start != end or not start < stop <= count:
            raise ValueError('Gap, overlap, duplicate or invalid root interval')
        end = stop
    if end != count:
        raise ValueError('Incomplete root coverage')


def partition_model(columns, detector_count, cap, roots, start, stop):
    # roots are ORIGINAL logical-odd column indices. Row reduction must not
    # redefine this list. Column order is preserved by the verified reducer.
    if not 0 <= start < stop <= len(roots):
        raise ValueError('Invalid root interval')
    if len(set(roots)) != len(roots) or roots != sorted(roots):
        raise ValueError('Root indices must be unique and sorted')
    model, variables = build_model(columns, detector_count, cap)
    for j in roots[:start]:
        model.add(variables[j] == 0)
    model.add(sum(variables[j] for j in roots[start:stop]) >= 1)
    return model, variables


def solve_partition(columns, detector_count, cap, roots, start, stop,
                    materialize, out, seconds, workers, seed=579):
    if seconds <= 0 or workers < 1:
        raise ValueError('Finite positive time and thread count required')
    out.mkdir(parents=True, exist_ok=True)
    model, variables = partition_model(columns, detector_count, cap, roots, start, stop)
    def export(name):
        with gzip.open(out/name, 'wt', encoding='utf-8') as f:
            f.write(str(model.proto))
    export('initial_model.pbtxt.gz')
    began = time.monotonic()
    cuts = []
    iterations = []
    witness = None
    status = 'UNKNOWN'
    with (out/'solver.log').open('w', encoding='utf-8') as log:
        while time.monotonic()-began < seconds:
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = max(.001, seconds-(time.monotonic()-began))
            solver.parameters.num_search_workers = workers
            solver.parameters.random_seed = seed
            solver.parameters.log_search_progress = True
            solver.parameters.log_to_stdout = False
            def callback(s):
                log.write(s+'\n')
                log.flush()
            solver.log_callback = callback
            raw = solver.solve(model)
            iterations.append(dict(status=solver.status_name(raw), response=solver.response_stats()))
            if raw == cp_model.INFEASIBLE:
                status = 'INFEASIBLE'
                break
            if raw not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                status = 'UNKNOWN' if raw == cp_model.UNKNOWN else 'MODEL_INVALID'
                break
            selected = [j for j,v in enumerate(variables) if solver.value(v)]
            chosen_roots = [i for i,j in enumerate(roots) if j in set(selected)]
            if not chosen_roots or not start <= min(chosen_roots) < stop:
                raise ValueError('Solver result violates unique partition attribution')
            witness = materialize(selected)
            if witness is not None:
                if witness['status'] != 'UNDETECTED_LOGICAL' or witness['physical_weight'] > cap:
                    raise ValueError('Invalid physical witness')
                status = 'PHYSICAL_WITNESS'
                iterations[-1]['selected_columns'] = selected
                break
            # Exact site matching failed: no physical solution can contain
            # this unassignable set. Supersets cannot repair site exclusivity.
            cuts.append(selected)
            model.add(sum(variables[j] for j in selected) <= len(selected)-1)
    export('final_model.pbtxt.gz')
    return dict(status=status, elapsed_seconds=time.monotonic()-began,
                solver_workers=workers, time_limit_seconds=seconds, seed=seed,
                cuts=cuts, iterations=iterations,
                physical_weight=None if witness is None else witness['physical_weight'])

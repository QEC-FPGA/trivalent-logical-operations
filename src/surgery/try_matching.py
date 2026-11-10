"""Strict correlated-PyMatching preparation using the archived S repair."""
from pathlib import Path
import argparse, json, time
import stim
from circuit_builder import bits, indices, add_basis, sha
from matching_repair import repair, FLAGS

def extract(c):
    rows=[]; total=0
    for op in c.flattened():
        if op.name=='DETECTOR':rows.append(bits(total+t.value for t in op.targets_copy()))
        one=stim.Circuit();one.append(op);total+=one.num_measurements
    return rows,total

def annotate(c,rows):
    out=stim.Circuit()
    for op in c.flattened():
        if op.name!='DETECTOR':out.append(op)
    total=out.num_measurements
    for row in rows:out.append('DETECTOR',[stim.target_rec(k-total) for k in indices(row)])
    return out

def strict_dem(c):
    try:c.detector_error_model(**FLAGS);return True,''
    except ValueError as e:return False,str(e)

def physical(c):
    out=stim.Circuit()
    for op in c.flattened():
        if op.name!='DETECTOR':out.append(op)
    return out

def attempt(c,out,seconds):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    def callback(candidate,operations,event):
        print(json.dumps(event),flush=True)
        (out/'checkpoint.stim').write_text(str(candidate),encoding='utf-8')
    candidate,report=repair(c,extract=extract,annotate=annotate,strict_dem=strict_dem,seconds=seconds,callback=callback)
    report.update(input_sha256=sha(c),strict_flags=FLAGS)
    if candidate is not None:
        old,_=extract(c);new,_=extract(candidate);computed=list(old)
        for i,j in report['operations']:computed[i]^=computed[j]
        assert computed==new
        assert physical(c)==physical(candidate)
        b={}
        assert all(add_basis(b,v) for v in old)
        assert len(new)==len(old)
        for row in new:
            v=row
            while v and v.bit_length()-1 in b:v^=b[v.bit_length()-1]
            assert v==0
        b2={};assert all(add_basis(b2,v) for v in new)
        (out/'circuit.stim').write_text(str(candidate),encoding='utf-8')
        dem=candidate.detector_error_model(**FLAGS);(out/'matching.dem').write_text(str(dem),encoding='utf-8')
        report.update(output_sha256=sha(candidate),dem_sha256=sha(dem),rank=len(new),row_space_identical=True,physical_and_observables_identical=True)
    (out/'repair_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(report['status'],flush=True)
    return candidate,report

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('circuit');a.add_argument('--out',required=True);a.add_argument('--seconds',type=float,default=120)
    args=a.parse_args();attempt(stim.Circuit(Path(args.circuit).read_text(encoding='utf-8')),args.out,args.seconds)

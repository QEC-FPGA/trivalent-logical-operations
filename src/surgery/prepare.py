"""Explicitly compile generated cases for the pinned correlated decoder."""
from pathlib import Path
import argparse,json,time
import stim
from generate import generate,sha,front_coordinates
from try_matching import attempt,extract,annotate
from matching_repair import FLAGS

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--distances',type=int,nargs='+',default=[5])
    a.add_argument('--measurements',nargs='+',choices=['ZZ','XX','XZ'],default=['ZZ'])
    a.add_argument('--p',type=float,nargs='+',default=[.001,.002])
    a.add_argument('--out',type=Path,default=Path('frozen'))
    a.add_argument('--repair-seconds',type=float,default=0)
    args=a.parse_args();out=args.out
    if (out/'index.json').exists():raise FileExistsError('This frozen output already exists')
    index=[]
    for d in args.distances:
      for axis in args.measurements:
        folder=out/'generated'/f'{axis}_d{d}'
        ideal,c,meta=generate(d,.001,axis,folder)
        fixed,report=attempt(c,out/'compilation'/f'{axis}_d{d}',args.repair_seconds)
        if fixed is None:raise RuntimeError('Strict decomposition not accepted; sampling blocked. No faults dropped.')
        rows,_=extract(fixed);ideal=front_coordinates(annotate(ideal,rows))
        for p in args.p:
            if not 0<p<.1:raise ValueError('Positive physical error probability required')
            scaled=stim.Circuit()
            for op in fixed:
                if op.name in ['DEPOLARIZE1','DEPOLARIZE2','X_ERROR','Y_ERROR','Z_ERROR']:
                    scaled.append(op.name,op.targets_copy(),[v*p/.001 for v in op.gate_args_copy()])
                else:scaled.append(op)
            scaled=front_coordinates(scaled);dem=scaled.detector_error_model(**FLAGS)
            name=f'{axis}_d{d}_p{p:.6f}';dest=out/name;dest.mkdir(parents=True,exist_ok=False)
            (dest/'circuit.stim').write_text(str(scaled),encoding='utf-8')
            (dest/'ideal.stim').write_text(str(ideal),encoding='utf-8')
            (dest/'matching.dem').write_text(str(dem),encoding='utf-8')
            m=dict(meta,p=p,case=name,circuit_sha256=sha(scaled),ideal_sha256=sha(ideal),dem_sha256=sha(dem),strict_dem_flags=FLAGS,
                   decoder='pymatching correlated 2.3.1',formal_sampling_started=False)
            (dest/'metadata.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
            index.append({k:m[k] for k in ['case','d','measurement','p','circuit_sha256','dem_sha256']})
    (out/'index.json').write_text(json.dumps(index,indent=2),encoding='utf-8')
    print(f'{len(index)} cases prepared; no sampling started')
if __name__=='__main__':main()

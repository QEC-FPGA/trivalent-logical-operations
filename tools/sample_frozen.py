"""Sample a frozen circuit with correlated PyMatching; write a new independent run."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import stim, pymatching

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--circuit',type=Path,required=True);p.add_argument('--dem',type=Path,required=True)
    p.add_argument('--shots',type=int,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--batch-size',type=int,default=2048);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.shots<=0 or a.batch_size<=0:p.error('shots and batch-size must be positive')
    if a.out.exists():p.error('output already exists')
    if stim.__version__!='1.16.0' or pymatching.__version__!='2.3.1':p.error('requires Stim 1.16.0 and PyMatching 2.3.1')
    c=stim.Circuit.from_file(a.circuit);dem=stim.DetectorErrorModel.from_file(a.dem)
    if c.num_observables!=dem.num_observables or c.num_detectors!=dem.num_detectors:p.error('circuit/DEM dimensions differ')
    matching=pymatching.Matching.from_detector_error_model(dem,enable_correlations=True)
    sampler=c.compile_detector_sampler(seed=a.seed);errors=0;hist=np.zeros(2**c.num_observables,dtype=np.int64)
    for start in range(0,a.shots,a.batch_size):
        ds,obs=sampler.sample(min(a.batch_size,a.shots-start),separate_observables=True)
        residual=matching.decode_batch(ds,enable_correlations=True)^obs
        errors+=int(np.any(residual,axis=1).sum())
        codes=residual.astype(np.int64)@(1<<np.arange(c.num_observables))
        hist+=np.bincount(codes,minlength=len(hist))
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    out=dict(shots=a.shots,errors=errors,rate=errors/a.shots,seed=a.seed,batch_size=a.batch_size,
        circuit_sha256=digest(a.circuit),dem_sha256=digest(a.dem),residual_histogram=hist.tolist(),
        decoder='pymatching_correlated',stim=stim.__version__,pymatching=pymatching.__version__,numpy=np.__version__,
        definition='Any logical observable fails in the whole circuit; no per-round normalization')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()

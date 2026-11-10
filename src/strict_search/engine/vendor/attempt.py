"""Equivalent detector bases and strict graphlike-decomposition attempts at d=7,9.
No physical gates, noise channels, reference boundaries or observables are changed.
"""
from pathlib import Path
import sys,json,time,hashlib,collections
import stim
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'lib' if (HERE/'lib').exists() else HERE.parent/'S_gate_submission_20261001/frozen'))
from reverse_faults import nmeas,FaultIndex
from fault_checks import NOISE,signature

def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def bits(v):
 while v:
  b=v&-v;yield b.bit_length()-1;v^=b
def basis(rows):
 p={}
 for j,v in enumerate(rows):
  mask=1<<j
  while v:
   k=v.bit_length()-1
   if k not in p:p[k]=(v,mask);break
   w,a=p[k];v^=w;mask^=a
 return p
def solve(v,p):
 mask=0
 while v:
  k=v.bit_length()-1
  if k not in p:return None
  w,a=p[k];v^=w;mask^=a
 return mask
def extract(c):
 rows=[];meas=[];nm=tick=0;xy=c.get_final_qubit_coordinates()
 for op in c:
  if op.name=='TICK':tick+=1
  if op.name=='DETECTOR':rows.append(sum(1<<(nm+t.value) for t in op.targets_copy()))
  if op.name in ('M','MX','MY'):
   for t in op.targets_copy():meas.append(dict(record=len(meas),tick=tick,qubit=t.value,coordinate=xy.get(t.value),basis=op.name))
  elif op.name=='MPP':
   for _ in range(nmeas(op)):meas.append(dict(record=len(meas),tick=tick,basis='MPP'))
  nm+=nmeas(op)
 return rows,meas
def annotate(c,rows):
 out=stim.Circuit()
 for op in c:
  if op.name!='DETECTOR':out.append(op)
 for row in rows:out.append('DETECTOR',[stim.target_rec(i-c.num_measurements) for i in bits(row)])
 return out
def probe(c,dest):
 start=time.monotonic();dem=c.detector_error_model(decompose_errors=False);hist=collections.Counter()
 for op in dem:
  if op.type=='error':hist[sum(t.is_relative_detector_id() for t in op.targets_copy())]+=1
 result=dict(detectors=c.num_detectors,degree_histogram=dict(sorted(hist.items())),max_degree=max(hist),decompositions=[])
 for block in (True,False):
  try:
   dec=c.detector_error_model(decompose_errors=True,ignore_decomposition_failures=False,block_decomposition_from_introducing_remnant_edges=block)
   dec.to_file(str(dest/f'decomposed_remnants{int(not block)}.dem'))
   result['decompositions'].append(dict(block_remnants=block,success=True))
  except Exception as e:result['decompositions'].append(dict(block_remnants=block,success=False,error=str(e)))
 result['seconds']=time.monotonic()-start;dump(dest/'probe.json',result)
 return result

def main():
 start=time.monotonic()
 for d,ch in cases():
  source=HERE/f'd{d}_{ch}'/'original.stim'
  c=stim.Circuit.from_file(str(source));dest=HERE/f'd{d}_{ch}';dest.mkdir(parents=True,exist_ok=True);c.to_file(str(dest/'original.stim'))
  original,meas=extract(c);p=basis(original);assert len(p)==c.num_detectors
  candidates={r for r in original};prefix=stim.Circuit();accepted=excluded=0;cuts=0
  for op in c:
   if op.name in NOISE or op.name in ('DETECTOR','OBSERVABLE_INCLUDE'):continue
   prefix.append(op)
   if not nmeas(op):continue
   cuts+=1
   for f in prefix.flow_generators():
    if any(f.input_copy()) or any(f.output_copy()):continue
    mask=sum(1<<i for i in f.measurements_copy())
    if mask and solve(mask,p) is not None:candidates.add(mask);accepted+=1
    else:excluded+=1
  def score(v):
   ids=list(bits(v));ts=[meas[i]['tick'] for i in ids]
   xy=[meas[i].get('coordinate') for i in ids];xy=[q for q in xy if q]
   spread=(max(q[0] for q in xy)-min(q[0] for q in xy)+max(q[1] for q in xy)-min(q[1] for q in xy)) if xy else 0
   return (v.bit_count(),max(ts)-min(ts),spread,max(ts),v)
  selected=[];piv={}
  for v in sorted(candidates,key=score):
   w=v
   while w:
    k=w.bit_length()-1
    if k not in piv:piv[k]=w;selected.append(v);break
    w^=piv[k]
  assert len(selected)==len(original)
  transform=[solve(v,p) for v in selected];assert len(basis(transform))==len(original)
  nc=annotate(c,selected);nc.to_file(str(dest/'prefix_local.stim'));result=probe(nc,dest)
  dump(dest/'basis.json',dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),cuts=cuts,accepted_flows=accepted,excluded_flows=excluded,candidates=len(candidates),rank=len(selected),
   old_records=[list(bits(x)) for x in original],new_records=[list(bits(x)) for x in selected],new_from_old=[list(bits(x)) for x in transform],measurements=meas))
  print(d,ch,'cuts',cuts,'candidates',len(candidates),'degree',result['max_degree'],'decomposition',result['decompositions'],flush=True)
 dump(HERE/'time.json',dict(seconds=time.monotonic()-start,full_distance_search=False,performance_sampling=False))
def cases():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--d',type=int,choices=[5,7,9]);ap.add_argument('--channel',choices=['XY','ZZ']);a=ap.parse_args()
 return [(d,ch) for d in ([a.d] if a.d else [5,7,9]) for ch in ([a.channel] if a.channel else ['XY','ZZ'])]
if __name__=='__main__':main()

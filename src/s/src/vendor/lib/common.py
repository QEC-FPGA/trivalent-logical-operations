from pathlib import Path
import sys,json,time,hashlib
import stim
HERE=Path(__file__).resolve().parent
WORK=HERE.parents[1]
PREVIOUS=WORK/'outputs/S_gate_distance_evidence_20260930'
OLD=WORK/'outputs/high_d_w_analysis_20260929'
if False: # This package always uses its pinned, local frozen source.
 sys.path.insert(0,str(PREVIOUS))
 import audit
 from audit import moving,base,noise_sites,propagate,injected,unpack
 from reverse_faults import NOISE,nmeas
 from analyze import rows_from
else:
 # Standalone delivery: same pinned generator and physical-fault checker.
 from types import SimpleNamespace
 sys.path[:0]=[str(HERE/'frozen'),str(HERE/'frozen/native')]
 import generate_moving as moving
 from fault_checks import noise_sites,propagate,injected,unpack
 from reverse_faults import NOISE,nmeas
 base=moving.base
 original_complete=base.complete_detectors
 base.complete_detectors=lambda physical,obs,dual,prefixes=None:original_complete(physical,obs,dual,None)
 audit=SimpleNamespace(SCAN=HERE/'frozen')
 def rows_from(b,ch):
  tick=2 if ch=='YX' else 0;out=[]
  for name,c in b.segments.items():
   if any(op.name in ('CX','CZ') for op in c):out.append(dict(name=name,tick=tick))
   tick+=c.num_ticks
  return out
def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def wc(d):return 2*((d+6)//6)+1
def source(d,w,ch='YX'):
 p=HERE/f'circuits/d{d}_w{w}_{ch}.stim'
 if p.exists():return p
 p=PREVIOUS/f'd{d}_w{w}_{ch}/d{d}_{ch}_SI1000_body.stim'
 if p.exists():return p
 p=audit.SCAN/f'cases/d{d}_w{w}/d{d}_{ch}_SI1000_body.stim'
 if p.exists():return p
 return HERE/f'cases/d{d}_w{w}_{ch}/d{d}_{ch}_SI1000_body.stim'
class Budget:
 def __init__(self,name):
  self.name=name;self.start=time.monotonic();self.p=HERE/'computation.json'
  self.old=json.loads(self.p.read_text()) if self.p.exists() else []
  assert sum(x['seconds'] for x in self.old)<1800,'30 minute new-computation budget exhausted'
 def remaining(self):return 1800-sum(x['seconds'] for x in self.old)-(time.monotonic()-self.start)
 def save(self):
  elapsed=time.monotonic()-self.start
  self.old.append(dict(task=self.name,seconds=elapsed));dump(self.p,self.old)
  print('BUDGET',self.name,round(elapsed,2),'total',round(sum(x['seconds'] for x in self.old),2),flush=True)
def abstract(d,w,weight=None):
 weight=d-1 if weight is None else weight
 a=3*weight-2*d-2*w+3;n=2*(d+w-weight-3);delta=a-n//2
 if a<1 or n<0:return None
 e=[(j,2,[(1.5,d-.5+delta-j),(2,d-1+delta-j)],'XI','X leg') for j in range(a)]
 e += [(a,2,[(1,d-a-1+delta),(.5,d-a-.5+delta)],'IZ','turn Z'),(a,4,[(1.5,d-a-.5+delta),(2,d-a+delta)],'IY','turn Y')]
 e += [(a+j,3,[(2+j/2,d-a+j/2+delta),(2.5+j/2,d-a+.5+j/2+delta)],'YI','Y leg') for j in range(1,n+1)]
 e += [(a+n+1,4,[(2+n/2,d),(2.5+n/2,d+.5)],'ZX','closure')]
 return e
def map_events(c,d,w,ch='YX',weight=None):
 spec=abstract(d,w,weight)
 if spec is None:return [],['OUTSIDE_FAMILY_DOMAIN']
 rows=rows_from(moving.builder(d,w,3),ch);first=next(i for i,r in enumerate(rows) if r['name'].startswith('wide_'))
 xy=c.get_final_qubit_coordinates();inv={tuple(v):q for q,v in xy.items()}
 sites={(t,tuple(qs)):(i,qs) for i,t,op,qs in noise_sites(c) if op.name=='DEPOLARIZE2'}
 out=[];missing=[]
 for k,(block,t,points,ps,part) in enumerate(spec):
  qs=tuple(inv.get(tuple(x)) for x in points);row=rows[first+block];tick=row['tick']+t;hit=sites.get((tick,qs))
  if hit is None:missing.append(k);continue
  i,qs=hit;out.append(dict(instruction_offset=i,tick=tick,noise_targets=list(qs),paulis={str(q):p for q,p in zip(qs,ps) if p!='I'},block=block,local_tick=t,part=part,stage=row['name'],gate_coordinates=points,pauli_branch=ps))
 return out,missing

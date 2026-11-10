"""Exact backward linear-response enumeration of the given full Pauli circuit.
Validation is against independent Stim undecomposed DEM and forward frame replay.
A Pauli branch on one DEPOLARIZE2 pair counts as ONE physical fault.
"""
from pathlib import Path
from collections import Counter
import hashlib,json,random,sys,time
import stim
SRC=Path(__file__).resolve().parent/'sources'/'counterexample'
sys.path.insert(0,str(SRC))
from fault_checks import signature,propagate,injected,unpack,mpp_products
NOISE={'X_ERROR','Y_ERROR','Z_ERROR','DEPOLARIZE1','DEPOLARIZE2'}
PAULI='IXYZ'

def nmeas(op):
 if op.name in ('M','MX','MY'):return len(op.targets_copy())
 if op.name=='MPP':
  ts=op.targets_copy();return sum(t.is_x_target or t.is_y_target or t.is_z_target for t in ts)-sum(t.is_combiner for t in ts)
 return 0

class FaultIndex:
 def __init__(self,c):
  assert c.num_observables==1
  self.c=c;self.ops=list(c);self.ticks=[];t=0
  self.D=c.num_detectors;self.target=1<<self.D
  self.records=[0]*c.num_measurements
  m=0;d=0
  for op in self.ops:
   self.ticks.append(t)
   if op.name=='TICK':t+=1
   if op.name in ('DETECTOR','OBSERVABLE_INCLUDE'):
    if op.name=='OBSERVABLE_INCLUDE':assert op.gate_args_copy()==[0]
    val=1<<(d if op.name=='DETECTOR' else self.D)
    for r in op.targets_copy():
     assert r.is_measurement_record_target
     self.records[m+r.value]^=val
    if op.name=='DETECTOR':d+=1
   else:m+=nmeas(op)
  assert m==len(self.records) and d==self.D
  self.reps={};self.branch_counts=Counter();self.sites=0

 def enumerate(self):
  x=[0]*self.c.num_qubits;z=x.copy();m=self.c.num_measurements
  for i in range(len(self.ops)-1,-1,-1):
   op=self.ops[i];nm=op.name;ts=op.targets_copy()
   if nm in ('TICK','QUBIT_COORDS','DETECTOR','OBSERVABLE_INCLUDE'):continue
   if nm=='MPP':
    products=mpp_products(op);m-=len(products)
    for j,product in enumerate(products):
     v=self.records[m+j]
     for t in product:
      if t.is_z_target or t.is_y_target:x[t.value]^=v
      if t.is_x_target or t.is_y_target:z[t.value]^=v
    continue
   qs=[t.value for t in ts]
   if nm in NOISE:
    step=2 if nm=='DEPOLARIZE2' else 1
    for j in range(0,len(qs),step):
     self.sites+=1;a=qs[j]
     pa=(0,x[a],x[a]^z[a],z[a])
     if step==2:
      b=qs[j+1];pb=(0,x[b],x[b]^z[b],z[b])
      for p1 in range(4):
       for p2 in range(4):
        if p1==p2==0:continue
        sig=pa[p1]^pb[p2];code=(i<<32)|(j<<4)|(p1*4+p2)
        self.reps.setdefault(sig,code)
        self.branch_counts[nm]+=1
     else:
      opts=(1,2,3) if nm=='DEPOLARIZE1' else (PAULI.index(nm[0]),)
      for p in opts:
       code=(i<<32)|(j<<4)|p;self.reps.setdefault(pa[p],code);self.branch_counts[nm]+=1
    continue
   if nm=='R':
    for q in qs:x[q]=z[q]=0
   elif nm=='H':
    for q in qs:x[q],z[q]=z[q],x[q]
   elif nm in ('S','S_DAG'):
    for q in qs:x[q]^=z[q]
   elif nm=='CX':
    for a,b in zip(qs[::2],qs[1::2]):x[a]^=x[b];z[b]^=z[a]
   elif nm=='CZ':
    for a,b in zip(qs[::2],qs[1::2]):x[a]^=z[b];x[b]^=z[a]
   elif nm in ('M','MX','MY'):
    m-=len(qs)
    for j,q in enumerate(qs):
     v=self.records[m+j]
     if nm in ('M','MY'):x[q]^=v
     if nm in ('MX','MY'):z[q]^=v
   else:raise ValueError(nm)
  assert m==0
  return self

 def event(self,code):
  i=code>>32;j=(code&0xffffffff)>>4;branch=code&15
  op=self.ops[i];qs=[t.value for t in op.targets_copy()]
  if op.name=='DEPOLARIZE2':
   q=qs[j:j+2];p=[PAULI[branch//4],PAULI[branch%4]]
  else:q=qs[j:j+1];p=[PAULI[branch]]
  return dict(instruction_offset=i,tick=self.ticks[i],noise_targets=q,noise_kind=op.name,paulis={str(a):b for a,b in zip(q,p) if b!='I'})

 def validate(self,seed=281):
  start=time.time();dem=self.c.detector_error_model(decompose_errors=False)
  ds={signature(op,self.D) for op in dem if op.type=='error'}
  rs=set(self.reps)-{0};assert ds==rs,('DEM mismatch',len(ds),len(rs),len(ds-rs),len(rs-ds))
  keys=list(self.reps);chosen=random.Random(seed).sample(keys,min(96,len(keys)))
  events=[self.event(self.reps[k]) for k in chosen]
  got=propagate(self.c,events);assert got==chosen
  # Three individual physical injections validate locations and independent response.
  for k in chosen[:3]:assert injected(self.c,[self.event(self.reps[k])])[0]==k
  return dict(physical_branches=sum(self.branch_counts.values()),branch_counts=dict(self.branch_counts),physical_sites=self.sites,nonzero_signatures=len(rs),full_undecomposed_DEM_signature_set_equal=True,forward_frame_columns_checked=len(chosen),individual_Stim_E1_checks=3,validation_seconds=time.time()-start)

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('circuit');ap.add_argument('--out',required=True);a=ap.parse_args();t=time.time();c=stim.Circuit.from_file(a.circuit)
 index=FaultIndex(c).enumerate();print('ENUM',time.time()-t,sum(index.branch_counts.values()),len(index.reps),flush=True)
 result=index.validate();result.update(circuit_sha256=hashlib.sha256(Path(a.circuit).read_bytes()).hexdigest(),total_seconds=time.time()-t)
 Path(a.out).write_text(json.dumps(result,indent=2));print(result,flush=True)

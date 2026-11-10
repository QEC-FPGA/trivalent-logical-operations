"""Generate the author-repaired MZZ circuit; no distance search or sampling."""
from pathlib import Path
import argparse,json,hashlib
import stim
from seam_repair import entry_detectors
ROOT=Path(__file__).resolve().parent
def sha(c):return hashlib.sha256(str(c).encode()).hexdigest()
def strip_coords(c):
    out=stim.Circuit()
    for op in c:
        if op.name!='QUBIT_COORDS':out.append(op)
    return out
def front_coordinates(c):
    coords=c.get_final_qubit_coordinates();out=stim.Circuit()
    missing=sorted(set(range(c.num_qubits))-set(coords))
    for i,q in enumerate(missing):coords[q]=[-4,4+12*i]
    for q,pos in sorted(coords.items()):out.append('QUBIT_COORDS',[q],pos)
    return out+strip_coords(c)

def reference_d5(p):
    from circuit_builder import si1000
    path=ROOT/'reference/MZZ_d5_repaired_ideal.stim'
    cfg=json.loads((ROOT/'reference/binding.json').read_text(encoding='utf-8'))
    if hashlib.sha256(path.read_bytes()).hexdigest()!=cfg['reference_sha256']:raise ValueError('Reference file changed')
    supplied=stim.Circuit(path.read_text(encoding='utf-8-sig'))
    extra,relations=entry_detectors(supplied,5)
    extra_coordinates={tuple(op.gate_args_copy()) for op in extra}
    ideal=stim.Circuit();physical=[];annotations=[];seen_annotation=False
    for op in supplied:
        if op.name in ['DETECTOR','OBSERVABLE_INCLUDE']:
            seen_annotation=True
            if op.name=='DETECTOR' and tuple(op.gate_args_copy()) in extra_coordinates:continue
            annotations.append(op)
        elif op.name!='QUBIT_COORDS':
            if seen_annotation:raise ValueError('Expected terminal annotations in author reference')
            physical.append(op)
    for op in supplied:
        if op.name=='QUBIT_COORDS':ideal.append(op)
    for op in physical+annotations:ideal.append(op)
    ideal+=extra
    prefix_n=cfg['ideal_prefix_instructions'];body_n=cfg['physical_body_instructions']
    if len(physical)!=prefix_n+body_n+cfg['ideal_suffix_instructions']:raise ValueError('Reference operation structure changed')
    blocks=[]
    for subset in [physical[:prefix_n],physical[prefix_n:prefix_n+body_n],physical[prefix_n+body_n:]]:
        block=stim.Circuit()
        for op in subset:block.append(op)
        blocks.append(block)
    noisy=stim.Circuit()
    for op in supplied:
        if op.name=='QUBIT_COORDS':noisy.append(op)
    noisy+=blocks[0]+si1000(blocks[1],p)+blocks[2]
    for op in annotations:noisy.append(op)
    noisy+=extra
    meta=dict(d=5,Dm=5,measurement='ZZ',p=p,author_reference=True,
        generation='author-supplied physical schedule; six entry checks rebuilt from coordinate/measurement indices',
        entry_relations=relations,reference_sha256=cfg['reference_sha256'],merge_extraction_rounds=5,
        initial_and_final_boundary='ideal encoded Bell references and terminal MPP; SI1000 only on physical body',
        observable_definitions=['joint_ZZ','output_Z1','output_Z2','output_XX'],
        ideal_sha256=sha(ideal),circuit_sha256=sha(noisy),validation_not_rerun=True)
    return ideal,noisy,meta

def generate(d,p,measurement,out):
    if d<5 or d%2!=1:raise ValueError('Supported sizes: odd d >=5; d3 is deferred')
    if not 0<=p<.1:raise ValueError('Require 0<=p<0.1')
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Choose an empty output folder')
    if d==5 and measurement=='ZZ':ideal,c,meta=reference_d5(p)
    else:
        from circuit_builder import make
        ideal,c,_,meta=make(d,dm=d,p=p,measurement=measurement)
        meta.update(author_reference=False,generation='coordinate rule generalized from author d5; no local acceptance claim')
    ideal=front_coordinates(ideal);c=front_coordinates(c)
    out.mkdir(parents=True,exist_ok=True)
    (out/'ideal.stim').write_text(str(ideal),encoding='utf-8')
    (out/'circuit.stim').write_text(str(c),encoding='utf-8')
    meta.update(ideal_sha256=sha(ideal),circuit_sha256=sha(c),all_coordinates_at_start=True,
                strict_decoder_compilation='not run by generate.py',formal_sampling_started=False)
    (out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(out)
    return ideal,c,meta
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--d',type=int,default=5);a.add_argument('--p',type=float,default=.001)
    a.add_argument('--measurement',choices=['ZZ','XX','XZ'],default='ZZ');a.add_argument('--out',type=Path,required=True)
    q=a.parse_args();generate(q.d,q.p,q.measurement,q.out)

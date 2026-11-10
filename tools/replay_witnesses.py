"""Replay the archived physical fault witnesses, including detected controls."""
import hashlib,json
from pathlib import Path
import stim

NOISE={'X_ERROR','Y_ERROR','Z_ERROR','DEPOLARIZE1','DEPOLARIZE2'}

def inject(circuit,events):
    ticks={};tick=0
    for i,op in enumerate(circuit):
        assert not isinstance(op,stim.CircuitRepeatBlock)
        if op.name=='TICK':tick+=1
        ticks[i]=tick
    at={};seen=set()
    for e in events:
        i=e['instruction_offset'];op=circuit[i]
        assert op.name in NOISE and op.gate_args_copy()[0]>0
        assert e.get('noise_kind',op.name)==op.name
        assert e.get('tick',ticks[i])==ticks[i]
        qs=[t.value for t in op.targets_copy()];size=2 if op.name=='DEPOLARIZE2' else 1
        group=tuple(e['noise_targets'])
        assert group in [tuple(qs[j:j+size]) for j in range(0,len(qs),size)]
        assert e['paulis'] and set(map(int,e['paulis']))<=set(group)
        assert set(e['paulis'].values())<=set('XYZ')
        if op.name.endswith('_ERROR'):assert set(e['paulis'].values())=={op.name[0]}
        assert (i,group) not in seen;seen.add((i,group));at.setdefault(i,[]).append(e)
    result=stim.Circuit()
    for i,op in enumerate(circuit):
        if op.name in NOISE:
            for e in at.get(i,[]):
                result.append('E',[getattr(stim,'target_'+p.lower())(int(q)) for q,p in e['paulis'].items()],1)
        else:result.append(op)
    return result

def response(circuit):
    detectors=set();logicals=set()
    for op in circuit.detector_error_model(decompose_errors=False).flattened():
        if op.type!='error':continue
        assert op.args_copy()==[1.0]
        for t in op.targets_copy():
            assert not t.is_separator()
            (detectors if t.is_relative_detector_id() else logicals).symmetric_difference_update({t.val})
    sim=stim.FlipSimulator(batch_size=1,num_qubits=circuit.num_qubits,disable_stabilizer_randomization=True)
    sim.do(circuit)
    assert detectors==set(sim.get_detector_flips()[:,0].nonzero()[0].tolist())
    assert logicals==set(sim.get_observable_flips()[:,0].nonzero()[0].tolist())
    return sorted(detectors),sorted(logicals)

def main():
    root=Path(__file__).resolve().parents[1]/'bundles/distance-evidence/witnesses'
    folders=sorted(root.glob('*/result.json'));assert len(folders)==12
    for path in folders:
        info=json.loads(path.read_text());payload=(path.parent/'circuit.stim').read_bytes()
        assert hashlib.sha256(payload).hexdigest()==info['circuit_sha256']
        data=json.loads((path.parent/'events.json').read_text());events=data['events'] if isinstance(data,dict) else data
        assert len(events)==info['weight']
        d,l=response(inject(stim.Circuit(payload.decode('utf-8-sig')),events))
        assert (d,l)==(info['detector_ids'],info['logical_ids']),info['case']
        print(info['case'],f'PASS: weight={len(events)}, D={d}, L={l}')

if __name__=='__main__':main()

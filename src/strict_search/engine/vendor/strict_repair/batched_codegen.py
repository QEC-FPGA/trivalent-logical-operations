"""Batch ONLY the frozen generator's repeated terminal-flow validation calls.

The terminal operators and all generated physical and detector instructions are
unchanged. The exact source substitution is asserted and recorded for review.
"""
import inspect
import stim

def batch_terminal_flows(probe,zero,outputs):
    sols=probe.solve_flow_measurements([stim.Flow(input=zero,output=p) for p in outputs])
    if any(s is None for s in sols):raise ValueError('Missing terminal flow')
    signs=[1]*len(outputs)
    def plus(indices):
        return [stim.Flow(input=zero,output=outputs[i],measurements=sols[i]) for i in indices]
    def classify(indices):
        if not indices or probe.has_all_flows(plus(indices)):return
        if len(indices)==1:
            i=indices[0]
            assert probe.has_flow(stim.Flow(input=zero,output=-outputs[i],measurements=sols[i]))
            signs[i]=-1;return
        mid=len(indices)//2;classify(indices[:mid]);classify(indices[mid:])
    classify(list(range(len(outputs))))
    assert probe.has_all_flows([stim.Flow(input=zero,output=outputs[i]*signs[i],measurements=sols[i]) for i in range(len(outputs))])
    return [dict(exists=True,sign=signs[i],measurement_count=len(s),measurements=list(s)) for i,s in enumerate(sols)]

def install(compiler):
    source=inspect.getsource(compiler.code_stabilizers)
    old='flows = [base.flow(probe, stim.PauliString(n), p) for p in output]'
    assert source.count(old)==1,'Frozen validation call has changed'
    changed=source.replace(old,'flows = batch_terminal_flows(probe, stim.PauliString(n), output)')
    namespace=dict(vars(compiler));namespace['batch_terminal_flows']=batch_terminal_flows
    exec(compile(changed,'<batched terminal validation>','exec'),namespace)
    compiler.code_stabilizers=namespace['code_stabilizers']
    return changed

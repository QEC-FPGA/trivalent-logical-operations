"""Single entry point. No sampling command is selected implicitly."""
from pathlib import Path
import argparse,os,sys,json
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'src'))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=HERE/'output')
    sub=ap.add_subparsers(dest='command',required=True)
    sub.add_parser('check');sub.add_parser('list')
    for name in ('prepare','pilot','budget','confirm','report','mechanism'):
        p=sub.add_parser(name);p.add_argument('--group',choices=list(json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))['groups'])+(['all'] if name=='report' else []),default='d7_p0002')
        if name in ('prepare','pilot','confirm','mechanism'):p.add_argument('--workers',type=int,default=2)
        if name in ('prepare','pilot','mechanism'):p.add_argument('--ids',nargs='+')
        if name=='pilot':p.add_argument('--batch',type=int,default=0);p.add_argument('--shots',type=int)
        if name=='budget':p.add_argument('--max-total-shots',type=int)
        if name=='mechanism':p.add_argument('--search-seconds',type=float,default=60)
    a=ap.parse_args();os.environ['MOVE_W_OUTPUT']=str(a.output.resolve())
    import pipeline as p
    if a.command=='check':
        print(json.dumps(p.runtime(True),indent=2));print('Code SHA256:',p.code_digest())
    elif a.command=='list':
        ps=p.protocol()['points']
        for group in p.protocol()['groups']:
            rows=[v for v in ps if v['group']==group]
            print(group,len(rows),'conditions',sum(v['pilot_shots'] for v in rows),'pilot shots')
        for v in ps:print(v['id'],v['group'],'moves='+str(v['moves']))
    elif a.command=='prepare':p.prepare(a.group,a.workers,a.ids)
    elif a.command=='pilot':p.pilot(a.group,a.workers,a.batch,a.shots,a.ids)
    elif a.command=='budget':p.budget(a.group,a.max_total_shots)
    elif a.command=='confirm':p.confirm(a.group,a.workers)
    elif a.command=='report':
        from reporting import report
        report(a.group)
    elif a.command=='mechanism':
        from mechanism import run
        run(a.group,a.workers,a.ids,a.search_seconds)

if __name__=='__main__':main()

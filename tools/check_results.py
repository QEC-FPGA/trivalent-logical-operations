"""Check paper counts and optional archive hashes. Does not run Monte Carlo or prove distances."""
import argparse, csv, hashlib, json, math, tarfile
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXPECTED_ROWS = {
    'fig09_H_counts.csv': 108,
    'fig10_S_mainline_counts.csv': 46,
    'fig11a_d13_Wscan_counts.csv': 13,
    'fig11b_W1_Wc_ratio_counts.csv': 15,
    'fig11c_W3_Wc_ratio_counts.csv': 12,
    'fig12_surgery_counts.csv': 81,
}
def rows(name):return list(csv.DictReader((ROOT/'data/counts'/name).open()))
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--assets-dir',type=Path);ap.add_argument('--members',action='store_true');a=ap.parse_args()
    if a.members and not a.assets_dir:ap.error('--members requires --assets-dir')
    counts=0
    for name, expected in EXPECTED_ROWS.items():
        rr=rows(name);assert len(rr)==expected,name
        for r in rr:
            if 'shots' in r and 'errors' in r:
                n,e=int(r['shots']),int(r['errors']);assert 0<=e<=n and n>0
                assert math.isclose(float(r['rate']),e/n,rel_tol=1e-10,abs_tol=1e-15)
        counts+=len(rr)
    sums=defaultdict(lambda:[0,0]);shards=0
    contexts=json.loads((ROOT/'data/h/protocol.json').read_text())['contexts'];seen=set()
    for r in csv.DictReader((ROOT/'data/h/stats.csv').open()):
        assert contexts[r['context_id']]['decoder']=='pymatching_correlated'
        assert r['record_sha256'] not in seen;seen.add(r['record_sha256'])
        n,e=int(r['shots']),int(r['mistakes']);assert 0<=e<=n and n>0
        key=(r['circuit_sha256'],r['dem_sha256']);sums[key][0]+=n;sums[key][1]+=e;shards+=1
    assert shards==254
    for r in rows('fig09_H_counts.csv'):assert sums[(r['circuit_sha256'],r['dem_sha256'])]==[int(r['shots']),int(r['errors'])]
    d13=json.loads((ROOT/'data/fig10_d13_cohorts.json').read_text())
    for r in d13:
        row=next(x for x in rows('fig10_S_mainline_counts.csv') if int(x['d'])==13 and x['channel']=='XY' and float(x['p'])==r['p'])
        assert (int(row['shots']),int(row['errors']))==(r['shots'],r['errors'])
        assert sum(x['shots'] for x in r['cohorts'])==r['shots'] and sum(x['errors'] for x in r['cohorts'])==r['errors']
    print(f'PASS: {counts} paper rows; {shards} H shards; 3 d13 cohort bindings')
    if a.assets_dir:
        for asset in json.loads((ROOT/'release-assets.json').read_text()):
            p=a.assets_dir/asset['name']
            if not p.exists():
                print('NOT CHECKED (not supplied):',p.name);continue
            with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==asset['sha256'],p
            assert p.stat().st_size==asset['bytes']
            if a.members:
                actual={};manifest=None
                with tarfile.open(p,'r|gz') as tf:
                    for m in tf:
                        assert m.isfile() and not m.name.startswith('/') and '..' not in Path(m.name).parts
                        n=m.name.removeprefix(asset['destination']+'/');assert n!=m.name and n not in actual
                        f=tf.extractfile(m)
                        if n=='MANIFEST.json':manifest=json.load(f)
                        else:actual[n]=dict(sha256=hashlib.file_digest(f,'sha256').hexdigest(),bytes=m.size)
                assert actual==manifest,p.name
            print('PASS:',p.name)
if __name__=='__main__':main()

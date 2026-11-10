from collections import defaultdict
from pathlib import Path
import sys
src=Path(sys.argv[1]); dst=Path(sys.argv[2]); max_pass=int(sys.argv[3]) if len(sys.argv)>3 else 20
with src.open() as f:
 M,N,K=map(int,f.readline().split()); target=list(map(int,f.readline().split())); assert K==1
 L=target[0]; rows=[set() for _ in range(M)]; cols=[]
 for c,line in enumerate(f):
  a=list(map(int,line.split()))[1:]; cols.append(set(a))
  for r in a:rows[r].add(c)
ops=[]
def total():return sum(map(len,rows))
for ps in range(max_pass):
 changed=0
 for i in sorted(range(M),key=lambda r:len(rows[r]),reverse=True):
  if not rows[i]:continue
  cnt=defaultdict(int)
  for c in rows[i]:
   for j in cols[c]:
    if j!=i and not(i!=L and j==L):cnt[j]+=1
  best=None;bg=0
  for j,ov in cnt.items():
   g=2*ov-len(rows[j])
   if g>bg:bg=g;best=j
  if best is None or bg<=0:continue
  for c in list(rows[best]):
   if c in rows[i]:rows[i].remove(c);cols[c].remove(i)
   else:rows[i].add(c);cols[c].add(i)
  ops.append((i,best));changed+=1
 if changed==0:break
with dst.open('w') as f:
 f.write(f'{M} {N} 1\n{L}\n')
 for s in cols:
  a=sorted(s);f.write(str(len(a))+(' '+' '.join(map(str,a)) if a else '')+'\n')
Path(str(dst)+'.rowops').write_text('\n'.join(f'{a} {b}' for a,b in ops)+'\n')
print('done ops',len(ops),'nnz',total(),'maxcol',max(map(len,cols)))

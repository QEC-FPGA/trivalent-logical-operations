from pipeline import final_counts,ROOT,write
from pathlib import Path
import csv,math
def interval(k,n):
 z=1.95996398454;q=k/n;v=1+z*z/n;h=z*math.sqrt(q*(1-q)/n+z*z/(4*n*n))/v;m=(q+z*z/(2*n))/v
 return m-h,m+h
def report(group):
 rows=final_counts(group);out=ROOT/'reports';out.mkdir(parents=True,exist_ok=True)
 for r in rows:r['lo95'],r['hi95']=interval(r['mistakes'],r['shots'])
 write(out/(group+'_counts.json'),rows)
 keys=['id','d','step','W','moves','channel','p','shots','mistakes','rate','lo95','hi95','gate_extraction_rounds','extraction_rounds','two_qubit_layers','plan_id']
 with (out/(group+'_counts.csv')).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
 print(out/(group+'_counts.csv'))

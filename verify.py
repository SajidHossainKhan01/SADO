# Independent re-computation (no shared code with analyze.py)
import json, re, statistics as st
from scipy.stats import wilcoxon
runs=[json.loads(l) for l in open('/home/claude/marl/results/runs.jsonl')]
mac=dict(re.findall(r'\\newcommand\{\\(\w+)\}\{([^}]*)\}', open('/home/claude/marl/paper/numbers.tex').read()))
def pick(exp,key,**c):
    out={}
    for r in runs:
        if r['exp']==exp and r['key']==key and all(r['over'].get(k)==v for k,v in c.items()): out[r['seed']]=r
    return out
checks=[]
def chk(name,val,fmt='{:.1f}'):
    got=mac.get(name); exp=fmt.format(val)
    checks.append((name,got,exp,got==exp))
E1=lambda k: pick('E1',k)
for k,K in [('prop','Sado'),('iql','Iql'),('po2','Potwo'),('greedy','Greedy'),('oracle','Oracle'),('random','Rand'),('local','Local')]:
    d=E1(k); chk('DSR'+K,100*st.mean(r['DSR'] for r in d.values())); chk('HP'+K,100*st.mean(r['DSR_hp'] for r in d.values()))
    chk('LAT'+K,st.mean(r['lat_ms'] for r in d.values()),'{:.0f}'); chk('EN'+K,st.mean(r['energy_mJ'] for r in d.values()))
for k,K in [('iql','Iql'),('po2','Potwo'),('greedy','Greedy')]:
    a=E1('prop'); b=E1(k); s=sorted(a)
    x=[a[i]['DSR'] for i in s]; y=[b[i]['DSR'] for i in s]
    p=wilcoxon(x,y).pvalue; chk('PE'+K,p,'{:.3f}' if p>=0.001 else '<0.001')
    chk('DE'+K,100*st.mean(u-v for u,v in zip(x,y)))
def dsr(exp,key,**c): d=pick(exp,key,**c); return 100*st.mean(r['DSR'] for r in d.values()), len(d)
for k,K in [('greedy','Greedy'),('po2','Potwo'),('iql','Iql'),('prop','Sado')]:
    a,_=dsr('E2',k,B=1); b,_=dsr('E2',k,B=100)
    chk('DSR'+K+'BOne',a); chk('DSR'+K+'BHundred',b)
    # loss uses unrounded means
    chk('LOSS'+K,a-b)
for k,K in [('prop','Sado'),('po2','Potwo'),('iql','Iql')]:
    for N,NW in [(40,'NForty'),(80,'NEighty')]:
        v,n=dsr('E5',k,N=N); chk('DSR'+K+NW,v); print('E5',k,N,'seeds',n)
for k,K in [('prop_corr','Corr'),('aoi','Aoi'),('iql_soft05','Soft')]:
    v,_=dsr('E6',k,B=100); chk('DSR'+K+'BHundred',v)
v,_=dsr('E3','prop',p=0.24); chk('DSRSadoPTwentyfour',v)
v,_=dsr('E3','oracle',p=0.24); chk('DSROraclePTwentyfour',v)
v,_=dsr('E4','iql',B=20,phase='random'); chk('DSRIqlBTwentyRand',v)
bad=[c for c in checks if not c[3]]
print(f'{len(checks)} checks, {len(bad)} mismatches'); [print('MISMATCH',c) for c in bad]
# completeness
import collections
print(collections.Counter(r['exp'] for r in runs)); print('dup tags', len(runs)-len({r['tag'] for r in runs}))

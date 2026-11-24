import sim, json, os, sys
from multiprocessing import Pool
SEEDS = list(range(101, 111))
OUT = '/home/claude/marl/results/runs.jsonl'
def caps_for(M): base=(80.,60.,50.,40.); return tuple(base[i%4] for i in range(M))
def jobs():
    J=[]
    for k in ['local','random','greedy','po2','oracle','iql','aoi','iql_soft05','prop','prop_corr']:
        for s in SEEDS: J.append(('E1',k,s,dict(p=0.20,B=20)))
    for B in [1,5,50,100]:
        for k in ['greedy','po2','iql','prop']:
            for s in SEEDS: J.append(('E2',k,s,dict(p=0.20,B=B)))
    for B in [50,100]:
        for k in ['aoi','iql_soft05','prop_corr']:
            for s in SEEDS: J.append(('E6',k,s,dict(p=0.20,B=B)))
    for p in [0.12,0.16,0.24]:
        for k in ['random','greedy','po2','oracle','iql','prop']:
            for s in SEEDS: J.append(('E3',k,s,dict(p=p,B=20)))
    for B in [20,50]:
        for k in ['greedy','po2','iql','prop']:
            for s in SEEDS: J.append(('E4',k,s,dict(p=0.20,B=B,phase='random')))
    for N,M,SS in [(40,8,SEEDS),(80,16,SEEDS[:5])]:
        for k in ['greedy','po2','oracle','iql','prop']:
            for s in SS: J.append(('E5',k,s,dict(p=0.20,B=20,N=N,M=M,caps=caps_for(M))))
    return J
def tag(j): return f"{j[0]}|{j[1]}|{j[2]}|{json.dumps(j[3],sort_keys=True)}"
def work(j):
    e,k,s,over=j; r=sim.run(k,s,**over); r['exp']=e; r['tag']=tag(j)
    r['over']={a:(list(b) if isinstance(b,tuple) else b) for a,b in over.items()}
    return {a:(float(b) if hasattr(b,'item') else b) for a,b in r.items()}
if __name__=='__main__':
    done=set()
    if os.path.exists(OUT):
        for l in open(OUT): done.add(json.loads(l)['tag'])
    J=[j for j in jobs() if tag(j) not in done]
    J.sort(key=lambda j: 0 if j[0]!='E5' else 1)
    print('todo',len(J),flush=True)
    with Pool(2) as P, open(OUT,'a') as f:
        for i,r in enumerate(P.imap_unordered(work,J)):
            f.write(json.dumps(r)+'\n'); f.flush()
            if i%20==0: print(i,r['tag'],round(r['DSR'],3),flush=True)

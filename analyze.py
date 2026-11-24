import json, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
R='/home/claude/marl/results/'; F='/home/claude/marl/figs/'
rows=[json.loads(l) for l in open(R+'runs.jsonl')]
for r in rows:
    for k,v in r['over'].items():
        if not isinstance(v,list): r[k]=v
df=pd.DataFrame(rows)
NAME={'local':'Local-only','random':'Random','greedy':'Greedy-stale','po2':'Po2-stale','oracle':'Oracle-greedy',
      'iql':'IQL','aoi':'IQL+AoI','iql_soft05':'IQL+Soft','prop':'SADO (proposed)','prop_corr':'SADO+Corr'}
COL={'prop':'#2a78d6','iql':'#eb6834','po2':'#1baf7a','greedy':'#eda100','random':'#e87ba4','oracle':'#6b6b6b'}
MK={'prop':'o','iql':'s','po2':'^','greedy':'D','random':'v','oracle':None}
def ci(x):
    x=np.asarray(x,float); x=x[~np.isnan(x)]; n=len(x)
    if n<2: return (x.mean() if n else np.nan),np.nan
    return x.mean(), stats.t.ppf(0.975,n-1)*x.std(ddof=1)/np.sqrt(n)
def sub(e,**kw):
    d=df[df.exp==e]
    for k,v in kw.items():
        if k not in d.columns: return d.iloc[0:0]
        d=d[d[k]==v]
    return d
out={}
# ---- E1 main table
e1=df[df.exp=='E1']; metrics=['DSR','DSR_hp','utility','lat_ms','energy_mJ','drop','herd']
order=['local','random','greedy','po2','iql','prop','oracle']
abl=['iql','aoi','iql_soft05','prop','prop_corr']
T={}
for k in set(order+abl):
    d=e1[e1.key==k]
    if len(d): T[k]={m:ci(d[m]) for m in metrics}; T[k]['n']=len(d)
out['E1']={k:{m:list(map(float,v)) if isinstance(v,tuple) else v for m,v in T[k].items()} for k in T}
# Wilcoxon prop vs others (paired by seed) on DSR
W={}
p=e1[e1.key=='prop'].set_index('seed')
for k in T:
    if k=='prop': continue
    o=e1[e1.key==k].set_index('seed'); s=p.index.intersection(o.index)
    if len(s)>=5:
        for m in ['DSR','DSR_hp','utility']:
            dlt=p.loc[s,m]-o.loc[s,m]
            pv=stats.wilcoxon(p.loc[s,m],o.loc[s,m]).pvalue if (dlt!=0).any() else 1.0
            W[f'{k}|{m}']=dict(p=float(pv),mean_diff=float(dlt.mean()),wins=int((dlt>0).sum()),n=len(s))
out['wilcoxon_E1']=W
def fmt(m,v,pct=False,d=1):
    mu,h=v
    if pct: return f"{100*mu:.{d}f} $\\pm$ {100*h:.{d}f}"
    return f"{mu:.{d}f} $\\pm$ {h:.{d}f}"
def row(k):
    t=T[k]
    return (f"{NAME[k]} & {fmt('',t['DSR'],1)} & {fmt('',t['DSR_hp'],1)} & {fmt('',t['utility'],d=3)} & "
            f"{fmt('',t['lat_ms'],d=0)} & {fmt('',t['energy_mJ'],d=1)} & " +
            (f"{fmt('',t['herd'],d=2)}" if not np.isnan(t['herd'][0]) else "--") + r" \\")
with open(R+'tab_main.tex','w') as f:
    for k in order:
        if k in T: f.write(row(k)+'\n')
        if k=='prop': f.write('\\midrule\n')
with open(R+'tab_abl.tex','w') as f:
    for k in abl:
        if k in T: f.write(row(k)+'\n')
# ---- E2 staleness sweep (B=20 from E1)
def sweep(e,var,vals,keys,extra_from_E1=None,**kw):
    res={}
    for k in keys:
        res[k]=[]
        for v in vals:
            d=sub(e,key=k,**{var:v},**kw)
            if extra_from_E1 is not None and v==extra_from_E1: d=e1[e1.key==k]
            res[k].append(ci(d['DSR']) if len(d) else (np.nan,np.nan))
    return res
Bs=[1,5,20,50,100]; k2=['greedy','po2','iql','prop']
S2=sweep('E2','B',Bs,k2,extra_from_E1=20); out['E2']={k:[list(map(float,x)) for x in v] for k,v in S2.items()}
S2h={}
for k in k2:
    S2h[k]=[ci((e1[e1.key==k] if B==20 else sub('E2',key=k,B=B))['DSR_hp']) for B in Bs]
out['E2_hp']={k:[list(map(float,x)) for x in v] for k,v in S2h.items()}
orc=ci(e1[e1.key=='oracle']['DSR']) if (e1.key=='oracle').any() else (np.nan,np.nan)
def lineplot(ax,xs,S,keys,xlabel,logx=False,ref=None):
    for k in keys:
        mu=np.array([a for a,b in S[k]]); h=np.array([b for a,b in S[k]])
        ax.plot(xs,100*mu,color=COL[k],marker=MK[k],ms=6,lw=2,label=NAME[k])
        ax.fill_between(xs,100*(mu-h),100*(mu+h),color=COL[k],alpha=0.15,lw=0)
    if ref is not None and not np.isnan(ref[0]):
        ax.axhline(100*ref[0],color=COL['oracle'],ls='--',lw=1.5,label='Oracle-greedy (fresh info)')
    if logx: ax.set_xscale('log')
    ax.set_xlabel(xlabel); ax.set_ylabel('Deadline satisfaction ratio (%)')
    ax.grid(alpha=0.25,lw=0.6); [ax.spines[s].set_visible(False) for s in ['top','right']]
plt.rcParams.update({'font.size':10,'font.family':'serif'})
fig,ax=plt.subplots(figsize=(5.2,3.6)); lineplot(ax,Bs,S2,k2,'Broadcast interval $B$ (slots of 10 ms, log scale)',logx=True,ref=orc)
ax.set_xticks(Bs); ax.set_xticklabels([str(b) for b in Bs]); ax.legend(frameon=False,fontsize=8,loc='lower left')
fig.tight_layout(); fig.savefig(F+'fig_staleness.pdf'); plt.close(fig)
# ---- E3 load sweep (p=0.20 from E1)
ps=[0.12,0.16,0.20,0.24]; k3=['random','greedy','po2','iql','prop','oracle']
S3={}
for k in k3:
    S3[k]=[ci((e1[e1.key==k] if p_==0.20 else sub('E3',key=k,p=p_))['DSR']) for p_ in ps]
out['E3']={k:[list(map(float,x)) for x in v] for k,v in S3.items()}
fig,ax=plt.subplots(figsize=(5.2,3.6))
lineplot(ax,[20*p_ for p_ in ps],S3,['random','greedy','po2','iql','prop'],'Aggregate arrival rate (tasks per slot)')
mu=np.array([a for a,b in S3['oracle']]); ax.plot([20*p_ for p_ in ps],100*mu,color=COL['oracle'],ls='--',lw=1.5,label='Oracle-greedy (fresh info)')
ax.legend(frameon=False,fontsize=8,loc='lower left'); fig.tight_layout(); fig.savefig(F+'fig_load.pdf'); plt.close(fig)
# ---- E4 phase
E4={}
for B in [20,50]:
    for k in k2:
        sy=(e1[e1.key==k] if B==20 else sub('E2',key=k,B=50))['DSR']; ra=sub('E4',key=k,B=B)['DSR']
        E4[f'{k}|{B}']=dict(sync=list(map(float,ci(sy))),rand=list(map(float,ci(ra))))
out['E4']=E4
with open(R+'tab_phase.tex','w') as f:
    for k in k2:
        c=[E4[f'{k}|{B}'][ph] for B in [20,50] for ph in ['sync','rand']]
        f.write(NAME[k]+' & '+' & '.join(fmt('',x,1) for x in c)+r' \\'+'\n')
# ---- E5 scalability
E5={}
for N in [20,40,80]:
    for k in ['greedy','po2','iql','prop','oracle']:
        d=e1[e1.key==k] if N==20 else sub('E5',key=k,N=N)
        E5[f'{k}|{N}']=list(map(float,ci(d['DSR']))) if len(d) else [np.nan,np.nan]
out['E5']=E5
with open(R+'tab_scale.tex','w') as f:
    for k in ['greedy','po2','iql','prop','oracle']:
        f.write(NAME[k]+' & '+' & '.join(fmt('',E5[f'{k}|{N}'],1) for N in [20,40,80])+r' \\'+'\n')
# ---- convergence curve
fig,ax=plt.subplots(figsize=(5.2,3.4))
for k in ['iql','prop']:
    C=[r['curve'] for r in rows if r['exp']=='E1' and r['key']==k]
    if not C: continue
    L=min(len(c) for c in C); t=np.array([C[0][i][0] for i in range(L)])/1000*10
    M=np.array([[c[i][1] for i in range(L)] for c in C]); mu=M.mean(0); h=stats.t.ppf(0.975,len(C)-1)*M.std(0,ddof=1)/np.sqrt(len(C))
    ax.plot(t,100*mu,color=COL[k],marker=MK[k],ms=4,markevery=5,lw=2,label=NAME[k]); ax.fill_between(t,100*(mu-h),100*(mu+h),color=COL[k],alpha=0.15,lw=0)
ax.axvline(300,color='#6b6b6b',ls=':',lw=1); yl=ax.get_ylim(); ax.text(296,yl[1]-0.15*(yl[1]-yl[0]),'training | evaluation (frozen)',fontsize=8,color='#444',ha='center',bbox=dict(fc='white',ec='none',pad=1))
ax.set_xlabel('Simulated time (s)'); ax.set_ylabel('DSR per 10 s window (%)'); ax.grid(alpha=0.25,lw=0.6)
[ax.spines[s].set_visible(False) for s in ['top','right']]; ax.legend(frameon=False,fontsize=8,loc='upper left')
fig.tight_layout(); fig.savefig(F+'fig_convergence.pdf'); plt.close(fig)
# herding vs B
fig,ax=plt.subplots(figsize=(5.2,3.4))
for k in k2:
    H=[ci((e1[e1.key==k] if B==20 else sub('E2',key=k,B=B))['herd']) for B in Bs]
    mu=np.array([a for a,b in H]); ax.plot(Bs,mu,color=COL[k],marker=MK[k],ms=6,lw=2,label=NAME[k])
    out.setdefault('herd_B',{})[k]=[list(map(float,x)) for x in H]
cap_share=80/230; ax.axhline(cap_share,color='#6b6b6b',ls='--',lw=1.2,label='Capacity share of largest server')
ax.set_xscale('log'); ax.set_xticks(Bs); ax.set_xticklabels([str(b) for b in Bs])
ax.set_xlabel('Broadcast interval $B$ (slots, log scale)'); ax.set_ylabel('Herding index'); ax.grid(alpha=0.25,lw=0.6)
[ax.spines[s].set_visible(False) for s in ['top','right']]; ax.legend(frameon=False,fontsize=8,ncol=3,loc='upper center',bbox_to_anchor=(0.5,-0.22))
fig.tight_layout(); fig.savefig(F+'fig_herding.pdf',bbox_inches='tight'); plt.close(fig)
json.dump(out,open(R+'summary.json','w'),indent=1,default=float)
print('runs',len(df),df.groupby('exp').size().to_dict())

# ================= extended: ablation across B, paired tests, macros =================
def getd(k,B=20,p_=0.20,N=20):
    if N!=20: return sub('E5',key=k,N=N)
    if p_!=0.20: return sub('E3',key=k,p=p_)
    if B==20: return e1[e1.key==k]
    for e in ['E2','E6']:
        d=sub(e,key=k,B=B)
        if len(d): return d
    return df.iloc[0:0]
def paired(a,b,m='DSR'):
    a=a.set_index('seed')[m]; b=b.set_index('seed')[m]; s=a.index.intersection(b.index)
    if len(s)<5: return dict(p=float('nan'),diff=float('nan'),wins=0,n=len(s))
    dl=a[s]-b[s]; pv=stats.wilcoxon(a[s],b[s]).pvalue if (dl!=0).any() else 1.0
    return dict(p=float(pv),diff=float(dl.mean()),wins=int((dl>0).sum()),n=int(len(s)))
AB={}
with open(R+'tab_abl.tex','w') as f:
    for k in abl:
        cells=[]
        for B in [20,50,100]:
            d=getd(k,B=B); mu,h=ci(d['DSR']) if len(d) else (np.nan,np.nan)
            pr=paired(getd('prop',B=B),d) if k!='prop' else None
            AB[f'{k}|{B}']=dict(mu=float(mu),h=float(h),test=pr)
            cells.append(f"{100*mu:.1f} $\\pm$ {100*h:.1f}" if not np.isnan(mu) else '--')
            if k!='prop':
                cells[-1]+= f" ({pr['p']:.3f})" if not np.isnan(pr['p']) else ''
        f.write(NAME[k]+' & '+' & '.join(cells)+r' \\'+'\n')
out['ablation']=AB
TESTS={}
for B in Bs:
    for k in ['greedy','po2','iql']:
        TESTS[f'B{B}|{k}']=paired(getd('prop',B=B),getd(k,B=B))
        TESTS[f'B{B}|{k}|hp']=paired(getd('prop',B=B),getd(k,B=B),'DSR_hp')
for p_ in ps:
    for k in ['random','greedy','po2','iql','oracle']:
        TESTS[f'p{p_}|{k}']=paired(getd('prop',p_=p_),getd(k,p_=p_))
for N in [40,80]:
    for k in ['greedy','po2','iql','oracle']:
        TESTS[f'N{N}|{k}']=paired(getd('prop',N=N),getd(k,N=N))
out['tests']=TESTS
json.dump(out,open(R+'summary.json','w'),indent=1,default=float)
# ---- macros
KW={'local':'Local','random':'Rand','greedy':'Greedy','po2':'Potwo','oracle':'Oracle','iql':'Iql','aoi':'Aoi','iql_soft05':'Soft','prop':'Sado','prop_corr':'Corr'}
BW={1:'BOne',5:'BFive',20:'BTwenty',50:'BFifty',100:'BHundred'}
PW={0.12:'PTwelve',0.16:'PSixteen',0.20:'PTwenty',0.24:'PTwentyfour'}
NW={20:'NTwenty',40:'NForty',80:'NEighty'}
M=[]
def mac(name,val): M.append(f"\\newcommand{{\\{name}}}{{{val}}}")
def pct(x,d=1): return 'n/a' if x!=x else f"{100*x:.{d}f}"
for k,K in KW.items():
    d=e1[e1.key==k]
    if not len(d): continue
    mac('DSR'+K,pct(d.DSR.mean())); mac('DSRci'+K,pct(ci(d.DSR)[1]))
    mac('HP'+K,pct(d.DSR_hp.mean())); mac('UT'+K,f"{d.utility.mean():.3f}")
    mac('LAT'+K,f"{d.lat_ms.mean():.0f}"); mac('EN'+K,f"{d.energy_mJ.mean():.1f}")
    mac('DROP'+K,pct(d['drop'].mean()))
    if d.herd.notna().any(): mac('HERD'+K,f"{d.herd.mean():.2f}")
    mac('OFF'+K,pct(d.offload.mean(),0))
for B,BWd in BW.items():
    for k in ['greedy','po2','iql','prop','aoi','iql_soft05','prop_corr']:
        d=getd(k,B=B)
        if len(d):
            mac('DSR'+KW[k]+BWd,pct(d.DSR.mean())); mac('HP'+KW[k]+BWd,pct(d.DSR_hp.mean()))
            if d.herd.notna().any(): mac('HERD'+KW[k]+BWd,f"{d.herd.mean():.2f}")
for p_,PWd in PW.items():
    for k in ['random','greedy','po2','iql','prop','oracle']:
        d=getd(k,p_=p_)
        if len(d): mac('DSR'+KW[k]+PWd,pct(d.DSR.mean()))
for N,NWd in NW.items():
    for k in ['greedy','po2','iql','prop','oracle']:
        d=getd(k,N=N)
        if len(d): mac('DSR'+KW[k]+NWd,pct(d.DSR.mean()))
def pfmt(p): return 'n/a' if p!=p else ('<0.001' if p<0.001 else f"{p:.3f}")
for key,t in TESTS.items():
    nm=key.replace('|hp','HP')
    for a,b in [('B1','BOne'),('B5','BFive'),('B20','BTwenty'),('B50','BFifty'),('B100','BHundred'),('p0.12','PTwelve'),('p0.16','PSixteen'),('p0.2','PTwenty'),('p0.24','PTwentyfour'),('N40','NForty'),('N80','NEighty')]:
        if nm.startswith(a+'|'): nm=b+nm[len(a):]; break
    parts=nm.split('|'); cond=parts[0]; k=parts[1]; hp='HP' if len(parts)>2 else ''
    mac('P'+cond+KW[k.replace('HP','')]+('HP' if 'HP' in k else ''),pfmt(t['p']))
    mac('D'+cond+KW[k.replace('HP','')]+('HP' if 'HP' in k else ''),pct(t['diff']))
    mac('W'+cond+KW[k.replace('HP','')]+('HP' if 'HP' in k else ''),f"{t['wins']}/{t['n']}")
for kk,v in W.items():
    k,m=kk.split('|'); mm={'DSR':'','DSR_hp':'HP','utility':'UT'}[m]
    mac('PE'+KW[k]+mm,pfmt(v['p'])); mac('WE'+KW[k]+mm,f"{v['wins']}/{v['n']}")
    mac('DE'+KW[k]+mm,pct(v['mean_diff']) if m!='utility' else f"{v['mean_diff']:.3f}")
for k in ['greedy','po2','iql','prop','aoi','iql_soft05','prop_corr']:
    for B in [20,50,100]:
        a=AB.get(f'{k}|{B}')
        if a and a['test']: mac('PA'+KW[k]+BW[B],pfmt(a['test']['p']))
mac('NRUNS',str(len(df)))
for k in ['iql','prop']:
    Mc=np.array([[c[1] for c in r['curve']] for r in rows if r['exp']=='E1' and r['key']==k])
    mac('CURVEend'+KW[k],pct(Mc[:,25:30].mean())); mac('CURVEeval'+KW[k],pct(Mc[:,30:].mean())); mac('CURVEstart'+KW[k],pct(Mc[:,:6].mean()))
for k in ['greedy','po2','iql','prop']:
    H=[getd(k,B=B).herd.mean() for B in Bs]; mac('HERDmin'+KW[k],f"{min(H):.2f}"); mac('HERDmax'+KW[k],f"{max(H):.2f}")
for k in ['greedy','po2','iql','prop']:
    a=getd(k,B=1).DSR.mean(); b=getd(k,B=100).DSR.mean(); mac('LOSS'+KW[k],pct(a-b))
for k in ['greedy','po2','iql','prop']:
    for B in [20,50]:
        d=sub('E4',key=k,B=B)
        if len(d): mac('DSR'+KW[k]+BW[B]+'Rand',pct(d.DSR.mean()))
for k in ['iql','prop','aoi','iql_soft05','prop_corr']:
    for B in [20,50,100]:
        d=getd(k,B=B)
        if len(d): mac('CI'+KW[k]+BW[B],pct(ci(d.DSR)[1]))
for N in [40,80]:
    for k in ['iql','prop','po2']:
        d=getd(k,N=N); mac('CI'+KW[k]+NW[N],pct(ci(d.DSR)[1]))
seen=set(); MM=[]
for line in M:
    nm=line.split('}')[0]
    if nm not in seen: seen.add(nm); MM.append(line)
open('/home/claude/marl/paper/numbers.tex','w').write('% auto-generated by analyze.py from results/runs.jsonl -- do not edit\n'+'\n'.join(MM)+'\n')
print('macros',len(MM))

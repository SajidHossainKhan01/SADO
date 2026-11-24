"""Edge-IoT offloading simulator with stale load broadcasts.
Slot = 10 ms. Units: cycles in Mcycles, data in Mbit, energy in J."""
import numpy as np
from bisect import bisect_right
from collections import deque

DEF = dict(N=20, M=4, caps=(80., 60., 50., 40.), f_local=10., p=0.15,
           cyc=(50., 150.), data=(0.5, 2.0), rate=(0.10, 0.40), fade_sigma=0.3,
           hp_frac=0.3, dl_hp=(15, 30), dl_np=(30, 80), w_hp=2.0, w_np=1.0,
           B=20, phase='sync', kappa=1e-27, p_tx=0.5, lam=0.2, e_norm=0.1,
           T_train=30000, T_eval=20000)

SLACK_EDGES = [0.5, 0.75, 1.0, 1.5]      # ratio T_hat/slack buckets -> 5
AOI_EDGES = [1, 6, 21, 51]               # aoi buckets -> 5

def bucket(x, edges):
    return bisect_right(edges, x)

class Policy:
    learns = False
    def __init__(self, cfg, rng): self.cfg, self.rng = cfg, rng
    def act(self, ctx): raise NotImplementedError
    def feedback(self, info, r, actual_wait): pass
    def set_phase(self, training, frac): pass

class LocalOnly(Policy):
    name = 'Local-only'
    def act(self, ctx): return 0, None

class RandomPol(Policy):
    name = 'Random'
    def act(self, ctx): return int(self.rng.integers(0, self.cfg['M'] + 1)), None

def est_times(ctx, cfg, backlog_view):
    """estimated completion time for each action (0=local, m+1=edge m)."""
    c = ctx['cyc']
    t = np.empty(cfg['M'] + 1)
    t[0] = (ctx['local_backlog'] + c) / cfg['f_local']
    caps = np.asarray(cfg['caps'])
    t[1:] = ctx['tx'] + (backlog_view + c) / caps
    return t

class GreedyStale(Policy):
    name = 'Greedy-stale'
    def act(self, ctx):
        t = est_times(ctx, self.cfg, ctx['obs_backlog'])
        return int(np.argmin(t)), None

class PowerOfTwo(Policy):
    name = 'Po2-stale'
    def act(self, ctx):
        t = est_times(ctx, self.cfg, ctx['obs_backlog'])
        cand = [0] + list(1 + self.rng.choice(self.cfg['M'], 2, replace=False))
        return int(min(cand, key=lambda a: t[a])), None

class OracleGreedy(Policy):
    name = 'Oracle-greedy'
    def act(self, ctx):
        t = est_times(ctx, self.cfg, ctx['true_backlog_incl_transit'])
        return int(np.argmin(t)), None

class QAgent(Policy):
    """Shared (parameter-sharing) action-feature Q-learning.
    use_aoi: AoI bucket in features; use_corr: feedback-learned staleness
    correction; softmax: Boltzmann execution policy."""
    learns = True
    def __init__(self, cfg, rng, use_aoi, use_corr, softmax, name,
                 alpha=0.05, eps0=0.3, eps1=0.02, tau=0.15, corr_beta=0.05, shared_corr=False):
        super().__init__(cfg, rng)
        self.use_aoi, self.use_corr, self.softmax, self.name = use_aoi, use_corr, softmax, name
        self.alpha, self.eps0, self.eps1, self.tau, self.cb = alpha, eps0, eps1, tau, corr_beta
        self.shared_corr = shared_corr
        self.Q = np.zeros((2, 5, 5, 2))          # prio, ratio bucket, aoi bucket, type(local/edge)
        # per-device, per-server, per-aoi-bucket correction of waiting time (slots)
        self.corr = np.zeros((cfg['N'], cfg['M'], 5))
        self.training, self.eps = True, eps0
    def _d(self, dev): return 0 if self.shared_corr else dev
    def set_phase(self, training, frac):
        self.training = training
        self.eps = self.eps0 + (self.eps1 - self.eps0) * frac if training else 0.0
    def feats(self, ctx):
        cfg = self.cfg; M = cfg['M']
        t = est_times(ctx, cfg, ctx['obs_backlog'])
        ab = bucket(ctx['aoi'], AOI_EDGES)
        if self.use_corr:
            t[1:] = t[1:] + np.maximum(self.corr[self._d(ctx['dev']), :, ab], -t[1:] + ctx['tx'][:] + ctx['cyc'] / np.asarray(cfg['caps']))
        slack = max(ctx['slack'], 1e-6)
        F = []
        for a in range(M + 1):
            rb = bucket(t[a] / slack, SLACK_EDGES)
            aoi_b = ab if (self.use_aoi and a > 0) else 0
            F.append((ctx['prio'], rb, aoi_b, 0 if a == 0 else 1))
        return F, ab
    def act(self, ctx):
        F, ab = self.feats(ctx)
        q = np.array([self.Q[f] for f in F])
        if self.training and self.rng.random() < self.eps:
            a = int(self.rng.integers(0, len(F)))
        elif self.softmax:
            z = (q - q.max()) / self.tau
            pr = np.exp(z); pr /= pr.sum()
            a = int(self.rng.choice(len(F), p=pr))
        else:
            best = np.flatnonzero(q >= q.max() - 1e-12)
            a = int(self.rng.choice(best))
        pred_wait = ctx['obs_backlog'][a - 1] / self.cfg['caps'][a - 1] if a > 0 else 0.0
        return a, (F[a], ctx['dev'], a, ab, pred_wait)
    def feedback(self, info, r, actual_wait):
        f, dev, a, ab, pred_wait = info
        if self.training:
            self.Q[f] += self.alpha * (r - self.Q[f])
        if self.use_corr and a > 0 and actual_wait is not None:
            e = actual_wait - pred_wait
            d = self._d(dev); self.corr[d, a - 1, ab] += self.cb * (e - self.corr[d, a - 1, ab])

def make_policy(key, cfg, rng):
    return {
        'local': lambda: LocalOnly(cfg, rng),
        'random': lambda: RandomPol(cfg, rng),
        'greedy': lambda: GreedyStale(cfg, rng),
        'po2': lambda: PowerOfTwo(cfg, rng),
        'oracle': lambda: OracleGreedy(cfg, rng),
        'iql': lambda: QAgent(cfg, rng, False, False, False, 'IQL'),
        'aoi': lambda: QAgent(cfg, rng, True, False, False, 'IQL+AoI'),
        'corr': lambda: QAgent(cfg, rng, False, True, False, 'IQL+Corr'),
        'aoi_corr': lambda: QAgent(cfg, rng, True, True, False, 'IQL+AoI+Corr'),
        'aoi_soft': lambda: QAgent(cfg, rng, True, False, True, 'IQL+AoI+Soft'),
        'corr_soft': lambda: QAgent(cfg, rng, False, True, True, 'IQL+Corr+Soft'),
        'iql_soft': lambda: QAgent(cfg, rng, False, False, True, 'IQL+Soft'),
        'aoi_scorr': lambda: QAgent(cfg, rng, True, True, False, 'IQL+AoI+SharedCorr', shared_corr=True),
        'aoi_soft05': lambda: QAgent(cfg, rng, True, False, True, 'IQL+AoI+Soft(0.05)', tau=0.05),
        'dsa_s': lambda: QAgent(cfg, rng, True, True, True, 'Full+SharedCorr', shared_corr=True),
        'iql_soft05': lambda: QAgent(cfg, rng, False, False, True, 'IQL+Soft', tau=0.05),
        'prop': lambda: QAgent(cfg, rng, True, False, True, 'SADO (proposed)', tau=0.05),
        'prop_corr': lambda: QAgent(cfg, rng, True, True, True, 'SADO+Corr', tau=0.05, shared_corr=True),
        'dsa': lambda: QAgent(cfg, rng, True, True, True, 'SACO (proposed)'),
    }[key]()

def run(key, seed, **over):
    cfg = dict(DEF); cfg.update(over)
    N, M = cfg['N'], cfg['M']
    caps = np.asarray(cfg['caps'], float)
    env_rng = np.random.default_rng(seed)
    pol = make_policy(key, cfg, np.random.default_rng(seed + 10_000))
    base_rate = env_rng.uniform(*cfg['rate'], size=(N, M))
    phase = np.zeros(N, int) if cfg['phase'] == 'sync' else env_rng.integers(0, cfg['B'], N)
    # queues: list of [remaining, task]
    lq = [deque() for _ in range(N)]; lback = np.zeros(N)
    sq = [deque() for _ in range(M)]; sback = np.zeros(M)
    transit = {}; transit_load = np.zeros(M)
    obs = np.zeros((N, M)); obs_t = np.zeros(N, int)
    T = cfg['T_train'] + cfg['T_eval'] if pol.learns else cfg['T_eval']
    t0_eval = T - cfg['T_eval']
    stats = dict(n=0, met=0, n_hp=0, met_hp=0, util=0., lat=[], energy=0., drop=0,
                 offl=0, choice=np.zeros(M + 1), herd=[], curve=[])
    win = dict(n=0, met=0); herd_cnt = np.zeros(M)

    def finish(task, t_done, dropped):
        dl_met = (not dropped) and t_done <= task['dl']
        w = cfg['w_hp'] if task['prio'] else cfg['w_np']
        r = w * (1.0 if dl_met else -1.0) - cfg['lam'] * task['E'] / cfg['e_norm']
        pol.feedback(task['info'], r, task.get('wait'))
        win['n'] += 1; win['met'] += dl_met
        if task['t0'] >= t0_eval:
            s = stats; s['n'] += 1; s['met'] += dl_met; s['util'] += (w if dl_met else 0.)
            s['energy'] += task['E']; s['drop'] += dropped
            if task['prio']: s['n_hp'] += 1; s['met_hp'] += dl_met
            if not dropped: s['lat'].append(t_done - task['t0'])

    def serve(q, cap, t, back):
        budget = cap; used = 0.
        while q and budget > 1e-9:
            task = q[0]
            if 'start' not in task:
                if t > task['dl']:           # expired before service -> drop
                    q.popleft(); back_dec = task['rem']; used += 0; task['wait'] = t - task['arr']
                    finish(task, t, True); back[0] -= back_dec; continue
                task['start'] = t; task['wait'] = t - task['arr']
            take = min(budget, task['rem']); task['rem'] -= take; budget -= take; back[0] -= take
            if task['rem'] <= 1e-9:
                q.popleft(); finish(task, t + (cap - budget) / cap, False)
        return

    for t in range(T):
        training = pol.learns and t < t0_eval
        if t % 1000 == 0:
            pol.set_phase(training, t / max(t0_eval, 1))
            if win['n']: stats['curve'].append((t, win['met'] / win['n']))
            win['n'] = win['met'] = 0
        # broadcasts
        upd = ((t - phase) % cfg['B']) == 0
        if upd.any():
            obs[upd] = sback; obs_t[upd] = t
        if t % cfg['B'] == 0 and t > 0:
            tot = herd_cnt.sum()
            if tot >= 5 and t >= t0_eval: stats['herd'].append(herd_cnt.max() / tot)
            herd_cnt[:] = 0
        # arrivals at servers
        for (m, task) in transit.pop(t, []):
            task['arr'] = t; sq[m].append(task); sback[m] += task['rem']; transit_load[m] -= task['rem']
        # new tasks
        gen = env_rng.random(N) < cfg['p']
        for n in np.flatnonzero(gen):
            c = env_rng.uniform(*cfg['cyc']); d = env_rng.uniform(*cfg['data'])
            hp = env_rng.random() < cfg['hp_frac']
            dl = env_rng.integers(*cfg['dl_hp']) if hp else env_rng.integers(*cfg['dl_np'])
            rate = base_rate[n] * np.exp(env_rng.normal(0, cfg['fade_sigma'], M) - cfg['fade_sigma'] ** 2 / 2)
            tx = d / rate
            ctx = dict(dev=n, cyc=c, tx=tx, slack=float(dl), prio=int(hp), local_backlog=lback[n],
                       obs_backlog=obs[n], aoi=t - obs_t[n], true_backlog_incl_transit=sback + transit_load)
            a, info = pol.act(ctx)
            task = dict(t0=t, dl=t + dl, prio=int(hp), rem=c, info=info)
            if a == 0:
                task['E'] = cfg['kappa'] * (cfg['f_local'] * 1e8) ** 2 * c * 1e6
                task['arr'] = t; lq[n].append(task); lback[n] += c
            else:
                m = a - 1; task['E'] = cfg['p_tx'] * tx[m] * 0.01
                ta = t + int(np.ceil(tx[m]))
                transit.setdefault(ta, []).append((m, task)); transit_load[m] += c
                herd_cnt[m] += 1
                if t >= t0_eval: stats['offl'] += 1
            if t >= t0_eval: stats['choice'][a] += 1
        # service
        for m in range(M):
            b = [sback[m]]; serve(sq[m], caps[m], t, b); sback[m] = b[0]
        for n in range(N):
            if lq[n]:
                b = [lback[n]]; serve(lq[n], cfg['f_local'], t, b); lback[n] = b[0]
    s = stats
    out = dict(policy=pol.name, key=key, seed=seed, n=s['n'],
               DSR=s['met'] / max(s['n'], 1), DSR_hp=s['met_hp'] / max(s['n_hp'], 1),
               utility=s['util'] / max(s['n'], 1),
               lat_ms=10 * float(np.mean(s['lat'])) if s['lat'] else np.nan,
               energy_mJ=1000 * s['energy'] / max(s['n'], 1), drop=s['drop'] / max(s['n'], 1),
               offload=s['offl'] / max(s['n'], 1), herd=float(np.mean(s['herd'])) if s['herd'] else np.nan,
               curve=s['curve'])
    out.update({k: v for k, v in over.items() if not isinstance(v, (list, tuple))})
    return out

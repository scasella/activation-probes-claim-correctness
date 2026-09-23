"""Grouped (by-biography) vs claim-level paired bootstrap for the FActScore test set.

The headline intervals resample claims, but the 742 test claims come from 24
biographies, so claims are not independent. This resamples whole biographies
(all of a biography's claims together) and reports both intervals side by side.
Run: python scripts/grouped_bootstrap_factscore.py

Inputs are the committed per-claim artifacts in scasella/activation-probes-claim-correctness
@309b8e3 (artifacts/runs/factscore_chatgpt_validation_eval.json). No model is run.
"""
import json, re, numpy as np
from sklearn.metrics import roc_auc_score

ROOT = __import__('pathlib').Path(__file__).resolve().parents[1]
e = json.load(open(ROOT / 'artifacts/runs/factscore_chatgpt_validation_eval.json'))
rp = e['residual_probe']['test']
ids = rp['claim_ids']; y = np.array(rp['y_true'], float)
probe = np.array([rp['scores'][i] for i in ids], float) if isinstance(rp['scores'], dict) else np.array(rp['scores'], float)
sr = e['llama_self_report']['test']['scores']
self_ = np.array([sr[i] for i in ids], float)
bio = np.array([re.sub(r'-fact-\d+$', '', i) for i in ids])
ubio = np.unique(bio)
print('claims', len(ids), 'biographies', len(ubio), 'true', int(y.sum()))

def stats(idx):
    yy = y[idx]
    a_p = roc_auc_score(yy, probe[idx]); a_s = roc_auc_score(yy, self_[idx])
    b_p = np.mean((probe[idx] - yy) ** 2); b_s = np.mean((self_[idx] - yy) ** 2)
    base = yy.mean(); b_c = base * (1 - base)
    return a_p, a_s, a_p - a_s, b_p, b_s, b_c

pt = stats(np.arange(len(y)))
print('point  probe %.4f  self %.4f  delta %.4f | brier probe %.4f self %.4f const %.4f' % pt)

rng = np.random.default_rng(20260424)
B = 10000
groups = {b: np.where(bio == b)[0] for b in ubio}
res = {'claim': [], 'bio': []}
for _ in range(B):
    res['claim'].append(stats(rng.integers(0, len(y), len(y))))
    pick = rng.choice(ubio, len(ubio), replace=True)
    res['bio'].append(stats(np.concatenate([groups[b] for b in pick])))
names = ['probe AUROC', 'self AUROC', 'delta AUROC', 'probe Brier', 'self Brier', 'base-rate Brier']
out = {'point': dict(zip(names, pt))}
for k, v in res.items():
    v = np.array(v); lo, hi = np.percentile(v, [2.5, 97.5], axis=0)
    out[k] = {n: [round(a, 4), round(b, 4)] for n, a, b in zip(names, lo, hi)}
    # Brier: probe minus base-rate constant, per resample
    d = v[:, 3] - v[:, 5]; out[k]['probe minus base-rate Brier'] = [round(x, 4) for x in np.percentile(d, [2.5, 97.5])]
out['n_resamples'] = B; out['seed'] = 20260424
print(json.dumps(out, indent=1))
json.dump(out, open(ROOT / 'artifacts/runs/factscore_grouped_bootstrap_ci.json', 'w'), indent=1)

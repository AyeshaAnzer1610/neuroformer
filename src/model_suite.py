# -*- coding: utf-8 -*-
"""
SHARED MODEL SUITE  (used by 17_permutation_suite / 18_bootstrap_ci / 19_ds003478_suite)
========================================================================================
Single source of truth for the ~14-model benchmark, so the permutation test,
the per-model confidence intervals, and the ds003478 replication all evaluate
the SAME models under the SAME leakage-free subject-level CV.

HONESTY NOTE
------------
The authors' original NeuroFormer training pipeline is NOT in this repo (see
docs/resubmission/Reviewer-Response-Worklog.md). These are *faithful, standard
reference implementations* of each model family (GMU: Arevalo 2017; TFN: Zadeh
2017; LMF: Liu 2018; cross-modal attention as in src/11_fusion_models.py;
NeuroFormer encoder from src/advanced_multimodal.py). They are run on the real
locked N=92 features. Numbers you obtain are the honest output of THIS suite;
they are not asserted to be bit-identical to the authors' external pipeline.
Report them as produced by these scripts.

Everything is CPU-friendly and seeded. Nothing is fabricated here: this module
only DEFINES models + evaluation; the driver scripts run them on real data.

Cohorts
-------
MIPDB : locked EEG+eye cohort in cohort_final.txt (data/aligned_resting +
        per-window EEG in data/resting_features).
ds003478 : EEG-only windows in data/ds2_features (extracted by
        ds2_extract_and_benchmark.py). Eye/behaviour do not exist there, so
        multimodal-fusion models are marked N/A rather than run degenerately.
"""
import os, glob, warnings
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.set_num_threads(int(os.environ.get('SUITE_THREADS', '4')))
    torch.manual_seed(0)
    try:                                            # best-effort CPU determinism
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    _HAS_TORCH = True
except Exception:                                   # pragma: no cover
    _HAS_TORCH = False


# ---------------------------------------------------------------- data bundle
@dataclass
class Bundle:
    """All representations a model might consume, aligned by subject index."""
    subjects: list
    y: np.ndarray                     # (N,) binary
    eeg: np.ndarray                   # (N, Keeg) subject-level (top-var reduced)
    eye: np.ndarray                   # (N, Keye) subject-level ; empty if none
    seq: np.ndarray                   # (N, T, Keeg) padded per-window EEG
    seq_len: np.ndarray               # (N,) valid window counts
    age: np.ndarray                   # (N,)
    sex: np.ndarray                   # (N,)
    has_eye: bool                     # False for ds003478
    var_idx: np.ndarray = field(default=None)  # top-var feature indices used

    @property
    def N(self): return len(self.y)


def _topvar_idx(X, k):
    v = np.nan_to_num(X).var(0)
    return np.argsort(v)[::-1][:min(k, X.shape[1])]


def load_mipdb(k_eeg=128, cohort_file='cohort_final.txt',
               align='data/aligned_resting', feat_dir='data/resting_features',
               labels='data/labels.csv'):
    """Locked MIPDB EEG+eye cohort with per-window EEG sequences."""
    A = Path(align)
    coh = [l.strip() for l in open(cohort_file) if l.strip()]
    cohset = set(coh)
    lab = pd.read_csv(labels, index_col='ID')
    man = pd.read_csv(A / 'manifest.csv')
    subs_all = man['subject'].tolist()
    pos = {s: i for i, s in enumerate(subs_all)}

    Xeeg_all = np.nan_to_num(np.load(A / 'X_eeg.npy'))
    Xeye_all = np.nan_to_num(np.load(A / 'X_eye.npy'))

    # keep only locked-cohort subjects, in cohort order, that exist in manifest
    subjects = [s for s in coh if s in pos]
    sel = [pos[s] for s in subjects]
    Xeeg = Xeeg_all[sel]
    Xeye = Xeye_all[sel]
    y = np.array([int(lab.loc[s, 'DX_Status'] == 1) for s in subjects])
    age = np.array([float(lab.loc[s, 'Age']) for s in subjects], float)
    sex = np.array([1.0 if str(lab.loc[s, 'Sex']).upper() in ('M', '1', 'MALE')
                    else 0.0 for s in subjects], float)

    var_idx = _topvar_idx(Xeeg, k_eeg)
    Xeeg_r = Xeeg[:, var_idx]

    # per-window EEG sequences (same top-var reduction), for temporal models
    win = {}
    for f in glob.glob(str(Path(feat_dir) / '*.npz')):
        s = os.path.basename(f).split('_')[0]
        if s in cohset:
            W = np.nan_to_num(np.load(f, allow_pickle=True)['features'])
            if W.ndim == 2 and W.shape[0] > 0:
                win[s] = W[:, var_idx].astype(np.float32)
    T = max((win[s].shape[0] for s in subjects if s in win), default=1)
    seq = np.zeros((len(subjects), T, len(var_idx)), np.float32)
    seq_len = np.ones(len(subjects), int)
    for i, s in enumerate(subjects):
        if s in win:
            w = win[s][:T]
            seq[i, :w.shape[0]] = w
            seq_len[i] = w.shape[0]
        else:                              # fall back to subject-mean as 1 window
            seq[i, 0] = Xeeg_r[i]
    return Bundle(subjects, y, Xeeg_r, Xeye, seq, seq_len, age, sex,
                  has_eye=True, var_idx=var_idx)


def load_ds003478(k_eeg=128, feat_dir='data/ds2_features',
                  root='data/ds003478', bdi_cut=14):
    """EEG-only ds003478 cohort (subject-mean + per-window sequences)."""
    part = pd.read_csv(Path(root) / 'participants.tsv', sep='\t').set_index('participant_id')
    subjects, means, seqs, y, age, sex = [], [], [], [], [], []
    for f in sorted(glob.glob(str(Path(feat_dir) / 'sub-*.npz'))):
        sid = os.path.basename(f)[:-4]
        if sid not in part.index:
            continue
        try:
            bdi = float(part.loc[sid, 'BDI'])
        except Exception:
            continue
        if np.isnan(bdi):
            continue
        W = np.nan_to_num(np.load(f)['features'])
        if W.ndim != 2 or W.shape[0] == 0:
            continue
        subjects.append(sid)
        means.append(W.mean(0))
        seqs.append(W.astype(np.float32))
        y.append(int(bdi >= bdi_cut))
        try: age.append(float(part.loc[sid, 'age']))
        except Exception: age.append(np.nan)
        try: sex.append(float(part.loc[sid, 'sex']))
        except Exception: sex.append(np.nan)

    # align feature length across subjects (64 vs 66/67 ch -> truncate to min)
    minf = min(m.shape[0] for m in means)
    means = np.array([m[:minf] for m in means], np.float32)
    y = np.array(y); age = np.array(age, float); sex = np.array(sex, float)
    if np.isnan(age).any(): age[np.isnan(age)] = np.nanmedian(age)
    if np.isnan(sex).any(): sex[np.isnan(sex)] = np.nanmedian(sex)

    var_idx = _topvar_idx(means, k_eeg)
    eeg_r = means[:, var_idx]
    T = max(s.shape[0] for s in seqs)
    seq = np.zeros((len(subjects), T, len(var_idx)), np.float32)
    seq_len = np.ones(len(subjects), int)
    for i, s in enumerate(seqs):
        w = s[:, :minf][:, var_idx][:T]
        seq[i, :w.shape[0]] = w
        seq_len[i] = w.shape[0]
    empty = np.zeros((len(subjects), 0), np.float32)
    return Bundle(subjects, y, eeg_r, empty, seq, seq_len, age, sex,
                  has_eye=False, var_idx=var_idx)


# ---------------------------------------------------------------- torch models
if _HAS_TORCH:
    class GMU(nn.Module):
        """Gated Multimodal Unit (Arevalo et al., 2017)."""
        def __init__(s, de, dy, h=32):
            super().__init__()
            s.he = nn.Linear(de, h); s.hy = nn.Linear(dy, h)
            s.z = nn.Linear(de + dy, h)
            s.out = nn.Linear(h, 1)
        def forward(s, e, yv):
            he = torch.tanh(s.he(e)); hy = torch.tanh(s.hy(yv))
            z = torch.sigmoid(s.z(torch.cat([e, yv], -1)))
            return s.out(z * he + (1 - z) * hy).squeeze(-1)

    class CrossModalTF(nn.Module):
        """Cross-modal attention fusion (matches src/11_fusion_models.py CrossAttn)."""
        def __init__(s, de, dy, h=32):
            super().__init__()
            s.pe = nn.Linear(de, h); s.py = nn.Linear(dy, h)
            s.q = nn.Linear(h, h); s.k = nn.Linear(h, h); s.v = nn.Linear(h, h)
            s.fc = nn.Sequential(nn.Linear(h * 2, h), nn.ReLU(), nn.Dropout(0.4), nn.Linear(h, 1))
        def forward(s, e, yv):
            te = torch.relu(s.pe(e)); ty = torch.relu(s.py(yv))
            att = torch.softmax((s.q(te) * s.k(ty)).sum(-1, keepdim=True) / (te.shape[-1] ** 0.5), 0)
            ctx = att * s.v(ty)
            return s.fc(torch.cat([te, ctx], -1)).squeeze(-1)

    class TFN(nn.Module):
        """Tensor Fusion Network (Zadeh et al., 2017): outer product w/ 1-augment."""
        def __init__(s, de, dy, h=8):
            super().__init__()
            s.pe = nn.Sequential(nn.Linear(de, h), nn.ReLU(), nn.Dropout(0.4))
            s.py = nn.Sequential(nn.Linear(dy, h), nn.ReLU(), nn.Dropout(0.4))
            s.out = nn.Linear((h + 1) * (h + 1), 1)
        def forward(s, e, yv):
            a = torch.cat([s.pe(e), torch.ones(e.shape[0], 1)], -1)
            b = torch.cat([s.py(yv), torch.ones(yv.shape[0], 1)], -1)
            fus = torch.bmm(a.unsqueeze(2), b.unsqueeze(1)).flatten(1)
            return s.out(fus).squeeze(-1)

    class LMF(nn.Module):
        """Low-rank Multimodal Fusion (Liu et al., 2018)."""
        def __init__(s, de, dy, h=8, rank=4):
            super().__init__()
            s.pe = nn.Sequential(nn.Linear(de, h), nn.ReLU(), nn.Dropout(0.4))
            s.py = nn.Sequential(nn.Linear(dy, h), nn.ReLU(), nn.Dropout(0.4))
            s.rank = rank
            s.fe = nn.Parameter(torch.randn(rank, h + 1) * 0.1)
            s.fy = nn.Parameter(torch.randn(rank, h + 1) * 0.1)
            s.w = nn.Parameter(torch.randn(1, rank) * 0.1)
            s.b = nn.Parameter(torch.zeros(1))
        def forward(s, e, yv):
            a = torch.cat([s.pe(e), torch.ones(e.shape[0], 1)], -1)
            b = torch.cat([s.py(yv), torch.ones(yv.shape[0], 1)], -1)
            za = a @ s.fe.t(); zb = b @ s.fy.t()          # (B, rank)
            return ((za * zb) @ s.w.t()).squeeze(-1) + s.b

    class CNN1D(nn.Module):
        """Temporal 1D-CNN over the per-window EEG feature sequence."""
        def __init__(s, feat, h=32):
            super().__init__()
            s.net = nn.Sequential(
                nn.Conv1d(feat, h, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(h),
                nn.Conv1d(h, h, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
            s.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(h, 1))
        def forward(s, x):                     # x: (B, T, feat)
            return s.fc(s.net(x.transpose(1, 2)).squeeze(-1)).squeeze(-1)

    class LSTMseq(nn.Module):
        def __init__(s, feat, h=32):
            super().__init__()
            s.rnn = nn.LSTM(feat, h, batch_first=True)
            s.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(h, 1))
        def forward(s, x):
            o, _ = s.rnn(x)
            return s.fc(o.mean(1)).squeeze(-1)

    class TCN(nn.Module):
        """Dilated temporal conv net (Bai et al., 2018), 2 blocks."""
        def __init__(s, feat, h=32):
            super().__init__()
            s.b1 = nn.Sequential(nn.Conv1d(feat, h, 3, padding=1, dilation=1), nn.ReLU())
            s.b2 = nn.Sequential(nn.Conv1d(h, h, 3, padding=2, dilation=2), nn.ReLU())
            s.pool = nn.AdaptiveAvgPool1d(1)
            s.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(h, 1))
        def forward(s, x):
            z = s.b2(s.b1(x.transpose(1, 2)))
            return s.fc(s.pool(z).squeeze(-1)).squeeze(-1)


def _train_torch(model, feed_train, ytr, feed_test, epochs, lr=1e-3, wd=1e-3, seed=0):
    torch.manual_seed(seed)
    pw = torch.tensor([(ytr == 0).sum() / max(1, (ytr == 1).sum())], dtype=torch.float32)
    lf = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), lr, weight_decay=wd)
    yt = torch.tensor(ytr, dtype=torch.float32)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        out = model(*feed_train)
        lf(out, yt).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(*feed_test)).numpy()


# ---------------------------------------------------------------- model registry
# Each entry: fit_predict(bundle, tr, te, y, seed, cfg) -> proba_te (len(te),)
# `needs` declares required modalities; ds003478 driver skips models needing 'eye'.

def _scaled(Xtr, Xte):
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte)

def _concat(b, tr, te):
    if b.has_eye and b.eye.shape[1] > 0:
        Xtr = np.column_stack([b.eeg[tr], b.eye[tr]])
        Xte = np.column_stack([b.eeg[te], b.eye[te]])
    else:
        Xtr, Xte = b.eeg[tr], b.eeg[te]
    return _scaled(Xtr, Xte)


def _sk(clf_factory):
    def fp(b, tr, te, y, seed, cfg):
        Xtr, Xte = _concat(b, tr, te)
        clf = clf_factory()
        clf.fit(Xtr, y[tr])
        return clf.predict_proba(Xte)[:, 1]
    return fp

def _late(b, tr, te, y, seed, cfg):
    pe_tr, pe_te = _scaled(b.eeg[tr], b.eeg[te])
    ce = LogisticRegression(max_iter=2000, class_weight='balanced').fit(pe_tr, y[tr])
    p = ce.predict_proba(pe_te)[:, 1]
    if b.has_eye and b.eye.shape[1] > 0:
        py_tr, py_te = _scaled(b.eye[tr], b.eye[te])
        cy = LogisticRegression(max_iter=2000, class_weight='balanced').fit(py_tr, y[tr])
        p = (p + cy.predict_proba(py_te)[:, 1]) / 2
    return p

def _demographics(b, tr, te, y, seed, cfg):
    Xtr, Xte = _scaled(np.column_stack([b.age[tr], b.sex[tr]]),
                        np.column_stack([b.age[te], b.sex[te]]))
    clf = LogisticRegression(max_iter=2000, class_weight='balanced').fit(Xtr, y[tr])
    return clf.predict_proba(Xte)[:, 1]

def _torch_fusion(net_factory):
    def fp(b, tr, te, y, seed, cfg):
        e_tr, e_te = _scaled(b.eeg[tr], b.eeg[te])
        y_tr, y_te = _scaled(b.eye[tr], b.eye[te])
        net = net_factory(e_tr.shape[1], y_tr.shape[1])
        feed_tr = (torch.tensor(e_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
        feed_te = (torch.tensor(e_te, dtype=torch.float32), torch.tensor(y_te, dtype=torch.float32))
        return _train_torch(net, feed_tr, y[tr], feed_te, cfg['epochs'],
                            lr=cfg.get('lr', 1e-3), wd=cfg.get('wd', 1e-3), seed=seed)
    return fp

def _torch_temporal(net_factory):
    def fp(b, tr, te, y, seed, cfg):
        flat = b.seq.reshape(-1, b.seq.shape[-1])
        sc = StandardScaler().fit(flat[np.repeat(np.isin(np.arange(b.N), tr), b.seq.shape[1])])
        Str = sc.transform(b.seq[tr].reshape(-1, b.seq.shape[-1])).reshape(len(tr), b.seq.shape[1], -1)
        Ste = sc.transform(b.seq[te].reshape(-1, b.seq.shape[-1])).reshape(len(te), b.seq.shape[1], -1)
        net = net_factory(b.seq.shape[-1])
        feed_tr = (torch.tensor(Str, dtype=torch.float32),)
        feed_te = (torch.tensor(Ste, dtype=torch.float32),)
        return _train_torch(net, feed_tr, y[tr], feed_te, cfg['epochs'],
                            lr=cfg.get('lr', 1e-3), wd=cfg.get('wd', 1e-3), seed=seed)
    return fp

def _neuroformer(b, tr, te, y, seed, cfg):
    # Uses the authors' in-repo encoder (src/advanced_multimodal.py). NOTE: as of
    # this writing that create_model() raises a shape error (FrequencyAwareAttention
    # collapses hidden_dim -> band_dim). If it raises, the drivers report NeuroFormer
    # with an ERROR status rather than substituting a different architecture under its
    # name -- wire your working NeuroFormer into create_model to get a real number.
    from advanced_multimodal import create_model
    e_tr, e_te = _scaled(b.eeg[tr], b.eeg[te])
    y_tr, y_te = _scaled(b.eye[tr], b.eye[te])
    beh_tr = np.zeros((len(tr), 6), np.float32); beh_te = np.zeros((len(te), 6), np.float32)
    torch.manual_seed(seed)
    net = create_model(eeg_dim=e_tr.shape[1], eye_dim=y_tr.shape[1], beh_dim=6)
    pw = (y[tr] == 0).sum() / max(1, (y[tr] == 1).sum())
    lf = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pw], dtype=torch.float32))
    opt = torch.optim.Adam(net.parameters(), cfg.get('lr', 1e-3), weight_decay=cfg.get('wd', 1e-4))
    et = torch.tensor(e_tr, dtype=torch.float32); yt = torch.tensor(y_tr, dtype=torch.float32)
    bt = torch.tensor(beh_tr); lab = torch.tensor(y[tr], dtype=torch.long)
    net.train()
    for _ in range(cfg['epochs']):
        opt.zero_grad(); logits, _ = net(et, yt, bt); lf(logits, lab).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        logits, _ = net(torch.tensor(e_te, dtype=torch.float32),
                        torch.tensor(y_te, dtype=torch.float32), torch.tensor(beh_te))
        return torch.softmax(logits, -1)[:, 1].numpy()


def build_registry():
    """Ordered list of (name, needs, fit_predict). 14 models."""
    reg = [
        ('LogReg (concat)',        (),      _sk(lambda: LogisticRegression(max_iter=2000, class_weight='balanced'))),
        ('SVM-RBF (concat)',       (),      _sk(lambda: SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=0))),
        ('RandomForest (concat)',  (),      _sk(lambda: RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=0))),
        ('MLP early-fusion',       (),      _sk(lambda: MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=0))),
        ('Late-fusion ensemble',   (),      _late),
        ('1D-CNN (temporal)',      (),      _torch_temporal(lambda f: CNN1D(f)) if _HAS_TORCH else None),
        ('LSTM (temporal)',        (),      _torch_temporal(lambda f: LSTMseq(f)) if _HAS_TORCH else None),
        ('TCN (temporal)',         (),      _torch_temporal(lambda f: TCN(f)) if _HAS_TORCH else None),
        ('Demographics (age+sex)', (),      _demographics),
        ('GMU (gated fusion)',        ('eye',), _torch_fusion(lambda de, dy: GMU(de, dy)) if _HAS_TORCH else None),
        ('Cross-modal Transformer',   ('eye',), _torch_fusion(lambda de, dy: CrossModalTF(de, dy)) if _HAS_TORCH else None),
        ('Tensor Fusion Network',     ('eye',), _torch_fusion(lambda de, dy: TFN(de, dy)) if _HAS_TORCH else None),
        ('Low-rank Multimodal Fusion',('eye',), _torch_fusion(lambda de, dy: LMF(de, dy)) if _HAS_TORCH else None),
        ('NeuroFormer (3-modality)',  ('eye',), _neuroformer if _HAS_TORCH else None),
    ]
    return reg


# ---------------------------------------------------------------- evaluation
def cv_perfold_auc(fit_predict, b, y, n_splits=5, n_repeats=10, seed=0, cfg=None):
    """Repeated stratified subject-level CV -> array of per-fold AUCs."""
    cfg = cfg or {'epochs': 60}
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    aucs = []
    for i, (tr, te) in enumerate(cv.split(np.zeros(b.N), y)):
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        p = fit_predict(b, tr, te, y, seed + i, cfg)
        aucs.append(roc_auc_score(y[te], p))
    return np.array(aucs)


def cv_oof(fit_predict, b, y, n_splits=5, seed=0, cfg=None):
    """Single stratified CV -> out-of-fold probabilities aligned to subjects."""
    cfg = cfg or {'epochs': 60}
    oof = np.full(b.N, np.nan)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for i, (tr, te) in enumerate(cv.split(np.zeros(b.N), y)):
        oof[te] = fit_predict(b, tr, te, y, seed + i, cfg)
    return oof


# ------------------------- helpers for extended schemes (scripts 20-23) -------
def subset(b, idx):
    """Return a Bundle restricted to subject indices `idx` (for matched subsampling)."""
    idx = np.asarray(idx)
    return Bundle(subjects=[b.subjects[i] for i in idx], y=b.y[idx],
                  eeg=b.eeg[idx], eye=b.eye[idx], seq=b.seq[idx],
                  seq_len=b.seq_len[idx], age=b.age[idx], sex=b.sex[idx],
                  has_eye=b.has_eye, var_idx=b.var_idx)


def X_concat(b):
    """Subject-level concat feature matrix (eeg[+eye]); for sklearn nested/632 loops."""
    if b.has_eye and b.eye.shape[1] > 0:
        return np.column_stack([b.eeg, b.eye])
    return b.eeg


def cv_oof_splitter(fit_predict, b, y, splitter, cfg=None, groups=None, seed=0):
    """Out-of-fold probabilities using ANY sklearn splitter (LOSO, custom, ...)."""
    cfg = cfg or {'epochs': 60}
    oof = np.full(b.N, np.nan)
    it = (splitter.split(np.zeros(b.N), y, groups) if groups is not None
          else splitter.split(np.zeros(b.N), y))
    for i, (tr, te) in enumerate(it):
        oof[te] = fit_predict(b, tr, te, y, seed + i, cfg)
    return oof


def window_concat(b):
    """Per-window concat matrix from the bundle: (n_win, Keeg[+Keye]), subject group,
    window label. Used for leaky (window-level) vs honest (group) CV comparisons."""
    rows, grp, yw = [], [], []
    for i in range(b.N):
        for w in range(int(b.seq_len[i])):
            feat = b.seq[i, w]
            if b.has_eye and b.eye.shape[1] > 0:
                feat = np.concatenate([feat, b.eye[i]])
            rows.append(feat); grp.append(i); yw.append(b.y[i])
    return np.array(rows, np.float32), np.array(grp), np.array(yw)


def sk_grids():
    """Estimator + hyperparameter grid for the tunable sklearn models (nested CV).
    Torch models are intentionally excluded from nested CV (documented in runbook)."""
    return {
        'LogReg (concat)':       (LogisticRegression(max_iter=2000, class_weight='balanced'),
                                  {'C': [0.01, 0.1, 1, 10]}),
        'SVM-RBF (concat)':      (SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=0),
                                  {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}),
        'RandomForest (concat)': (RandomForestClassifier(class_weight='balanced', random_state=0),
                                  {'n_estimators': [200, 400], 'max_depth': [3, 5, None]}),
        'MLP early-fusion':      (MLPClassifier(max_iter=500, random_state=0),
                                  {'alpha': [1e-4, 1e-3, 1e-2], 'hidden_layer_sizes': [(32,), (64,)]}),
    }


def sk_estimators():
    """Untuned sklearn estimator factories for the concat models (window-level
    honest-vs-leaky demonstration in script 21)."""
    return {
        'LogReg (concat)':       lambda: LogisticRegression(max_iter=2000, class_weight='balanced'),
        'SVM-RBF (concat)':      lambda: SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=0),
        'RandomForest (concat)': lambda: RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=0),
        'MLP early-fusion':      lambda: MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=0),
    }


def torch_grids():
    """cfg-override grids for the torch models, so nested CV can tune them through
    the fit_predict(cfg) interface (weight-decay; the main small-N regularizer).
    Kept tiny for tractability -- widen if you have the compute."""
    wd = {'wd': [1e-4, 1e-3, 1e-2]}
    return {
        '1D-CNN (temporal)': wd, 'LSTM (temporal)': wd, 'TCN (temporal)': wd,
        'GMU (gated fusion)': wd, 'Cross-modal Transformer': wd,
        'Tensor Fusion Network': wd, 'Low-rank Multimodal Fusion': wd,
        'NeuroFormer (3-modality)': wd,
    }


# ------------------- canonical OOF (single source of truth) -------------------
# So the benchmark point AUC, DeLong, and calibration all use the SAME out-of-fold
# predictions instead of recomputing them independently (which diverged because
# SVC's Platt calibration and torch CPU ops are not bit-reproducible). Scripts
# 18/22 load this file; script 24 produces it.
CANON_OOF = 'results/per_subject_oof.csv'

def compute_oof_all(b, reg, cfg=None, seed=0):
    """dict model_name -> OOF probability vector, for every model in reg."""
    cfg = cfg or {'epochs': 60}
    return {name: cv_oof(fp, b, b.y, seed=seed, cfg=cfg) for name, _, fp in reg}

def save_oof(b, oof, path=CANON_OOF):
    df = pd.DataFrame({'subject': b.subjects, 'group': b.y})
    for k, v in oof.items():
        df[k] = np.round(v, 6)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path

def load_oof(b, path=CANON_OOF):
    """Load the canonical OOF; asserts subject alignment with the bundle."""
    df = pd.read_csv(path)
    if list(df['subject']) != list(b.subjects):
        raise ValueError('per_subject_oof.csv subjects do not match bundle order; '
                         'regenerate with src/24_canonical_oof.py')
    return {c: df[c].values for c in df.columns if c not in ('subject', 'group')}

def get_oof(b, reg, cfg=None, path=CANON_OOF):
    """Load canonical OOF if it covers all models in reg; else compute (no save)."""
    names = [n for n, _, _ in reg]
    try:
        cached = load_oof(b, path)
        if all(n in cached for n in names):
            return {n: cached[n] for n in names}
    except (FileNotFoundError, ValueError):
        pass
    return compute_oof_all(b, reg, cfg)

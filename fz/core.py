"""Core helpers: differentiable TabPFN-3.5 classifier + soft-label relaxation.

Soft labels: TabPFN-3.5 embeds class labels via nn.Embedding(y.long()) and the
decoder builds F.one_hot(y.long()); both cut gradients w.r.t. y. When
SOFT_LABELS is enabled we monkeypatch those two spots so that a float label
y in [0, K-1] (binary: y in [0,1]) is treated as a convex mixture of the two
neighbouring classes. At integer y this is numerically identical to the
original model.
"""
from __future__ import annotations

import contextlib
import os

import numpy as np
import torch
import torch.nn.functional as F

from tabpfn import TabPFNClassifier
from tabpfn.architectures import tabpfn_v3_5 as v35
from tabpfn.constants import ModelVersion

SOFT = {"on": False}
CAPTURE = {"on": False, "attn": None}


def _soft_onehot(y: torch.Tensor, k: int) -> torch.Tensor:
    """Linear interpolation one-hot for float labels (differentiable)."""
    # clamp in value only; keep identity gradient (torch.clamp zeroes the
    # gradient at the lower boundary y=0, which is exactly where labels live)
    y = y + (y.clamp(0, k - 1) - y).detach()
    # lower class index clamped to k-2 so the top class (y=k-1) still has a
    # non-zero derivative (frac=1 on the [k-2, k-1] segment).
    lo = y.floor().clamp(max=k - 2).detach()
    frac = y - lo
    lo_l = lo.long()
    hi_l = lo_l + 1
    oh = F.one_hot(lo_l, k).to(y.dtype) * (1 - frac)[..., None]
    oh = oh + F.one_hot(hi_l, k).to(y.dtype) * frac[..., None]
    return oh


_orig_emb_forward = v35.TrainableOrthogonalEmbedding.forward
_orig_dec_forward = v35.ManyClassDecoder.forward


def _emb_forward(self, x):
    if not SOFT["on"]:
        return _orig_emb_forward(self, x)
    W = self.embedding.weight
    oh = _soft_onehot(x.to(W.dtype), W.shape[0])
    return oh @ W


def _dec_forward(self, train_keys_BNHD, test_embeddings_BME, targets_BN, *, num_present_classes):
    if not SOFT["on"]:
        return _orig_dec_forward(
            self, train_keys_BNHD, test_embeddings_BME, targets_BN,
            num_present_classes=num_present_classes,
        )
    B, M, _ = test_embeddings_BME.shape
    q_BMHD, train_keys_BNHD = self._project_queries(train_keys_BNHD, test_embeddings_BME)
    oh = _soft_onehot(targets_BN.to(q_BMHD.dtype), num_present_classes)
    if CAPTURE["on"]:
        CAPTURE["attn"] = self.attention_weights(train_keys_BNHD, test_embeddings_BME)
    oh_BNHT = oh.unsqueeze(2).expand(-1, -1, self.num_heads, -1).contiguous()
    out = v35._chunked_class_attention(
        q_BMHD, train_keys_BNHD, oh_BNHT, softmax_scaling_layer=self.softmax_scaling_layer
    ).mean(2)
    missing = self.max_num_classes - num_present_classes
    if missing:
        out = F.pad(out, (0, missing))
    out = out.transpose(0, 1)
    return torch.log(torch.clamp(out, min=1e-5) + 3e-5)


v35.TrainableOrthogonalEmbedding.forward = _emb_forward
v35.ManyClassDecoder.forward = _dec_forward


@contextlib.contextmanager
def soft_labels(on: bool = True):
    old = SOFT["on"]
    SOFT["on"] = on
    try:
        yield
    finally:
        SOFT["on"] = old


def make_diff_clf(n_classes: int = 2, n_estimators: int = 1, seed: int = 0, device="cuda",
                  fingerprint: bool = False):
    # NOTE: the default fingerprint feature hashes each row with a salt equal to
    # n_rows*n_cols, so adding/removing ANY row re-randomises every fingerprint
    # (and it is non-differentiable). We disable it for forensics.
    clf = TabPFNClassifier.create_default_for_version(
        ModelVersion.V3_5,
        device=device,
        n_estimators=n_estimators,
        random_state=seed,
        inference_precision=torch.float32,
        differentiable_input=True,
        ignore_pretraining_limits=True,
        fit_mode="fit_preprocessors",
        inference_config={"FINGERPRINT_FEATURE": fingerprint},
    )
    clf.n_classes_ = n_classes
    return clf


def proba(clf, Xtr: torch.Tensor, ytr: torch.Tensor, Xte: torch.Tensor) -> torch.Tensor:
    """Differentiable P(y|x) for Xte given context (Xtr, ytr)."""
    clf.fit_with_differentiable_input(Xtr, ytr)
    return clf.forward(Xte, use_inference_mode=True)


def make_std_clf(n_estimators: int = 1, seed: int = 0, device="cuda", **kw):
    return TabPFNClassifier.create_default_for_version(
        ModelVersion.V3_5, device=device, n_estimators=n_estimators,
        random_state=seed, ignore_pretraining_limits=True, **kw,
    )

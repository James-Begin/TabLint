"""Check which y paths carry autograd."""
import numpy as np, torch
from fz import core, data
from tabpfn.architectures import tabpfn_v3_5 as v35
dev="cuda"; rng=np.random.default_rng(0)
X,y=data.load("credit-g"); Xtr,Xte,ytr,yte=data.split(X,y,seed=0)
idx=rng.choice(len(ytr),300,replace=False)
Xc=torch.tensor(Xtr[idx],device=dev); yc=torch.tensor(ytr[idx],device=dev,dtype=torch.float32)
Xt=torch.tensor(Xte[:1],device=dev)
clf=core.make_diff_clf(seed=0)
fe, fd = v35.TrainableOrthogonalEmbedding.forward, v35.ManyClassDecoder.forward
def fe2(self,x):
    print("emb in", tuple(x.shape), x.dtype, x.requires_grad, "vals", x.flatten()[:5].tolist()); return fe(self,x)
def fd2(self,k,t,targ,**kw):
    print("dec targets", tuple(targ.shape), targ.dtype, targ.requires_grad, targ.flatten()[:5].tolist()); return fd(self,k,t,targ,**kw)
v35.TrainableOrthogonalEmbedding.forward=fe2; v35.ManyClassDecoder.forward=fd2
print("y first5", yc[:5].tolist())
with core.soft_labels(True):
    ya=yc.clone().requires_grad_(True)
    p=core.proba(clf,Xc,ya,Xt)[0,1]; p.backward(); print("grad first5", ya.grad[:5].tolist())

import json, time
import torch
from torch import nn
from teenie.data import make_dataset, split_dataset, Op
from teenie.model import RecursiveArithModel

torch.set_num_threads(1)
device='cuda'
ds=make_dataset(10,tuple(Op)); tr,va=split_dataset(ds,seed=0)
train=[x[tr.indices].to(device) for x in ds.tensors]; val=[x[va.indices].to(device) for x in ds.tensors]
class Numeric(nn.Module):
 def __init__(self,steps):
  super().__init__(); self.steps=steps
  self.a=nn.Linear(1,64); self.b=nn.Linear(1,64); self.op=nn.Embedding(3,64)
  self.cell=nn.Sequential(nn.Linear(193,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh())
  self.head=nn.Linear(64,1)
 def forward(self,a,b,op):
  ea=self.a(a.float()[:,None]/10); eb=self.b(b.float()[:,None]/10); h=ea+eb+self.op(op)
  for t in range(self.steps): h=self.cell(torch.cat([h,ea,eb,h.new_full((len(h),1),t)],-1))
  return self.head(h).squeeze(-1)*100
for kind,steps,epochs in [('categorical',4,1000),('numeric',1,5000),('numeric',4,5000)]:
 torch.manual_seed(0)
 m=(RecursiveArithModel(10,128,steps) if kind=='categorical' else Numeric(steps)).to(device)
 opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
 best=0; start=time.time()
 for e in range(1,epochs+1):
  opt.zero_grad(set_to_none=True)
  if kind=='categorical': out=m(*train[:3])[0][:,-1]; loss=nn.functional.cross_entropy(out,train[3]+100)
  else: out=m(*train[:3]); loss=nn.functional.mse_loss(out/100,train[3].float()/100)
  loss.backward(); opt.step()
  if e%500==0:
   with torch.no_grad():
    out=m(*val[:3]); pred=out[0][:,-1].argmax(-1)-100 if kind=='categorical' else out.round().long()
    acc=(pred==val[3]).float().mean().item(); best=max(best,acc)
    print(json.dumps(dict(kind=kind,steps=steps,epoch=e,loss=loss.item(),val_acc=acc,best=best,per_op={op.name:(pred[val[2]==op]==val[3][val[2]==op]).float().mean().item() for op in Op},seconds=time.time()-start)),flush=True)

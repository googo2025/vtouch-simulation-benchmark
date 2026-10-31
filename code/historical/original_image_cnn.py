"""Portable GPU CNN on untouched full-camera frames, no truth contact crop.

All four regions and both loads of a coordinate are held out together.
Fixed epochs/seeds; final weights, not test-selected checkpoints. One source,
12 developed locations and assumed optics/sensor: exploratory, not release.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,json,hashlib,time,traceback,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,a):
    temp=Path(str(p)+'.tmp');temp.write_text(json.dumps(a,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(p)
def load_frames(names,manifest):
    result=[]
    for name in names:
        file=HERE/name;assert sha(file)==manifest['files'][name]['sha256']
        with np.load(file,allow_pickle=False) as a:x=a['observed_frames'].copy()
        assert x.shape==(16,720,1280) and x.dtype==np.uint8
        result.append(x)
    out=np.concatenate(result);assert out.shape==(64,720,1280);return out

class ImageCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(3,12,3,padding=1),nn.SiLU(),
            nn.Conv2d(12,16,3,stride=2,padding=1),nn.SiLU(),
            nn.Conv2d(16,24,3,stride=2,padding=1),nn.SiLU(),
            nn.Conv2d(24,16,3,padding=1),nn.SiLU())
        self.attention=nn.Conv2d(16,1,1)
        self.detector=nn.Linear(17,1)
        self.regressor=nn.Sequential(nn.Linear(18,32),nn.Tanh(),nn.Linear(32,3))
    def forward(self,x):
        feat=self.encoder(x);heat=self.attention(feat).flatten(2)
        weight=torch.softmax(heat,dim=2)
        pooled=(feat.flatten(2)*weight).sum(2)
        h,w=feat.shape[-2:]
        yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=x.device),torch.linspace(-1,1,w,device=x.device),indexing='ij')
        coords=torch.stack((xx,yy)).reshape(1,2,-1)
        pos=(coords*weight).sum(2)
        strength=torch.logsumexp(heat,dim=2)-np.log(h*w)
        return self.detector(torch.cat((pooled,strength),1)).squeeze(1),self.regressor(torch.cat((pooled,pos),1))

def input_batch(raw,reference,variance,device):
    # Native 1280x720 input. No contact ROI, automatic exposure or resizing.
    image=torch.as_tensor(raw,device=device,dtype=torch.float32)
    diff=torch.clamp((image-reference)/torch.sqrt(variance),-10,10)
    return torch.stack((reference.expand_as(image)/255,diff,diff.abs()),1)
def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    c=(p+z*z/(2*n))/den;a=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(max(0,c-a)),float(min(1,c+a))]
def auc(pos,neg):
    d=pos[:,None]-neg[None];return float(np.mean((d>0)+.5*(d==0)))

def main(axis,gpu):
    assert axis in range(4);assert torch.cuda.is_available()
    torch.set_num_threads(4);device=torch.device(f'cuda:{gpu}');torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    m=read(HERE/'dataset_manifest.json');assert m['status']=='verified_existing_low2_image_pack'
    tilt=(0,12,34,60)[axis];static=next(s for s in m['statics'] if s['tilt_deg']==tilt)
    root=HERE/'results'/f'LED4_tilt{tilt:02d}';root.mkdir(parents=True,exist_ok=True)
    spec=dict(source_sha256=sha(__file__),dataset_manifest_sha256=sha(HERE/'dataset_manifest.json'),tilt_deg=tilt,
        epochs=64,batch_size=8,seeds=[1701,1702,1703],held_coordinate_indices=[0,1,2],
        train_press_noise_indices=list(range(16)),train_null_indices=list(range(32)),threshold_null_indices=list(range(32,64)),
        held_press_indices=list(range(64)),independent_null_indices=list(range(64)),
        split='Coordinate index held across ALL4 regions/BOTH loads/noise;8 training physical locations,4 held. No held statistics, early stopping or winner selection.',
        input='Native1280x720 reference/255, signed residual/static-noise-scale clipped+-10,absolute scaled residual. No region truth/crop or pressed geometry features.',
        variance='Static reference mean only, Poisson20ke/read3e/16frame reference plus linear8bit quantization variance; no touch label statistics.',
        model='Conv3to12_stride1,12to16_stride2,16to24_stride2,24to16_stride1;learned spatial attention,weighted features and pixel coordinates;detector and XYZ heads.',
        loss='Balanced BCE positive/null plus mean squared normalized XYZ on positive TRAIN images only.',
        optimization='Adam1e-3,64fixedepochs,gradientnormclip5,fullfloat32,noAMP/noTF32/deterministicalgorithms;final checkpoint only.',
        software=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,GPU=torch.cuda.get_device_name(device)),
        limits=m['limits']+' CNN training has only8 training physical sites per fold. Scores across64 noise views are conditional, not new locations. No dynamics/actions, candidate winner or real-hardware validation. Native input retains pixels but internal stride4 can lose weak subpixel changes; this architecture is a retained comparison, not a guarantee. Static reference is shared across held positions as calibration, never a pressed held frame.')
    if (root/'run_spec.json').exists():assert read(root/'run_spec.json')==spec
    else:save(root/'run_spec.json',spec)
    refpath=HERE/static['reference_file'];assert sha(refpath)==m['files'][static['reference_file']]['sha256']
    with np.load(refpath,allow_pickle=False) as a:reference_np=a['mean_reference_DN'].astype(np.float32)
    reference=torch.as_tensor(reference_np,device=device)
    variance=(torch.clamp(reference,min=0)*255/20000+(3*255/20000)**2+1/12)*(1+1/16)
    null=load_frames(static['threshold_null'],m);independent=load_frames(static['independent_null'],m)
    rows=[r for r in m['cases'] if r['tilt_deg']==tilt];assert len(rows)==24
    images={r['label']:load_frames(r['shards'],m) for r in rows}
    def infer(net,raw):
        cls=[];xyz=[]
        with torch.no_grad():
            for begin in range(0,len(raw),8):
                s,y=net(input_batch(raw[begin:begin+8],reference,variance,device))
                cls.append(s.cpu().numpy());xyz.append(y.cpu().numpy())
        return np.concatenate(cls),np.concatenate(xyz)
    reports=[]
    for held in (0,1,2):
        training=[r for r in rows if r['position_index']!=held];test=[r for r in rows if r['position_index']==held]
        trainids={(r['region'],r['position_index']) for r in training};testids={(r['region'],r['position_index']) for r in test}
        assert len(trainids)==8 and len(testids)==4 and not trainids.intersection(testids)
        positive=np.concatenate([images[r['label']][:16] for r in training])
        targets=np.concatenate([np.repeat(np.array(r['target_world_mm'])[None],16,0) for r in training]).astype(np.float32)
        ym=targets.mean(0);ys=np.maximum(targets.std(0),1e-6)
        traw=np.concatenate((positive,null[:32]));labels=np.r_[np.ones(len(positive)),np.zeros(32)].astype(np.float32)
        ty=np.vstack(((targets-ym)/ys,np.zeros((32,3)))).astype(np.float32)
        protocol=dict(train_position_ids=sorted(trainids),held_position_ids=sorted(testids),
            training_pixel_bytes_sha256=hashlib.sha256(traw.tobytes()).hexdigest(),training_targets_sha256=hashlib.sha256(ty.tobytes()).hexdigest(),
            target_mean_mm=ym.tolist(),target_scale_mm=ys.tolist(),run_spec_sha256=sha(root/'run_spec.json'))
        fold=root/f'heldCoordinate{held}';fold.mkdir(exist_ok=True)
        if (fold/'training_protocol.json').exists():assert read(fold/'training_protocol.json')==protocol
        else:save(fold/'training_protocol.json',protocol)
        for seed in spec['seeds']:
            dest=fold/f'seed{seed}';dest.mkdir(exist_ok=True);result=dest/'evaluation.json'
            if result.exists():
                old=read(result);assert old['protocol_sha256']==sha(fold/'training_protocol.json') and old['weights_sha256']==sha(dest/'weights.pt')
                assert old['predictions_sha256']==sha(dest/'predictions.npz');reports.append(old);continue
            np.random.seed(seed);random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            rng=np.random.default_rng(seed);net=ImageCNN().to(device);optimizer=torch.optim.Adam(net.parameters(),lr=1e-3)
            yy=torch.as_tensor(ty,device=device);lab=torch.as_tensor(labels,device=device)
            history=[];started=time.monotonic();net.train()
            # Global class-balanced fixed weights remain valid for all minibatches.
            balance=np.where(labels>0,len(labels)/(2*len(positive)),len(labels)/(2*32)).astype(np.float32)
            bw=torch.as_tensor(balance,device=device)
            for epoch in range(64):
                order=rng.permutation(len(traw));losses=[]
                for begin in range(0,len(order),8):
                    ids=order[begin:begin+8];x=input_batch(traw[ids],reference,variance,device)
                    score,pred=net(x);mask=lab[ids]>0
                    detection=(F.binary_cross_entropy_with_logits(score,lab[ids],reduction='none')*bw[ids]).mean()
                    # Sum / fixed expected positive fraction to avoid null-only batch failure.
                    regression=((pred-yy[ids]).square().mean(1)*mask).sum()/(len(ids)*len(positive)/len(traw))
                    loss=detection+regression
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss; preserve failure, do not evaluate')
                    optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);optimizer.step()
                    losses.append(float(loss.detach()))
                history.append(dict(epoch=epoch+1,mean_training_loss=float(np.mean(losses))))
                save(dest/'training_progress.json',dict(epochs_complete=epoch+1,epochs_registered=64,history=history,elapsed_seconds=time.monotonic()-started))
                if (epoch+1)%8==0:print(json.dumps(dict(tilt=tilt,held=held,seed=seed,epoch=epoch+1,loss=history[-1]['mean_training_loss'])),flush=True)
            weights=dest/'weights.pt';torch.save(dict(state_dict=net.state_dict(),target_mean_mm=ym.tolist(),target_scale_mm=ys.tolist(),protocol_sha256=sha(fold/'training_protocol.json')),weights)
            net.eval();cal_score,_=infer(net,null[32:]);threshold=float(np.quantile(cal_score,.95))
            null_score,_=infer(net,independent);details=[];predictions=[];scores=[];errors=[]
            for r in test:
                score,pred=infer(net,images[r['label']]);pred=pred*ys+ym
                error=np.linalg.norm(pred-np.array(r['target_world_mm']),axis=1);det=score>threshold
                details.append(dict(label=r['label'],region=r['region'],depth_mm=r['depth_mm'],
                    TPR=float(det.mean()),TPR_Wilson_conditional_noise=wilson(int(det.sum()),64),
                    FPR=float(np.mean(null_score>threshold)),AUC=auc(score,null_score),
                    unconditional_chord_error_median_mm=float(np.median(error)),unconditional_chord_error_p95_mm=float(np.quantile(error,.95)),
                    detected_and_localized_fraction_within_mm={str(mm):float(np.mean(det&(error<=mm))) for mm in (1.,2.,5.,10.)}))
                predictions.append(pred);scores.append(score);errors.append(error)
            prediction_file=dest/'predictions.npz'
            np.savez_compressed(prediction_file,held_labels=np.array([r['label'] for r in test]),held_prediction_world_mm=np.array(predictions),
                held_detection_scores=np.array(scores),held_chord_error_mm=np.array(errors),threshold=np.array(threshold),
                threshold_null_scores=cal_score,independent_null_scores=null_score,
                target_mean_mm=ym,target_scale_mm=ys)
            row=dict(tilt_deg=tilt,held_coordinate_index=held,seed=seed,algorithm='native_image_CNN',
                protocol_sha256=sha(fold/'training_protocol.json'),weights_file=str(weights.relative_to(HERE)),weights_sha256=sha(weights),
                predictions_file=str(prediction_file.relative_to(HERE)),predictions_sha256=sha(prediction_file),threshold=threshold,
                elapsed_seconds=time.monotonic()-started,history=history,held_field_details=details,limits=spec['limits'])
            save(result,row);reports.append(row);save(root/'progress.json',dict(completed_models=len(reports),registered_models=9,records=reports))
            print(json.dumps(dict(completed_native_CNN=[tilt,held,seed],models=len(reports),of=9)),flush=True)
            del net,optimizer;torch.cuda.empty_cache()
        del traw,positive
    assert len(reports)==9
    save(root/'manifest.json',dict(status='completed_exploratory_native_image_CNN_axis',**spec,records=reports))

if __name__=='__main__':
    axis,gpu=map(int,sys.argv[1:]);root=HERE/'results'/f'LED4_tilt{(0,12,34,60)[axis]:02d}'
    try:main(axis,gpu)
    except Exception:
        root.mkdir(parents=True,exist_ok=True);save(root/'failure.json',dict(source_sha256=sha(__file__),traceback=traceback.format_exc()));raise

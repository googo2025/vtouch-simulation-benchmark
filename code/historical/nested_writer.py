"""GPU nested physical-position validation on unchanged actual camera frames.

This supplements, never overwrites, the original64epoch CNN comparison.
Inner position validation selects finite LR/epoch budgets; outer tests are
evaluated after freeze. No claims of new physical locations or final ranking.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,json,random,hashlib,traceback,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import remote_full_image_cnn_v1 as original

HERE=Path(__file__).resolve().parent
OUT=HERE/'nested_results_v1'
SEEDS=(1701,1702,1703);LRS=(.001,.0003);BUDGETS=(64,128,256)
sha=original.sha;read=original.read;save=original.save

def infer(net,raw,reference,variance,device):
    net.eval();scores=[];coords=[]
    with torch.no_grad():
        for begin in range(0,len(raw),8):
            s,p=net(original.input_batch(raw[begin:begin+8],reference,variance,device))
            scores.append(s.cpu().numpy());coords.append(p.cpu().numpy())
    return np.concatenate(scores),np.concatenate(coords)

def fit(dest,seed,lr,budgets,positive,targets,nulltrain,reference,variance,device,validator):
    dest.mkdir(parents=True,exist_ok=True)
    raw=np.concatenate((positive,nulltrain));labels=np.r_[np.ones(len(positive)),np.zeros(len(nulltrain))].astype(np.float32)
    ym=targets.mean(0);ys=np.maximum(targets.std(0),1e-6)
    ty=np.vstack(((targets-ym)/ys,np.zeros((len(nulltrain),3)))).astype(np.float32)
    protocol=dict(seed=seed,learning_rate=lr,epoch_budgets=list(budgets),
        training_image_bytes_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
        training_target_bytes_sha256=hashlib.sha256(ty.tobytes()).hexdigest(),
        target_mean_mm=ym.tolist(),target_scale_mm=ys.tolist())
    protocolfile=dest/'training_protocol.json'
    if protocolfile.exists():assert read(protocolfile)==protocol
    else:save(protocolfile,protocol)
    if (dest/'complete.json').exists():
        report=read(dest/'complete.json');assert report['protocol_sha256']==sha(protocolfile)
        chosen=read(dest/'checkpoint_index.json')[-1]
        state=torch.load(dest/chosen['file'],map_location=device,weights_only=False)
        net=original.ImageCNN().to(device);net.load_state_dict(state['state_dict'])
        return net,ym,ys,report
    np.random.seed(seed);random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    rng=np.random.default_rng(seed);net=original.ImageCNN().to(device);optimizer=torch.optim.Adam(net.parameters(),lr=lr)
    lab=torch.as_tensor(labels,device=device);yy=torch.as_tensor(ty,device=device)
    weights=np.where(labels>0,len(labels)/(2*len(positive)),len(labels)/(2*len(nulltrain))).astype(np.float32)
    bw=torch.as_tensor(weights,device=device);history=[];checkpoints=[];start=0;best=None;bestvalue=float('inf')
    latest=dest/'latest_optimizer_checkpoint.pt'
    if latest.exists():
        state=torch.load(latest,map_location=device,weights_only=False)
        assert state['protocol_sha256']==sha(protocolfile)
        net.load_state_dict(state['state_dict']);optimizer.load_state_dict(state['optimizer'])
        rng.bit_generator.state=state['numpy_generator_state'];torch.set_rng_state(state['torch_rng'].cpu())
        torch.cuda.set_rng_state(state['cuda_rng'].cpu(),device);random.setstate(state['python_random_state'])
        start=state['epoch'];history=state['history'];checkpoints=state['checkpoints']
        best=state['best'];bestvalue=state['bestvalue']
    began=time.monotonic()
    for epoch in range(start,max(budgets)):
        net.train();order=rng.permutation(len(raw));losses=[];bces=[];regressions=[]
        for begin in range(0,len(order),8):
            ids=order[begin:begin+8];score,pred=net(original.input_batch(raw[ids],reference,variance,device))
            mask=lab[ids]>0
            bce=(F.binary_cross_entropy_with_logits(score,lab[ids],reduction='none')*bw[ids]).mean()
            reg=((pred-yy[ids]).square().mean(1)*mask).sum()/(len(ids)*len(positive)/len(raw))
            loss=bce+reg
            assert torch.isfinite(loss),'Nonfinite loss: preserve failure, do not evaluate.'
            optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);optimizer.step()
            losses.append(float(loss.detach()));bces.append(float(bce.detach()));regressions.append(float(reg.detach()))
        row=dict(epoch=epoch+1,training_total_loss=float(np.mean(losses)),
                 training_balanced_BCE=float(np.mean(bces)),training_normalized_XYZ_loss=float(np.mean(regressions)))
        if validator is not None and ((epoch+1)%8==0 or epoch+1 in budgets):
            row['validation']=validator(net,ym,ys)
            value=row['validation']['selection_loss']
            if value<bestvalue:
                bestvalue=value;best=dict(epoch=epoch+1,validation=row['validation'])
                torch.save(dict(state_dict=net.state_dict(),epoch=epoch+1,protocol_sha256=sha(protocolfile)),dest/'best_inner_validation_weights.pt')
        history.append(row)
        if epoch+1 in budgets:
            file=dest/f'epoch{epoch+1:04d}.pt'
            torch.save(dict(state_dict=net.state_dict(),target_mean_mm=ym.tolist(),target_scale_mm=ys.tolist(),
                            epoch=epoch+1,protocol_sha256=sha(protocolfile)),file)
            checkpoints.append(dict(epoch=epoch+1,file=file.name,sha256=sha(file),validation=row.get('validation')))
            save(dest/'checkpoint_index.json',checkpoints)
        if (epoch+1)%8==0 or epoch+1==max(budgets):
            state=dict(state_dict=net.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch+1,
                numpy_generator_state=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(device),python_random_state=random.getstate(),
                history=history,checkpoints=checkpoints,best=best,bestvalue=bestvalue,protocol_sha256=sha(protocolfile))
            tmp=dest/'latest_optimizer_checkpoint.tmp.pt';torch.save(state,tmp);tmp.replace(latest)
            save(dest/'training_progress.json',dict(epochs_complete=epoch+1,maximum_registered_epochs=max(budgets),
                 history=history,validation_available=validator is not None,elapsed_seconds=time.monotonic()-began))
            print(json.dumps(dict(training=str(dest.relative_to(OUT)),epoch=epoch+1,
                train_loss=row['training_total_loss'],validation=row.get('validation'))),flush=True)
    report=dict(protocol_sha256=sha(protocolfile),epochs_completed=max(budgets),history=history,checkpoints=checkpoints,
        best_inner_validation=best,stop_reason='predeclared_finite_budget_reached',optimizer_checkpoint_sha256=sha(latest),
        training_global_optimality_proven=False)
    save(dest/'complete.json',report)
    return net,ym,ys,report

def main(axis,gpu):
    assert axis in range(4) and torch.cuda.is_available()
    torch.set_num_threads(4);device=torch.device(f'cuda:{gpu}');torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    dataset=read(HERE/'dataset_manifest.json');tilt=(0,12,34,60)[axis]
    root=OUT/f'LED4_tilt{tilt:02d}';root.mkdir(parents=True,exist_ok=True)
    spec=dict(source_sha256=sha(__file__),original_architecture_source_sha256=sha(HERE/'remote_full_image_cnn_v1.py'),
        dataset_manifest_sha256=sha(HERE/'dataset_manifest.json'),tilt_deg=tilt,epoch_budgets=list(BUDGETS),learning_rates=list(LRS),seeds=list(SEEDS),
        registered_inner_fits=36,registered_outer_refits=9,
        split='Outer4held physicalsites coordinate index acrossALL4regions/BOTHloads/allnoise. Inner swap remainingcoordinate groups,4train/4validation sites.',
        inner_null_indices=dict(fit=list(range(16)),validation=list(range(16,32)),static_threshold=list(range(32,64))),
        outer_null_indices=dict(fit=list(range(32)),static_threshold=list(range(32,64)),test='separate independent_null'),
        positive_noise_indices=dict(fit=list(range(16)),inner_validation=list(range(16)),outer_test=list(range(64))),
        selection='Mean across2inner splits andALL3seeds of balanced validationBCE+training-normalized validationXYZ MSE. Predeclared budgets64/128/256,LR.001/.0003. No outer test or seed winner selection.',
        saving='8epoch optimizer/RNG/source/data checkpoint; separateBCE/XYZ curves, inner validation metrics/best weights, explicit finite-stop reason.',
        limitations=dataset['limits']+' Exploratory reused developed outer test,12physicalsites. No new FE/noise, dynamic actions, fullcandidate comparison, region/azimuth confirmation or architecture global optimum.')
    sp=root/'run_spec.json'
    if sp.exists():assert read(sp)==spec
    else:save(sp,spec)
    if (root/'manifest.json').exists():print('Completed nestedCNN preserved');return
    static=next(s for s in dataset['statics'] if s['tilt_deg']==tilt)
    ref=HERE/static['reference_file'];assert sha(ref)==dataset['files'][static['reference_file']]['sha256']
    with np.load(ref,allow_pickle=False) as a:reference=torch.as_tensor(a['mean_reference_DN'].astype(np.float32),device=device)
    variance=(torch.clamp(reference,min=0)*255/20000+(3*255/20000)**2+1/12)*(1+1/16)
    null=original.load_frames(static['threshold_null'],dataset);independent=original.load_frames(static['independent_null'],dataset)
    rows=[r for r in dataset['cases'] if r['tilt_deg']==tilt];assert len(rows)==24
    images={r['label']:original.load_frames(r['shards'],dataset) for r in rows}
    def stack(rs):
        return np.concatenate([images[r['label']][:16] for r in rs]),np.concatenate([
            np.repeat(np.array(r['target_world_mm'],np.float32)[None],16,0) for r in rs])
    def validator(vraw,vtarget):
        def run(net,ym,ys):
            positive,pred=infer(net,vraw,reference,variance,device);negative,_=infer(net,null[16:32],reference,variance,device)
            calibration,_=infer(net,null[32:],reference,variance,device);threshold=float(np.quantile(calibration,.95))
            ce=.5*float(np.mean(np.logaddexp(0,-positive))+.0)+.5*float(np.mean(np.logaddexp(0,negative)))
            normalized=float(np.mean((pred-(vtarget-ym)/ys)**2));world=pred*ys+ym;err=np.linalg.norm(world-vtarget,axis=1)
            return dict(selection_loss=ce+normalized,balanced_BCE=ce,normalized_XYZ_MSE=normalized,
                mean_chord_mm=float(err.mean()),median_chord_mm=float(np.median(err)),p95_chord_mm=float(np.quantile(err,.95)),
                TPR=float(np.mean(positive>threshold)),FPR=float(np.mean(negative>threshold)))
        return run
    records=[]
    for held in (0,1,2):
        dest=root/f'heldCoordinate{held}';dest.mkdir(exist_ok=True);report=dest/'evaluation.json'
        if report.exists():records.append(read(report));continue
        remaining=[j for j in (0,1,2) if j!=held];inner=[]
        for valgroup in remaining:
            tr=[r for r in rows if r['position_index'] not in (held,valgroup)];va=[r for r in rows if r['position_index']==valgroup]
            trids={(r['region'],r['position_index']) for r in tr};valids={(r['region'],r['position_index']) for r in va}
            testids={(r['region'],r['position_index']) for r in rows if r['position_index']==held}
            assert len(trids)==len(valids)==len(testids)==4 and not(trids&valids or trids&testids or valids&testids)
            positive,targets=stack(tr);vraw,vtarget=stack(va)
            save(dest/f'inner_split{valgroup}.json',dict(training_position_ids=sorted(trids),validation_position_ids=sorted(valids),outer_test_ids=sorted(testids)))
            for seed in SEEDS:
              for lr in LRS:
                dd=dest/f'inner_validationCoordinate{valgroup}'/f'seed{seed}_lr{lr}'
                net,_,_,trained=fit(dd,seed,lr,BUDGETS,positive,targets,null[:16],reference,variance,device,validator(vraw,vtarget))
                inner.append(dict(valgroup=valgroup,seed=seed,lr=lr,checkpoints=trained['checkpoints']))
                del net;torch.cuda.empty_cache()
        options=[]
        for lr in LRS:
          for epoch in BUDGETS:
            scores=[c['validation']['selection_loss'] for r in inner if r['lr']==lr for c in r['checkpoints'] if c['epoch']==epoch]
            assert len(scores)==6
            options.append(dict(learning_rate=lr,epochs=epoch,mean_inner_validation_loss=float(np.mean(scores)),all_inner_seed_split_values=scores))
        chosen=min(options,key=lambda r:(r['mean_inner_validation_loss'],r['epochs'],r['learning_rate']))
        freeze=dest/'selection_frozen_before_outer_evaluation.json'
        frozen=dict(chosen=chosen,all_candidates=options,outer_test_selection=False,all_seeds_retained=True,run_spec_sha256=sha(sp))
        if freeze.exists():assert read(freeze)==frozen
        else:save(freeze,frozen)
        tr=[r for r in rows if r['position_index']!=held];test=[r for r in rows if r['position_index']==held]
        positive,targets=stack(tr);comparisons=[]
        for seed in SEEDS:
            dd=dest/f'outer_refit_seed{seed}'
            net,ym,ys,trained=fit(dd,seed,chosen['learning_rate'],(chosen['epochs'],),positive,targets,null[:32],reference,variance,device,None)
            calibration,_=infer(net,null[32:],reference,variance,device);threshold=float(np.quantile(calibration,.95))
            negative,_=infer(net,independent,reference,variance,device);details=[];predictions=[];scores=[];errors=[]
            for r in test:
                score,pred=infer(net,images[r['label']],reference,variance,device);pred=pred*ys+ym
                error=np.linalg.norm(pred-np.array(r['target_world_mm']),axis=1);det=score>threshold
                details.append(dict(label=r['label'],region=r['region'],depth_mm=r['depth_mm'],TPR=float(det.mean()),
                    FPR=float(np.mean(negative>threshold)),AUC=original.auc(score,negative),
                    unconditional_chord_error_median_mm=float(np.median(error)),unconditional_chord_error_p95_mm=float(np.quantile(error,.95)),
                    detected_and_localized_fraction_within_mm={str(mm):float(np.mean(det&(error<=mm))) for mm in (1.,2.,5.,10.)}))
                predictions.append(pred);scores.append(score);errors.append(error)
            file=dd/'locked_outer_predictions.npz'
            np.savez_compressed(file,held_labels=np.array([r['label'] for r in test]),held_prediction_world_mm=np.array(predictions),
                held_detection_scores=np.array(scores),held_chord_error_mm=np.array(errors),threshold=np.array(threshold),
                threshold_null_scores=calibration,independent_null_scores=negative,target_mean_mm=ym,target_scale_mm=ys)
            comparisons.append(dict(seed=seed,predictions_file=str(file.relative_to(HERE)),predictions_sha256=sha(file),
                selection_freeze_sha256=sha(freeze),final_weights_file=str((dd/trained['checkpoints'][-1]['file']).relative_to(HERE)),
                final_weights_sha256=trained['checkpoints'][-1]['sha256'],held_field_details=details))
            del net;torch.cuda.empty_cache()
        row=dict(tilt_deg=tilt,held_coordinate_index=held,chosen=chosen,comparisons=comparisons,
                 outer_test_selection=False,limits=spec['limitations'])
        save(report,row);records.append(row);save(root/'progress.json',dict(completed_outer_folds=len(records),registered_outer_folds=3,records=records))
    save(root/'manifest.json',dict(status='completed_exploratory_nested_native_CNN',**spec,records=records,
                                 training_global_optimality_proven=False,candidate_hardware_selection_frozen=False))

if __name__=='__main__':
    axis,gpu=map(int,sys.argv[1:]);root=OUT/f'LED4_tilt{(0,12,34,60)[axis]:02d}'
    try:main(axis,gpu)
    except Exception:
        root.mkdir(parents=True,exist_ok=True);save(root/'failure.json',dict(source_sha256=sha(__file__),traceback=traceback.format_exc()));raise

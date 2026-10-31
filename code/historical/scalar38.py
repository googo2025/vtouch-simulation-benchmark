"""Exploratory true-position grouped detection and continuous 3D regression.

All load/noise views of each held physical position are excluded from fitting.
Coordinates are labels only. Low/adaptive runs use the same position protocol;
neither mode selects winning lights, models, seeds or candidates on tests.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json
from pathlib import Path
import numpy as np
from audit_v2_true_airgap_optical_pair_v1 import HERE,ROOT,V2,save,sha
from v2_common_camera_localization_baselines_v1 import nominal_surface
from v2_offcentre_camera_localization_baselines_v1 import peaks_with_features
from v2_whole_image_flow_features_v1 import setup
from v2_grouped_whole_image_detector_models_v1 import fit_pca,encode,train,serial,scores,evaluate,LAB
from candidate_numpy_models import predict

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def arrays(p):
    with np.load(p) as a:return {k:a[k].copy() for k in a.files}

def extend(a,cal,p):
    """Full-frame peaks plus fixed REST geometry; no contact truth input."""
    xy=a['pixel_peaks'];ray=[];valid=[];anchors=[]
    for method in range(2):
        points=[nominal_surface(v,p) for v in xy[:,method]]
        ray.append(np.array([v if v is not None else np.zeros(3) for v in points]))
        valid.append(np.array([v is not None for v in points],float)[:,None])
        nearest=np.argmin(np.sum((xy[:,method,None]-cal['pixel'][None])**2,axis=2),axis=1)
        anchors.append(cal['anchor'][nearest])
    x=np.c_[a['features'],xy.reshape(len(xy),4)/np.array([1280,720,1280,720]),
        a['box_sizes']/16,ray[0]/100,ray[1]/100,valid[0],valid[1],anchors[0]/100,anchors[1]/100]
    assert x.shape==(len(xy),38) and np.all(np.isfinite(x));return x

def rest_features(mode,axis,folder,cal,p):
    """Extract existing low REST frames once; adaptive already saved features."""
    if mode=='adaptive':
        return {name:extend(arrays(folder/(name+'_peaks.npz')),cal,p) for name in ('threshold_null','independent_null')},[]
    base=ROOT/'v2_common_camera_localization_baselines_v1'/folder.name
    reference=arrays(base/'reference.npz')['mean_reference_DN'];prepared=setup(reference,20000)
    out=ROOT/'v2_offcentre_grouped_detection_localization_v1_low2/rest_feature_cache';out.mkdir(parents=True,exist_ok=True)
    results={};sources=[]
    for name in ('threshold_null','independent_null'):
        meta=read(base/(name+'.json'));file=out/f'{folder.name}_{name}.npz';report=file.with_suffix('.json')
        source=dict(noise_meta_sha256=sha(base/(name+'.json')),reference_sha256=sha(base/'reference.npz'),
            original_peaks_sha256=sha(base/(name+'_peaks.npz')),extractor_sha256=sha(HERE/'v2_offcentre_camera_localization_baselines_v1.py'))
        if report.exists():
            old=read(report);assert old['sources']==source and sha(file)==old['sha256'];a=arrays(file)
        else:
            feature=[];peaks=[];sizes=[];score=[]
            for shard in meta['raw_shards']:
                assert sha(shard['file'])==shard['sha256']
                for image in arrays(shard['file'])['observed_frames']:
                    result,f=peaks_with_features(image,reference,prepared);feature.append(f)
                    peaks.append([r[1] for r in result]);sizes.append([r[2] for r in result]);score.append([r[0] for r in result])
            a=dict(features=np.array(feature),pixel_peaks=np.array(peaks),box_sizes=np.array(sizes))
            original=arrays(base/(name+'_peaks.npz'))
            np.testing.assert_array_equal(a['pixel_peaks'],original['pixel_peaks']);np.testing.assert_array_equal(a['box_sizes'],original['box_sizes'])
            np.testing.assert_allclose(score,original['scores'],rtol=1e-12,atol=1e-12)
            np.savez_compressed(file,**a);save(report,dict(sources=source,sha256=sha(file),actual_saved_frames_only=True))
        results[name]=extend(a,cal,p);sources.append(dict(file=str(report),sha256=sha(report)))
    return results,sources

def main(mode):
    assert mode in ('low2','adaptive');parent=ROOT/('v2_offcentre_camera_localization_baselines_v1_'+mode)
    out=ROOT/('v2_offcentre_grouped_detection_localization_v1_'+mode)
    if (out/'manifest.json').exists():print('Completed grouped offcentre models preserved',mode);return
    source=read(parent/'manifest.json');assert source['status']=='completed_actual_offcentre_localization_baselines'
    calfile=ROOT/'v2_common_camera_localization_baselines_v1/ideal_rest_marker_calibration.npz';cal=arrays(calfile);p=read(V2)['params']
    spec=dict(source_sha256=sha(__file__),mode=mode,parent_manifest_sha256=sha(parent/'manifest.json'),
        model_source_sha256=sha(LAB/'candidate_numpy_models.py'),ideal_rest_calibration_sha256=sha(calfile),
        registered_folds=12,tilts_deg=[0.,12.,34.,60.],held_coordinate_indices=[0,1,2],
        split='Hold coordinate index across ALL4 regions and BOTH loads; train8 physical positions/test4. All held noise/load views excluded from PCA, fitting, target normalization and thresholds.',
        training_press_noise_indices=list(range(16)),training_null_noise_indices=list(range(32)),threshold_null_noise_indices=list(range(32,64)),
        held_press_noise_indices=list(range(64)),independent_null_noise_indices=list(range(64)),
        features='18 fixed energy/gradient descriptors; two unknown full-image peak pixels, box sizes, nominal-ray coordinates/valid flags and REST-only ideal marker anchors.38total. No pressed truth/region label is a feature.',
        training_only_PCA_components=12,training_only_input_scale='Per-feature standard deviation floor1e-6, then training PCA centering/whitening; detector/regression fitted separately. No held data statistics.',
        models=['ridge','MLP24_tanh_300Adam'],MLP_seeds=[1701,1702,1703],
        threshold_quantile=.95,regression_target='Command undeformed external contact XYZ, training-only target mean/std; unconstrained XYZ prediction. Euclidean chord error, not geodesic resolution.',
        limits='Exploratory post-development12-position/coarse-model comparison, one source/four axes/two depths, assumed sensor/material. Noise repetitions are not independent locations; Wilson detection intervals conditional on noise only. All variants retained; no selection on held errors. Region and azimuth generalization, shape comparison, action recognition and confirmatory full-surface validation remain absent. Geometry-assisted REST inputs are idealized, not measured segmentation. Loaded-cut cases were explicitly excluded by upstream evaluation.')
    out.mkdir(exist_ok=True)
    if (out/'run_spec.json').exists():assert read(out/'run_spec.json')==spec
    else:save(out/'run_spec.json',spec)
    records=[]
    for axis,tilt in enumerate(spec['tilts_deg']):
        folder=parent/f'LED4_tilt{round(tilt):02d}';nulls,restsources=rest_features(mode,axis,folder,cal,p)
        rows=[r for r in source['records'] if r['tilt_deg']==tilt]
        assert len(rows)==24 and {r['position_index'] for r in rows}=={0,1,2}
        values={};targets={};provenance=[]
        for row in rows:
            file=folder/(row['label']+'_peaks.npz');a=arrays(file);assert sha(file)==row['raw_noise']['peaks_sha256']
            values[row['label']]=extend(a,cal,p)
            loc=arrays(row['comparisons'][0]['location_file']);assert sha(row['comparisons'][0]['location_file'])==row['comparisons'][0]['location_sha256']
            targets[row['label']]=loc['evaluation_only_target_world_mm']
            provenance.append(dict(label=row['label'],features_file=str(file),features_sha256=sha(file),target_file=row['comparisons'][0]['location_file'],target_sha256=row['comparisons'][0]['location_sha256']))
        for held in (0,1,2):
            dest=out/f'LED4_tilt{round(tilt):02d}_heldCoordinate{held}';dest.mkdir(exist_ok=True);report=dest/'evaluation.json'
            if report.exists():records.append(read(report));continue
            training=[r for r in rows if r['position_index']!=held];test=[r for r in rows if r['position_index']==held]
            trainids={(r['region'],r['position_index']) for r in training};testids={(r['region'],r['position_index']) for r in test}
            assert len(trainids)==8 and len(testids)==4 and not trainids.intersection(testids)
            # Noise views enrich conditional training, not the physical-position count.
            X=np.vstack([values[r['label']][:16] for r in training]);Y=np.vstack([np.repeat(targets[r['label']][None],16,axis=0) for r in training])
            DX=np.vstack([nulls['threshold_null'][:32],X]);DY=np.r_[np.zeros(32,int),np.ones(len(X),int)]
            dscale=np.maximum(DX.std(0),1e-6);rscale=np.maximum(X.std(0),1e-6)
            dpca=fit_pca(DX/dscale,12);rpca=fit_pca(X/rscale,12);ymean=Y.mean(0);yscale=np.maximum(Y.std(0),1e-6)
            freeze=dest/'training_protocol.json';save(freeze,dict(train_position_ids=sorted(trainids),held_position_ids=sorted(testids),
                training_labels_only=True,training_features_sha256=__import__('hashlib').sha256(np.ascontiguousarray(X).tobytes()).hexdigest(),
                source_provenance=provenance,rest_provenance=restsources,run_spec_sha256=sha(out/'run_spec.json')))
            comparisons=[]
            for kind,seeds in (('ridge',(1701,)),('mlp',(1701,1702,1703))):
                for seed in seeds:
                    detector=train(encode(DX/dscale,dpca),DY,kind,seed=seed,classification=True)
                    regressor=train(encode(X/rscale,rpca),(Y-ymean)/yscale,kind,seed=seed,classification=False)
                    threshold=float(np.quantile(scores(nulls['threshold_null'][32:]/dscale,dpca,detector),.95))
                    nullscore=scores(nulls['independent_null']/dscale,dpca,detector);details=[];saved=[]
                    for row in test:
                        x=values[row['label']];positive=scores(x/dscale,dpca,detector);pred=predict(encode(x/rscale,rpca),regressor)*yscale+ymean
                        assert np.all(np.isfinite(pred));error=np.linalg.norm(pred-targets[row['label']],axis=1);detected=positive>threshold
                        details.append(dict(label=row['label'],region=row['region'],position_index=held,depth_mm=row['depth_mm'],
                            detection=evaluate(positive,nullscore,threshold),unconditional_chord_error_median_mm=float(np.median(error)),
                            unconditional_chord_error_p95_mm=float(np.quantile(error,.95)),detected_count=int(detected.sum()),
                            conditional_detected_chord_error_median_mm=float(np.median(error[detected])) if detected.any() else None,
                            detected_and_localized_fraction_within_mm={str(r):float(np.mean(detected&(error<=r))) for r in (1.,2.,5.,10.)}))
                        saved.append((pred,error,positive,detected))
                    file=dest/f'{kind}_{seed}_weights_and_predictions.npz'
                    np.savez_compressed(file,**{'detector_PCA_'+k:v for k,v in dpca.items()},**{'regressor_PCA_'+k:v for k,v in rpca.items()},
                        **{'detector_'+k:v for k,v in serial(detector).items()},**{'regressor_'+k:v for k,v in serial(regressor).items()},
                        training_features=X,training_target_world_mm=Y,training_detector_features=DX,training_detector_labels=DY,
                        detector_input_scale=dscale,regressor_input_scale=rscale,
                        regression_target_mean_mm=ymean,regression_target_scale_mm=yscale,detector_threshold=np.asarray(threshold),
                        held_labels=np.array([r['label'] for r in test]),held_prediction_world_mm=np.array([s[0] for s in saved]),
                        held_chord_error_mm=np.array([s[1] for s in saved]),held_detection_scores=np.array([s[2] for s in saved]),held_detected=np.array([s[3] for s in saved]),
                        independent_null_scores=nullscore)
                    comparisons.append(dict(algorithm=kind,seed=seed,threshold=threshold,weights_and_predictions_file=str(file),sha256=sha(file),held_field_details=details))
            result=dict(mode=mode,tilt_deg=tilt,held_coordinate_index=held,independent_training_positions=8,independent_held_positions=4,
                held_field_states=8,protocol_sha256=sha(freeze),comparisons=comparisons,limits=spec['limits'])
            save(report,result);records.append(result);save(out/'progress.json',dict(completed_folds=len(records),registered_folds=12,records=records))
            print(json.dumps(dict(offcentre_grouped_models_completed=[mode,tilt,held],methods=4)),flush=True)
    save(out/'manifest.json',dict(status='completed_exploratory_offcentre_grouped_models',**spec,records=records))
    save(HERE/f'v2_offcentre_grouped_detection_localization_v1_{mode}_index.json',dict(manifest=str(out/'manifest.json'),sha256=sha(out/'manifest.json')))

if __name__=='__main__':
    assert len(sys.argv)==2;main(sys.argv[1])

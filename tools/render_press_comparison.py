import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from concurrent.futures import ProcessPoolExecutor
import multiprocessing,json,hashlib,time,resource,math
import argparse
ap=argparse.ArgumentParser(description='Render existing paired VTouch display exports; no simulation or learning.')
ap.add_argument('--source-root',type=Path,required=True,help='Folder containing cases/, meshes/ and planes/ selected display exports')
ap.add_argument('--output',type=Path,required=True)
args=ap.parse_args();C=args.source_root;OUT=args.output;OUT.mkdir(parents=True,exist_ok=True)
START=time.monotonic();CPU0=resource.getrusage(resource.RUSAGE_SELF);CH0=resource.getrusage(resource.RUSAGE_CHILDREN)
source_records=[];meta={};meshes={};planes={}
def read_verified(p,expected):
 b=p.read_bytes();h=hashlib.sha256(b).hexdigest();assert h==expected,p.name
 source_records.append({'file':p.name,'sha256':h,'bytes':len(b)})
 return b
for idx in [82,83,101]:
 d=json.loads((C/f'cases/{idx:03}.json').read_text());meta[idx]=d
 m=d['mesh_display'];meshes[idx]=np.frombuffer(read_verified(C/m['path'],m['SHA']),dtype='<f4').reshape(-1,7)[:,:3]
 for p in d['native_planes']:
  if p['actual_LED_index']==4:
   planes[idx,p['emission_tilt_deg']]=np.frombuffer(read_verified(C/p['path'],p['SHA']),dtype='<f4').reshape(720,1280)
for idx in [83,101]:
 assert meta[idx]['reference_case_index']==82
 assert meta[idx]['group_id']==meta[82]['group_id']
 assert meta[idx]['parent_SHA']==meta[82]['parent_SHA']
tilts=[0,12,34,60]
delta={(idx,t):(planes[idx,t]-planes[82,t]) for idx in [83,101] for t in tilts}
rawmax=max(float(np.max(np.abs(v))) for v in delta.values());assert rawmax>0
color_limit=rawmax/4
gray_white=max(float(np.max(v)) for v in planes.values())
ROIS={};roi_records={}
for idx in [83,101]:
 union=np.maximum.reduce([np.abs(delta[idx,t]) for t in tilts])
 y,x=np.unravel_index(np.argmax(union),union.shape)
 x=int(np.clip(x,80,1200));y=int(np.clip(y,80,640))
 ROIS[idx]=(x-80,y-80,x+80,y+80)
 roi_records[idx]={'crop_native_pixels':list(ROIS[idx]),'selection':'fixed 160x160 display crop centered on the across-light peak absolute image change; not a recognition target or new label','maximum_abs_saved_radiance_change':float(union.max())}
ENG='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';CJK='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
def font(n,lang):return ImageFont.truetype(CJK if lang=='zh' else ENG,n)
def colorize(values):
 t=np.clip(values/color_limit,-1,1)[...,None];v=np.abs(t)
 center=np.array([247,249,251.],np.float32);red=np.array([210,40,52.],np.float32);blue=np.array([34,96,210.],np.float32)
 ends=np.where(t>=0,red,blue);rgb=center*(1-v)+ends*v
 return Image.fromarray(np.rint(rgb).astype(np.uint8))
def grayscale(values):
 a=np.rint(np.power(np.clip(values/gray_white,0,1),0.45)*255).astype(np.uint8)
 return Image.fromarray(a).convert('RGB')
GRAY={k:grayscale(v) for k,v in planes.items()}
COL={k:colorize(v) for k,v in delta.items()}
def geometry_panel(idx,lang):
 im=Image.new('RGB',(225,592),'#112639');d=ImageDraw.Draw(im)
 nominal=np.array(meta[idx]['nominal_command_source'][0]['original_fields']['command_outer_target_world_mm'])
 phi=math.atan2(nominal[1],nominal[0]);cp,sp=math.cos(phi+.45),math.sin(phi+.45)
 def project(points):
  x=points[:,0]*cp+points[:,1]*sp;depth=-points[:,0]*sp+points[:,1]*cp
  y=points[:,2]*math.cos(.12)-depth*math.sin(.12)
  return np.stack([x,y],axis=1),depth
 rest=meshes[82];loaded=meshes[idx]
 nearest=np.argmin(np.sum((rest-nominal)**2,axis=1));focus=project(rest[nearest:nearest+1])[0][0]
 def scatter(points,col,zoom=False):
  pts=points[::5];xy,dep=project(pts)
  if zoom:
   a=xy-focus;mask=(np.abs(a[:,0])<=3)&(np.abs(a[:,1])<=3)&(dep>-6)
   px=112+a[mask,0]*29;py=467-a[mask,1]*29
  else:
   mask=dep>-2;px=112+xy[mask,0]*4.2;py=187-(xy[mask,1]-37)*4.2
  for x,y in zip(px,py):
   if 4<=x<221 and ((zoom and 375<=y<559) or (not zoom and 25<=y<354)):
    d.rectangle((float(x),float(y),float(x)+1,float(y)+1),fill=col)
 for zoom in [False,True]:
  scatter(rest,'#b7c4d0',zoom);scatter(loaded,'#38dbc4',zoom)
 d.text((10,5),'原三维曲面' if lang=='zh' else 'Original curved 3D',font=font(16,lang),fill='#f2f7fa')
 d.text((10,351),'局部三维特写 · 6 mm' if lang=='zh' else '3D close-up · 6 mm',font=font(15,lang),fill='#e2f2f7')
 d.text((10,563),'灰：未按压  青：按压' if lang=='zh' else 'Grey: rest  Cyan: press',font=font(13,lang),fill='#e2f2f7')
 return im
GEOS={(i,l):geometry_panel(i,l) for i in [83,101] for l in ['en','zh']}
labels={'en':{'title':'VTouch · Unpressed / pressed / optical change','native':'Original camera view · no unwrapping','before':'Unpressed reference','after':'Pressed observation','diff':'Signed optical difference','zoom':'Aligned 160 × 160 pixel close-up','light':'LED4 direction','region':'Region','depth':'command indentation 0.3 mm','legend':'Loaded - reference','scale':'Fixed scales across lights · image gamma 0.45','more':'Existing paired simulation states · grey = reference · cyan = loaded 3D'},
'zh':{'title':'VTouch · 未按压 / 按压后 / 光学变化','native':'原始相机视角 · 不展开','before':'未按压参考','after':'按压后观察','diff':'彩色光学差分','zoom':'同位置 160 × 160 像素放大特写','light':'LED4 光照方向','region':'区域','depth':'指令压入 0.3 毫米','legend':'按压后 − 未按压','scale':'照明间显示尺度固定 · 灰度图 gamma 0.45','more':'已有同源仿真状态 · 三维灰色为未按压，青色为按压后'}}
def render(item):
 idx,t,lang=item;L=labels[lang];im=Image.new('RGB',(1280,790),'#f2f6f9');d=ImageDraw.Draw(im)
 d.text((22,12),L['title'],font=font(27,lang),fill='#15364d')
 name='S04 · side' if idx==83 else 'H05 · apex'
 if lang=='zh':name='S04 · 侧面' if idx==83 else 'H05 · 顶端'
 d.text((24,55),L['region']+' '+name+'  |  '+L['depth']+'  |  '+L['light']+' '+str(t)+'°',font=font(18,lang),fill='#31596e')
 im.paste(GEOS[idx,lang],(18,105))
 xcols=[267,603,939];titles=[L['before'],L['after'],L['diff']];images=[GRAY[82,t],GRAY[idx,t],COL[idx,t]]
 for x,title,pic in zip(xcols,titles,images):
  d.text((x,106),title,font=font(19,lang),fill='#18394e')
  full=pic.resize((320,180),Image.Resampling.LANCZOS);im.paste(full,(x,139))
  a,b,c,e=ROIS[idx];d.rectangle((x+a/4,139+b/4,x+c/4,139+e/4),outline='#df8c33',width=2)
  crop=pic.crop(ROIS[idx]).resize((300,300),Image.Resampling.NEAREST);im.paste(crop,(x+10,365))
  d.rectangle((x+9,364,x+311,666),outline='#d79649',width=2)
 d.text((267,326),L['zoom']+' · '+L['native'],font=font(17,lang),fill='#305269')
 ramp=np.linspace(-color_limit,color_limit,320,dtype=np.float32)[None,:].repeat(18,axis=0)
 im.paste(colorize(ramp),(939,692))
 d.text((267,696),L['scale'],font=font(16,lang),fill='#31566c')
 d.text((939,718),'-'+format(color_limit,'.2e')+'     0     +'+format(color_limit,'.2e'),font=font(13,'en'),fill='#284b60')
 d.text((267,735),L['legend']+' · '+('保存辐亮度单位，超出色标时饱和' if lang=='zh' else 'saved radiance units; colors saturate at endpoints'),font=font(14,lang),fill='#486576')
 d.text((22,765),L['more'],font=font(13,lang),fill='#486576')
 return idx,t,lang,im
jobs=[(i,t,l) for l in ['en','zh'] for i in [83,101] for t in tilts]
with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('fork')) as ex:frames=list(ex.map(render,jobs))
# One shared palette retains the same color mapping throughout both animations.
contact=Image.new('RGB',(640*4,395*2),'white')
for n,(_,_,_,im) in enumerate(x for x in frames if x[2]=='en'):contact.paste(im.resize((640,395)),((n%4)*640,(n//4)*395))
palette=contact.quantize(colors=256)
exports=[]
for lang in ['en','zh']:
 selected=[x[3] for x in frames if x[2]==lang]
 for n,im in enumerate(selected):im.save(OUT/f'{lang}-condition-{n:02}.png')
 q=[x.quantize(palette=palette,dither=Image.Dither.NONE) for x in selected]
 p=OUT/f'press-reference-lighting-{lang}.gif';q[0].save(p,save_all=True,append_images=q[1:],loop=0,duration=1400,disposal=2,optimize=False)
 exports.append({'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size,'frames':len(q),'duration_seconds':len(q)*1.4,'dimensions':[1280,790]})
CPU1=resource.getrusage(resource.RUSAGE_SELF);CH1=resource.getrusage(resource.RUSAGE_CHILDREN)
report={'purpose':'display existing paired states only','cases':[83,101],'reference_case':82,'same_parent_group_verified':True,'LED':4,'illumination_tilts_deg':tilts,'source_shape':[720,1280],'native_projection_preserved':True,'unwrapped_height_maps_used':False,'geometry_amplification':1.0,'geometry_display_stride':5,'geometry_closeup_width_mm':6,'signed_difference_definition':'loaded saved radiance minus paired unloaded saved radiance','fixed_gray_white':gray_white,'fixed_observation_display_gamma':0.45,'optical_difference_uses_unmodified_source_values':True,'fixed_symmetric_difference_color_limit':color_limit,'maximum_abs_difference_all_frames':rawmax,'color_clipping':'saturation outside fixed endpoints, identical across every frame','crops':roi_records,'source_files':source_records,'source_scene_sha256':{i:meta[i]['scene_SHA'] for i in [82,83,101]},'exports':exports,'new_simulation_training_inference_or_recognition_features':False,'workers':3,'wall_seconds':time.monotonic()-START,'CPU_seconds_total':CPU1.ru_utime+CPU1.ru_stime-CPU0.ru_utime-CPU0.ru_stime+CH1.ru_utime+CH1.ru_stime-CH0.ru_utime-CH0.ru_stime,'parent_peak_RSS_KiB':CPU1.ru_maxrss,'children_peak_RSS_KiB':CH1.ru_maxrss,'source_bytes_read':sum(x['bytes'] for x in source_records)}
(OUT/'provenance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:report[k] for k in ['exports','fixed_symmetric_difference_color_limit','maximum_abs_difference_all_frames','crops','wall_seconds','CPU_seconds_total','parent_peak_RSS_KiB','children_peak_RSS_KiB','source_bytes_read']}))

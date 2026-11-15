import * as T from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';

const zh=document.documentElement.lang.startsWith('zh');
const lang=zh?'zh':'en';
const strings={
  en:{paper:'Paper C',eyebrow:'Optical tactile sensing',title:'Explore the curved tactile shell',subtitle:'Inspect the exact shell geometry, replay saved contact estimates, and compare the 17 designs from the static simulation study.',geometry:'3D shell',dimensions:'Ø32 × 76 mm',variant:'Shell variant',cut:'Cutaway',reset:'Reset view',drag:'Drag to rotate · scroll or pinch to zoom',locate:'Unseen static contacts',locdesc:'One archived frame from each of 12 off-axis boundary conditions. Markers are replayed on the 1.8 mm new regional shell. Coordinates identify the archived model target and predictions.',case:'Condition',truth:'Target',baseline:'Light CNN',routing:'With boundary routing',provenance:'Data provenance',comparison:'17 morphologies · repeat observations at four calibrated sites',desc:'Training and evaluation share the four contact locations and three depths. Evaluation uses different simulated noise draws. 0.000 mm can mean the model selected the matching calibration anchor.',morph:'Morphology',hits:'Detected',falses:'Empty false alarms',mean:'Mean XYZ (mm)',max:'Max XYZ (mm)',ablation:'Shared encoder ablation',ablationDesc:'Relative change in mean development-set error across three seeds. A decrease in detection or position error can coincide with a rise in displacement-field error.',footer:'VTouch Paper C · Simulation data · Interactive replay of archived predictions',wall:'Base wall',inner:'Nominal base ID',loading:'Loading mesh…',failed:'The 3D mesh could not be loaded.',noWebgl:'WebGL is unavailable. The study data below remain accessible.',dataError:'The study data could not be loaded.',meanError:'XYZ error'},
  zh:{paper:'论文 C',eyebrow:'曲面视触觉传感',title:'曲面触觉壳体交互',subtitle:'查看真实 CAD 网格、回放已保存的触点预测，并对照静态仿真研究中的 17 种壳体。',geometry:'壳体三维模型',dimensions:'Ø32 × 76 mm',variant:'壳体方案',cut:'剖切',reset:'重置视角',drag:'拖动旋转 · 滚轮或双指缩放',locate:'未见位置的静态接触',locdesc:'离轴交界的 12 个工况各显示一张存档帧。触点标记显示在 1.8 mm 新分区组合壳上，对应存档的目标与预测坐标。',case:'工况',truth:'目标坐标',baseline:'轻量 CNN',routing:'加边界分流',provenance:'数据来源',comparison:'17 种形貌选型 · 四个已标定位置的同点复测',desc:'训练和评分使用相同的四个位置和三个深度，评分帧使用独立仿真噪声。0.000 mm 可表示模型选中了匹配的标定坐标。',morph:'内表面形貌',hits:'检出',falses:'空载误报',mean:'平均 XYZ（mm）',max:'最大 XYZ（mm）',ablation:'共享编码器消融',ablationDesc:'三种子开发集平均误差的相对变化。检测和定位误差下降的同时，位移场误差可能升高。',footer:'VTouch 论文 C · 仿真数据 · 存档预测回放',wall:'基准壁厚',inner:'底部名义内径',loading:'正在载入网格…',failed:'无法载入三维网格。',noWebgl:'当前浏览器无法使用 WebGL，下面的研究数据仍可查看。',dataError:'无法载入研究数据。',meanError:'XYZ 误差'}
}[lang];

for(const node of document.querySelectorAll('[data-copy]')) node.textContent=strings[node.dataset.copy];
for(const link of document.querySelectorAll('nav a')) if(link.hreflang===lang) link.setAttribute('aria-current','page');
document.querySelector('.figures img').alt=strings.ablation;

const $=id=>document.getElementById(id);
const mm=n=>n.toFixed(2)+' mm';
const formatXYZ=a=>a.map(n=>n.toFixed(2)).join(', ')+' mm';
const asset=name=>new URL('./assets/'+name,import.meta.url);
const get=async url=>{const r=await fetch(url); if(!r.ok) throw new Error('HTTP '+r.status);return r;};
const reference='REF_T18';
let data;
try { data=await (await get(asset('study.json'))).json(); }
catch(error){ $('status').textContent=strings.dataError; throw error; }

for(const model of data.meshes) $('shell').add(new Option(model[lang]+' · '+mm(model.wall),model.id));
data.cases.forEach((c,i)=>$('sample').add(new Option('s '+c.station.toFixed(1)+' mm · '+c.az+'° · '+c.depth.toFixed(2)+' mm',i)));
for(const m of data.morphologies){
  const row=$('scores').insertRow();
  for(const value of [m[lang],m.hit+'/'+m.n,m.fp+'/'+m.empty,m.mean.toFixed(3),m.max.toFixed(3)]) row.insertCell().textContent=value;
}
$('pins').textContent='Static SHA-256: '+data.sources.static_sha256+' · Morphology SHA-256: '+data.sources.morphology_sha256;

const scene=new T.Scene(), camera=new T.PerspectiveCamera(40,1,.1,1400);
camera.up.set(0,0,1);
const clipping=new T.Plane(new T.Vector3(0,1,0),0);
let renderer,controls,mesh,markers,current=0;
const material=new T.MeshStandardMaterial({color:0x81cec0,roughness:.65,metalness:.12,side:T.DoubleSide,clippingPlanes:[clipping]});
function draw(){if(renderer) renderer.render(scene,camera);}
function reset(){
  if(!controls) return;
  camera.position.set(104,-148,102);
  controls.target.set(0,0,37);
  controls.update();
  draw();
}
function resize(){
  const {clientWidth:w,clientHeight:h}=$('viewport');
  if(!renderer||!w||!h) return;
  renderer.setSize(w,h,false);
  camera.aspect=w/h;camera.updateProjectionMatrix();
  draw();
}
try {
  renderer=new T.WebGLRenderer({antialias:true,alpha:true});
  renderer.localClippingEnabled=true;
  renderer.setPixelRatio(Math.min(devicePixelRatio,2));
  $('viewport').prepend(renderer.domElement);
  scene.add(new T.HemisphereLight(0xe3fff7,0x334f71,2.0));
  const key=new T.DirectionalLight(0xfff6e3,2.4);key.position.set(60,-85,115);scene.add(key);
  const rim=new T.DirectionalLight(0x92f6eb,1.4);rim.position.set(-30,65,45);scene.add(rim);
  controls=new OrbitControls(camera,renderer.domElement);
  controls.addEventListener('change',draw);
  controls.minDistance=38;controls.maxDistance=430;
  new ResizeObserver(resize).observe($('viewport'));
  reset(); resize();
} catch(error) {$('status').textContent=strings.noWebgl; renderer=null;}

function setMarkers(c){
  if(!renderer) return;
  if(markers){scene.remove(markers);markers.traverse(o=>{o.geometry?.dispose();o.material?.dispose();});markers=null;}
  if(!c||$('shell').value!==reference||!mesh){draw();return;}
  markers=new T.Group();
  for(const [coords,color] of [[c.truth,0x37b9a0],[c.base,0xe98d50],[c.routed,0xa48de9]]){
    const point=new T.Mesh(new T.SphereGeometry(1.15,20,12),new T.MeshBasicMaterial({color,depthTest:false}));
    point.position.set(...coords);point.renderOrder=4;
    markers.add(point);
  }
  scene.add(markers);draw();
}
function updateCase(showMarks=true){
  const c=data.cases[Number($('sample').value)];
  const specs=[[strings.truth,formatXYZ(c.truth)],[strings.baseline,formatXYZ(c.base)+' · '+strings.meanError+' '+mm(c.baseError)],[strings.routing,formatXYZ(c.routed)+' · '+strings.meanError+' '+mm(c.routedError)]];
  $('result').replaceChildren(...specs.map(([label,value])=>{
    const group=document.createElement('div'),dt=document.createElement('dt'),dd=document.createElement('dd');
    dt.textContent=label;dd.textContent=value;group.append(dt,dd);return group;
  }));
  setMarkers(showMarks?c:null);
}
async function changeShell(){
  const id=$('shell').value;
  const info=data.meshes.find(row=>row.id===id);
  $('wall').textContent=strings.wall+' '+mm(info.wall);
  $('inner').textContent=strings.inner+' Ø'+mm(info.inner);
  $('wall').title='STL SHA-256: '+info.sha;
  const ticket=++current;
  if(!renderer)return;
  $('status').textContent=strings.loading;
  setMarkers(null);
  try {
    const gz=await get(asset(info.file));
    const b=await new Response(gz.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
    if(ticket!==current)return;
    const dv=new DataView(b), n=dv.getUint32(0,true), k=dv.getUint32(4,true);
    if(n<3||k%3||b.byteLength!==8+12*n+4*k) throw new Error('Invalid mesh size');
    const geometry=new T.BufferGeometry();
    geometry.setAttribute('position',new T.BufferAttribute(new Float32Array(b,8,3*n),3));
    geometry.setIndex(new T.BufferAttribute(new Uint32Array(b,8+12*n,k),1));
    geometry.computeVertexNormals();
    if(mesh){scene.remove(mesh);mesh.geometry.dispose();}
    mesh=new T.Mesh(geometry,material);scene.add(mesh);
    $('status').textContent='';
    $('viewport').dataset.mesh=id;
    updateCase();
  }catch(error){if(ticket===current)$('status').textContent=strings.failed;}
}
$('cut').addEventListener('click',()=>{
  const on=$('cut').getAttribute('aria-pressed')!=='true';
  $('cut').setAttribute('aria-pressed',String(on));
  material.clippingPlanes=on?[clipping]:[];material.needsUpdate=true;draw();
});
$('reset').addEventListener('click',reset);
$('shell').addEventListener('change',changeShell);
$('sample').addEventListener('change',()=>{
  if($('shell').value!==reference){$('shell').value=reference;changeShell();}
  updateCase();
});
updateCase(); await changeShell();

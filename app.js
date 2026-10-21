/* Salary Insight - in-browser multivariate regression on a synthetic salary dataset. */
(()=>{
const $=s=>document.querySelector(s);
const SKILLS=['React','Node.js','Python','TypeScript','AWS','Docker','Kubernetes','SQL','ML Ops','System Design'];
// ---- synthetic dataset with realistic structure ----
const ROLE_BASE={frontend:72,backend:78,fullstack:80,data:88,ml:96,devops:86,mobile:76,pm:92}; // $k base
const EDU_BONUS=[0,8,16,26], LOC_MULT=[0.72,0.88,1.0,1.28], SIZE_BONUS=[0,6,10];
const EXP_GAIN=4.6, EXP_SQ=-0.085, SKILL_BONUS=1.9;
function mulberry(a){return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
const rand=mulberry(20261021);
function sampleSalary(p){ // p: {role,edu,exp,loc,size,nskill}
  let s=ROLE_BASE[p.role]+EDU_BONUS[p.edu]+EXP_GAIN*p.exp+EXP_SQ*p.exp*p.exp;
  s=s*LOC_MULT[p.loc]+SIZE_BONUS[p.size]+SKILL_BONUS*p.nskill;
  return s+(rand()*2-1)*7; // noise
}
const DATA=[];
const roles=Object.keys(ROLE_BASE);
for(let i=0;i<600;i++){
  const p={role:roles[Math.floor(rand()*roles.length)],edu:Math.floor(rand()*4),
    exp:Math.round(rand()*40)/2,loc:Math.floor(rand()*4),size:Math.floor(rand()*3),nskill:Math.floor(rand()*9)};
  DATA.push({...p,y:sampleSalary(p)});
}
// ---- features: [1, exp, exp^2, edu, loc, size, nskill, role one-hots(7)] ----
function feats(p){const f=[1,p.exp,p.exp*p.exp,p.edu,p.loc,p.size,p.nskill];
  roles.slice(1).forEach(r=>f.push(p.role===r?1:0));return f;}
// ridge regression via normal equations with Gauss-Jordan
function train(rows){
  const n=feats(rows[0]).length,A=Array.from({length:n},()=>new Float64Array(n)),b=new Float64Array(n);
  rows.forEach(r=>{const f=feats(r);for(let i=0;i<n;i++){b[i]+=f[i]*r.y;for(let j=0;j<=i;j++)A[i][j]+=f[i]*f[j];}});
  for(let i=0;i<n;i++)for(let j=0;j<i;j++)A[j][i]=A[i][j];
  for(let i=1;i<n;i++)A[i][i]+=0.5; // ridge
  const M=A.map((r,i)=>[...r,b[i]]);
  for(let c=0;c<n;c++){let p=c;for(let r=c+1;r<n;r++)if(Math.abs(M[r][c])>Math.abs(M[p][c]))p=r;
    [M[c],M[p]]=[M[p],M[c]];const d=M[c][c];for(let j=c;j<=n;j++)M[c][j]/=d;
    for(let r=0;r<n;r++){if(r===c)continue;const k=M[r][c];for(let j=c;j<=n;j++)M[r][j]-=k*M[c][j];}}
  return M.map(r=>r[n]);
}
const t0=performance.now();
const W=train(DATA);
const trainMs=Math.max(1,Math.round(performance.now()-t0));
// metrics
let sse=0,sae=0,mean=0;DATA.forEach(r=>mean+=r.y);mean/=DATA.length;
DATA.forEach(r=>{const p=predictRaw(r);sse+=(r.y-p)**2;sae+=Math.abs(r.y-p);});
const sst=DATA.reduce((a,r)=>a+(r.y-mean)**2,0);
const R2=1-sse/sst, MAE=sae/DATA.length, residStd=Math.sqrt(sse/(DATA.length-2));
function predictRaw(p){const f=feats(p);let s=0;for(let i=0;i<W.length;i++)s+=W[i]*f[i];return Math.max(25,s);}
// ---- UI ----
const skillsEl=$('#skills'),chosen=new Set(['TypeScript','SQL']);
SKILLS.forEach(s=>{const b=document.createElement('button');b.className='chip'+(chosen.has(s)?' on':'');b.textContent=s;
  b.onclick=()=>{chosen.has(s)?chosen.delete(s):chosen.add(s);b.classList.toggle('on');};skillsEl.append(b);});
$('#exp').oninput=e=>$('#expOut').textContent=e.target.value+' yrs';
const fmt=v=>'$'+Math.round(v)+'k';
function profile(){return{role:$('#role').value,edu:+$('#edu').value,exp:+$('#exp').value,loc:+$('#loc').value,size:+$('#size').value,nskill:chosen.size};}
let trainCache={};
function roleRows(role){return DATA.filter(r=>r.role===role);}
function run(){
  const p=profile(),y=predictRaw(p);
  const lo=Math.max(20,y-1.64*residStd),hi=y+1.64*residStd;
  $('#salary').textContent=fmt(y);
  $('#range').textContent=fmt(lo)+' - '+fmt(hi);
  const peers=roleRows(p.role),med=[...peers].sort((a,b)=>a.y-b.y)[Math.floor(peers.length/2)].y;
  const d=y-med,de=$('#delta');de.textContent=(d>=0?'+':'')+fmt(Math.abs(d)).replace('$',d>=0?'$':'-$');
  de.className='delta '+(d>=0?'up':'');if(d<0)de.style.color='#dc2626';
  $('#st-r2').textContent=R2.toFixed(2);$('#st-mae').textContent='$'+MAE.toFixed(1)+'k';
  $('#st-n').textContent=DATA.length;$('#st-ms').textContent=trainMs+' ms';
  // what-if
  const y2=predictRaw({...p,exp:Math.min(25,p.exp+2)});
  $('#whatif').textContent=fmt(y2)+' (+'+fmt(y2-y).slice(1)+'k)';
  // importance: |coef| * feature std
  const names=['Experience','Experience&sup2;','Education','Location','Company size','Skills','Role'];
  const idx=[1,2,3,4,5,6,7];
  const imps=idx.map(ix=>{
    if(ix===7){ // role block: max one-hot effect
      let m=0;for(let i=7;i<W.length;i++)m=Math.max(m,Math.abs(W[i]));return m;
    }
    const vals=DATA.map(r=>feats(r)[ix]);const mu=vals.reduce((a,b)=>a+b,0)/vals.length;
    const sd=Math.sqrt(vals.reduce((a,b)=>a+(b-mu)**2,0)/vals.length);
    return Math.abs(W[ix])*sd;});
  const mx=Math.max(...imps);
  $('#imp').innerHTML=names.map((n,i)=>{
    const pct=Math.round(imps[i]/mx*100);
    return `<div class="imp-row"><span>${n}</span><div class="imp-bar"><i style="width:${pct}%"></i></div><span class="imp-val">${pct}%</span></div>`;}).join('');
  drawChart(p,y);
}
function drawChart(p,y){
  const cv=$('#chart'),ctx=cv.getContext('2d'),dpr=devicePixelRatio||1;
  const w=cv.clientWidth,h=cv.clientHeight;cv.width=w*dpr;cv.height=h*dpr;ctx.scale(dpr,dpr);
  const cs=getComputedStyle(document.documentElement);
  const ink=cs.getPropertyValue('--muted').trim(),acc=cs.getPropertyValue('--accent').trim(),grid=cs.getPropertyValue('--border').trim();
  const X=25,Y=[30,170],L=44,B=h-26,T=8,R=w-8;
  const sx=v=>L+(v/X)*(R-L),sy=v=>B-((v-Y[0])/(Y[1]-Y[0]))*(B-T);
  ctx.strokeStyle=grid;ctx.fillStyle=ink;ctx.font='10px system-ui';ctx.lineWidth=1;
  for(let g=50;g<=150;g+=50){ctx.beginPath();ctx.moveTo(L,sy(g));ctx.lineTo(R,sy(g));ctx.stroke();ctx.fillText('$'+g+'k',8,sy(g)+3);}
  for(let g=0;g<=20;g+=5){const x=sx(g);ctx.beginPath();ctx.moveTo(x,B);ctx.lineTo(x,T);ctx.globalAlpha=.35;ctx.stroke();ctx.globalAlpha=1;ctx.fillText(g+'y',x-6,h-10);}
  const peers=roleRows(p.role);
  ctx.fillStyle=ink;ctx.globalAlpha=.34;
  peers.forEach(r=>{ctx.beginPath();ctx.arc(sx(Math.min(20,r.exp)),sy(Math.max(Y[0],Math.min(Y[1],r.y))),2.4,0,7);ctx.fill();});
  ctx.globalAlpha=1;
  // fitted curve across experience for current profile
  ctx.strokeStyle=acc;ctx.lineWidth=2.5;ctx.beginPath();
  for(let e=0;e<=20;e+=0.5){const py=predictRaw({...p,exp:e});const px=sx(e),pv=sy(Math.max(Y[0],Math.min(Y[1],py)));e===0?ctx.moveTo(px,pv):ctx.lineTo(px,pv);}
  ctx.stroke();
  // user point
  ctx.fillStyle=acc;ctx.strokeStyle='#fff';ctx.lineWidth=2.5;
  ctx.beginPath();ctx.arc(sx(Math.min(20,p.exp)),sy(Math.max(Y[0],Math.min(Y[1],y))),7,0,7);ctx.fill();ctx.stroke();
}
$('#predict').onclick=()=>{run();toast('Prediction updated');};
window.addEventListener('themechange',()=>{if($('#salary').textContent!=='$–')run();});
window.addEventListener('resize',()=>{if($('#salary').textContent!=='$–')drawChart(profile(),predictRaw(profile()));});
run();
})();

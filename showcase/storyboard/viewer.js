const $=s=>document.querySelector(s);
const board=await fetch('shots.json',{cache:'no-store'}).then(r=>r.json());
const assets=await Promise.all(board.shots.map(s=>fetch(s.asset,{cache:'no-store'}).then(r=>r.text())));
await document.fonts.ready;
$('#play').disabled=false;
let current=-1,time=0,playing=false,loop=false,last=0,currentTyping=[];
const stage=$('#stage'),timeline=$('#timeline');
const format=t=>`${String(Math.floor(t/60)).padStart(2,'0')}:${String(Math.floor(t%60)).padStart(2,'0')}`;
const defaults={x:0,y:0,s:1,sx:1,sy:1,r:0,sk:0,o:1};
// Resolve annotations from rendered CSV glyphs after every camera/layer update.
// The connector and its target can live in different moving coordinate spaces.
function visibleNode(node){let opacity=1;for(let p=node;p&&p!==stage;p=p.parentElement)opacity*=Number(p.getAttribute('opacity')??1);return opacity;}
function anchor(id){const cells=[...stage.querySelectorAll('[data-cell]')].filter(n=>n.dataset.cell===id);return cells.sort((a,b)=>visibleNode(b)-visibleNode(a))[0]||stage.querySelector('#'+id);}
function localBounds(node,parent){const b=node.getBBox(),matrix=parent.getCTM().inverse().multiply(node.getCTM());const pts=[[b.x,b.y],[b.x+b.width,b.y],[b.x,b.y+b.height],[b.x+b.width,b.y+b.height]].map(([x,y])=>new DOMPoint(x,y).matrixTransform(matrix));return {x:Math.min(...pts.map(p=>p.x)),y:Math.min(...pts.map(p=>p.y)),w:Math.max(...pts.map(p=>p.x))-Math.min(...pts.map(p=>p.x)),h:Math.max(...pts.map(p=>p.y))-Math.min(...pts.map(p=>p.y))};}
function updateAnnotations(){
  for(const el of stage.querySelectorAll('[data-annotation]')){
    const parent=el.parentElement,target=anchor(el.dataset.target||el.dataset.to);
    if(!target)throw new Error('Missing CSV anchor '+(el.dataset.target||el.dataset.to));
    const b=localBounds(target,parent),kind=el.dataset.annotation;
    if(kind==='box'){
      for(const [key,value] of Object.entries({x:b.x-4,y:b.y-3,width:b.w+8,height:b.h+6}))el.setAttribute(key,value);
    }else if(kind==='label'){
      const rect=el.querySelector('rect'),w=Number(rect.getAttribute('width')),h=Number(rect.getAttribute('height')),side=el.dataset.side;
      const x=side==='right'?Math.max(1440,b.x+b.w+42):b.x+b.w/2-w/2;
      const y=side==='right'?b.y+b.h/2-h/2:side==='below'?b.y+b.h+6:b.y-h-8;
      el.setAttribute('transform',`translate(${x} ${y})`);
    }else if(kind==='squiggle'){
      const steps=Math.max(1,Math.ceil(b.w/8)),dx=b.w/steps;
      el.setAttribute('d',`M${b.x} ${b.y+b.h+3} `+Array.from({length:steps},()=>`q ${dx/2} -4 ${dx} 0`).join(' '));
    }else{
      const source=anchor(el.dataset.from);if(!source)throw new Error('Missing source '+el.dataset.from);
      const a=localBounds(source,parent);let d;
      if(el.dataset.route==='panel-left'){
        const sx=a.x+a.w,sy=a.y+a.h/2,ex=b.x-5,ey=Math.max(b.y+40,Math.min(b.y+b.h-40,sy)),middle=(sx+ex)/2;
        d=`M${sx} ${sy} C${middle} ${sy} ${middle} ${ey} ${ex} ${ey}`;
      }else if(el.dataset.route==='left'){
        const sx=a.x,sy=a.y+a.h/2,ex=b.x+b.w+5,ey=b.y+b.h/2,middle=(sx+ex)/2;
        d=`M${sx} ${sy} C${middle} ${sy} ${middle} ${ey} ${ex} ${ey}`;
      }else if(el.dataset.route==='up'){
        const sx=a.x+a.w/2,sy=a.y,ex=b.x+b.w/2,ey=b.y+b.h+5,middle=(sy+ey)/2;
        d=`M${sx} ${sy} C${sx} ${middle} ${ex} ${middle} ${ex} ${ey}`;
      }else if(el.dataset.route==='right'){
        const sx=a.x+a.w,sy=a.y+a.h/2,ex=b.x+b.w+5,ey=b.y+b.h/2,rail=1770;
        d=`M${sx} ${sy} C${rail} ${sy} ${rail} ${ey} ${ex} ${ey}`;
      }else{
        const sx=a.x+a.w/2,sy=a.y+a.h,ex=b.x+b.w/2,ey=b.y-5,middle=(sy+ey)/2;
        d=`M${sx} ${sy} C${sx} ${middle} ${ex} ${middle} ${ex} ${ey}`;
      }
      el.setAttribute('d',d);
    }
  }
}
function state(keys,t){let prev={...defaults};const expanded=keys.map(k=>{prev={...prev,...k};return {...prev}});let a=expanded[0],b=a;if(t>=expanded.at(-1).t)return expanded.at(-1);for(let i=0;i<expanded.length-1;i++){if(t>=expanded[i].t&&t<=expanded[i+1].t){a=expanded[i];b=expanded[i+1];break}}let u=Math.max(0,Math.min(1,(t-a.t)/Math.max(.0001,b.t-a.t)));u=u*u*(3-2*u);return Object.fromEntries(Object.keys(defaults).map(p=>[p,a[p]+(b[p]-a[p])*u]));}
function show(i){if(i===current)return;current=i;const s=board.shots[i];stage.innerHTML=assets[i];currentTyping=s.typing.map(item=>{const node=stage.querySelector('#'+item.id),mask=stage.querySelector('#'+item.mask),box=node.getBBox();mask.setAttribute('x',box.x-1);return {...item,node,mask,width:box.width+3,prefix:new Map([[0,0],[item.text.length,box.width+3]])};});stage.setAttribute('aria-label',s.title);$('#shot-count').textContent=`SHOT ${String(s.id).padStart(2,'0')} / 14`;$('#detail-title').textContent=s.title;$('#time').textContent=`${format(s.start)} — ${format(s.start+s.duration)} · ${s.duration} seconds`;for(const field of ['action','camera','transition','voiceover','source'])$('#detail-'+field).textContent=s[field];$('#press-frame').hidden=!(s.interactions||[]).some(e=>e.mode==='press');document.querySelectorAll('#shots button').forEach((b,n)=>b.classList.toggle('active',n===i));}
function draw(){const i=board.shots.findIndex(s=>time<s.start+s.duration);show(i<0?board.shots.length-1:i);const s=board.shots[current],t=Math.max(0,Math.min(1,(time-s.start)/s.duration));for(const m of s.motions){const el=stage.querySelector('#'+m.id);if(!el)throw new Error(`Shot ${s.id}: missing layer ${m.id}`);const p=state(m.keys,t),[ax,ay]=m.origin;el.setAttribute('transform',`translate(${p.x} ${p.y}) translate(${ax} ${ay}) rotate(${p.r}) skewX(${p.sk}) scale(${p.s*p.sx} ${p.s*p.sy}) translate(${-ax} ${-ay})`);el.setAttribute('opacity',p.o);}for(const item of currentTyping){const fraction=Math.max(0,Math.min(1,(time-item.start)/item.duration)),count=Math.floor(item.text.length*fraction);if(!item.prefix.has(count))item.prefix.set(count,item.node.getSubStringLength(0,count)+1);item.mask.setAttribute('width',fraction>=1?item.width:item.prefix.get(count));}for(const event of s.interactions||[]){if(event.orbit_id){const phase=Math.max(0,Math.min(1,(time-s.start-event.at)/.65));stage.querySelector('#'+event.orbit_id).setAttribute('stroke-dashoffset',-event.perimeter*phase);}}updateAnnotations();const handoff=Math.max(0,Math.min(1,(time-s.start)/.35)),ease=1-Math.pow(1-handoff,3);stage.querySelector('svg').style.transform=`translateY(${14*(1-ease)}px)`;timeline.value=time;$('#clock').textContent=`${format(time)} / ${format(board.runtime)}`;}
function toggle(value=!playing){playing=value;$('#play').textContent=playing?'Ⅱ Pause':'▶ Play film';$('#play').setAttribute('aria-label',playing?'Pause storyboard':'Play full storyboard');last=0;if(playing&&time>=board.runtime)time=0;}
function seek(value){toggle(false);time=Math.max(0,Math.min(board.runtime,value));draw();}
board.shots.forEach((s,i)=>{const b=document.createElement('button');b.innerHTML=`<img src="assets/shot-${String(s.id).padStart(2,'0')}-action.svg?v=7" alt=""><span class="shot-label"><b>${String(s.id).padStart(2,'0')} / ${format(s.start)}</b>${s.title}</span>`;b.setAttribute('aria-label',`Select shot ${s.id}: ${s.title}`);b.onclick=()=>{seek(s.start+s.duration*.58);stage.scrollIntoView({behavior:'instant',block:'start'});};$('#shots').append(b);});
$('#press-frame').onclick=()=>{const s=board.shots[current],event=s.interactions.find(e=>e.mode==='press');if(event)seek(s.start+event.at+.325);};
$('#play').onclick=()=>toggle();$('#loop').onclick=()=>{loop=!loop;$('#loop').setAttribute('aria-pressed',loop);};$('#restart').onclick=()=>seek(0);timeline.oninput=()=>seek(Number(timeline.value));document.querySelectorAll('[data-phase]').forEach(b=>b.onclick=()=>{const s=board.shots[current];seek(s.start+s.duration*Number(b.dataset.phase)-(b.dataset.phase==='1'?.001:0));});
document.addEventListener('keydown',e=>{if(e.target.matches('input,textarea'))return;if(e.code==='Space'){e.preventDefault();toggle();}if(e.code==='ArrowRight'){e.preventDefault();seek(time+1);}if(e.code==='ArrowLeft'){e.preventDefault();seek(time-1);}});
function frame(now){if(playing){if(last){const dt=Math.min(.1,(now-last)/1000);time+=dt;const s=board.shots[current];if(loop&&time>=s.start+s.duration)time=s.start;if(time>=board.runtime){time=board.runtime;toggle(false)}}draw();}last=now;requestAnimationFrame(frame);}
timeline.max=board.runtime;
const requested=Math.max(1,Math.min(14,Number(new URLSearchParams(location.search).get('shot')||6)));time=board.shots[requested-1].start+board.shots[requested-1].duration*.58;draw();requestAnimationFrame(frame);

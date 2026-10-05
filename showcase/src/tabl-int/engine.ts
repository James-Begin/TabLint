import {film} from './assets';
import {timing} from './timing';

const defaults = {x:0,y:0,s:1,sx:1,sy:1,r:0,sk:0,o:1};
type State = typeof defaults;
type Key = Partial<State> & {t:number};
type Shot = typeof film.shots[number];
const clamp=(n:number)=>Math.max(0,Math.min(1,n));
function state(keys:Key[],t:number):State {
  let prev:State & {t:number}={...defaults,t:0};
  const expanded=keys.map(k=>(prev={...prev,...k}));
  let a=expanded[0],b=a;
  if(t>=expanded[expanded.length-1].t)return expanded[expanded.length-1];
  for(let i=0;i<expanded.length-1;i++)if(t>=expanded[i].t&&t<=expanded[i+1].t){a=expanded[i];b=expanded[i+1];break;}
  let u=clamp((t-a.t)/Math.max(.0001,b.t-a.t));u=u*u*(3-2*u);
  const result={...defaults};
  for(const p of Object.keys(defaults) as (keyof State)[])result[p]=a[p]+(b[p]-a[p])*u;
  return result;
}
function bounds(node:SVGGraphicsElement,parent:SVGGraphicsElement){
  const b=node.getBBox(),matrix=parent.getCTM()!.inverse().multiply(node.getCTM()!);
  const pts=[[b.x,b.y],[b.x+b.width,b.y],[b.x,b.y+b.height],[b.x+b.width,b.y+b.height]].map(([x,y])=>new DOMPoint(x,y).matrixTransform(matrix));
  const x=Math.min(...pts.map(p=>p.x)),y=Math.min(...pts.map(p=>p.y));
  return {x,y,w:Math.max(...pts.map(p=>p.x))-x,h:Math.max(...pts.map(p=>p.y))-y};
}
export function createEngine(stage:HTMLDivElement,shot:Shot,prefix='',audit=false){
  const query=<T extends Element>(id:string)=>{
    const node=stage.querySelector<T>('#'+prefix+id);
    if(!node)throw new Error(`Shot ${shot.id}: missing ${id}`);
    return node;
  };
  const typing=shot.typing.map(item=>{
    const node=query<SVGTextContentElement>(item.id),mask=query<SVGRectElement>(item.mask),box=node.getBBox();
    mask.setAttribute('x',String(box.x-1));
    return {...item,node,mask,width:box.width+3,prefix:new Map([[0,0],[item.text.length,box.width+3]])};
  });
  const motions=shot.motions.map(m=>({...m,node:query<SVGGElement>(m.id)}));
  const typedNodes=new Map(typing.map(item=>[item.node,item]));
  const annotations=Array.from(stage.querySelectorAll<SVGGraphicsElement>('[data-annotation]'));
  const cells=Array.from(stage.querySelectorAll<SVGGraphicsElement>('[data-cell]'));
  const visible=(node:Element)=>{let opacity=1;for(let p:Element|null=node;p&&p!==stage;p=p.parentElement)opacity*=Number(p.getAttribute('opacity')??1);return opacity;};
  const anchor=(id:string)=>cells.filter(n=>n.dataset.cell===id).sort((a,b)=>visible(b)-visible(a))[0]||query<SVGGraphicsElement>(id);
  const glyphReady=(target:SVGGraphicsElement)=>{
    for(let node:Element|null=target;node&&node!==stage;node=node.parentElement){
      const item=typedNodes.get(node as SVGTextContentElement);
      if(item){
        const b=bounds(target,item.node);
        return b.x+b.w<=Number(item.mask.getAttribute('x'))+Number(item.mask.getAttribute('width'));
      }
    }
    return true;
  };
  return (time:number,movieSeconds=0)=>{
    const local=time-shot.start,t=clamp(local/shot.duration);
    for(const m of motions){
      const p=state(m.keys,t),[ax,ay]=m.origin;
      m.node.setAttribute('transform',`translate(${p.x} ${p.y}) translate(${ax} ${ay}) rotate(${p.r}) skewX(${p.sk}) scale(${p.s*p.sx} ${p.s*p.sy}) translate(${-ax} ${-ay})`);
      m.node.setAttribute('opacity',String(p.o));
    }
    for(const item of typing){
      const fraction=clamp((time-item.start)/item.duration),count=Math.floor(item.text.length*fraction);
      if(!item.prefix.has(count))item.prefix.set(count,item.node.getSubStringLength(0,count)+1);
      item.mask.setAttribute('width',String(fraction>=1?item.width:item.prefix.get(count)));
    }
    for(const event of shot.interactions||[])if(event.orbit_id)query<SVGRectElement>(event.orbit_id).setAttribute('stroke-dashoffset',String(-event.perimeter*clamp((local-event.at)/.65)));
    if(shot.id===5){
      // Loading has its own clock and leads straight into the findings, without reading holds.
      query<SVGRectElement>('loading-fill').setAttribute('width',String(681*clamp((movieSeconds-timing.loadingStartSeconds)/timing.loadingDurationSeconds)));
    }
    if(shot.id===7){
      const raw=clamp((local-5.15)/1.5),p=raw*raw*(3-2*raw);
      const x=726+(1420-726)*p,value=Math.round(1877+(4354-1877)*p);
      query<SVGGElement>('range-beam').setAttribute('transform',`rotate(${3.2*p} 726 888)`);
      const dot=query<SVGCircleElement>('weight-slider');
      dot.setAttribute('cx',String(x));dot.setAttribute('r',String(7+4*p));
      const counter=query<SVGTextElement>('weight-counter');
      counter.setAttribute('x',String(x));counter.textContent=value.toLocaleString('en-US');
    }
    for(const el of annotations){
      const parent=el.parentElement as unknown as SVGGraphicsElement,target=anchor((el.dataset.target||el.dataset.to)!);
      const b=bounds(target,parent),kind=el.dataset.annotation;
      // A box, underline or leader cannot precede the actual CSV glyph it references.
      el.setAttribute('opacity',glyphReady(target)&&visible(target)>.001?'1':'0');
      if(kind==='box'){
        for(const [key,value] of Object.entries({x:b.x-4,y:b.y-3,width:b.w+8,height:b.h+6}))el.setAttribute(key,String(value));
      }else if(kind==='label'){
        const rect=el.querySelector('rect')!,w=Number(rect.getAttribute('width')),h=Number(rect.getAttribute('height')),side=el.dataset.side;
        const x=side==='right'?Math.max(1440,b.x+b.w+42):b.x+b.w/2-w/2;
        const gap=Number(el.dataset.gap??(side==='below'?6:8));
        const y=side==='right'?b.y+b.h/2-h/2:side==='below'?b.y+b.h+gap:b.y-h-gap;
        el.setAttribute('transform',`translate(${x} ${y})`);
      }else if(kind==='squiggle'){
        const steps=Math.max(1,Math.ceil(b.w/8)),dx=b.w/steps;
        el.setAttribute('d',`M${b.x} ${b.y+b.h+3} `+Array.from({length:steps},()=>`q ${dx/2} -4 ${dx} 0`).join(' '));
      }else{
        const a=bounds(anchor(el.dataset.from!),parent);let d:string;
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
          const sx=a.x+a.w,sy=a.y+a.h/2,ex=b.x+b.w+5,ey=b.y+b.h/2;
          d=`M${sx} ${sy} C1770 ${sy} 1770 ${ey} ${ex} ${ey}`;
        }else{
          const sx=a.x+a.w/2,sy=a.y+a.h,ex=b.x+b.w/2,ey=b.y-5,middle=(sy+ey)/2;
          d=`M${sx} ${sy} C${sx} ${middle} ${ex} ${middle} ${ex} ${ey}`;
        }
        el.setAttribute('d',d);
      }
    }
    if(audit){
      const issues:string[]=[];
      for(const el of annotations){
        const target=anchor((el.dataset.target||el.dataset.to)!);
        if(visible(el)>.001&&!glyphReady(target))issues.push('Highlight precedes CSV value: '+target.dataset.cell);
      }
      const panels=new Set(['car-label','horsepower-label','weight-label','hover','technical-card','ledger-explainer','ledger','undone','reapplied','reapply']);
      for(const item of typing){
        if(visible(item.node)<.1)continue;
        let holder:Element|null=item.node.parentElement;
        while(holder&&holder!==stage&&!panels.has(holder.id.replace(prefix,'')))holder=holder.parentElement;
        if(!holder||holder===stage)continue;
        const rect=holder.querySelector<SVGRectElement>('rect');if(!rect)continue;
        const b=bounds(item.node,rect.parentElement as unknown as SVGGraphicsElement),r=rect.getBBox();
        if(b.x<r.x-1||b.y<r.y-1||b.x+b.w>r.x+r.width+1||b.y+b.h>r.y+r.height+1)issues.push('Text exceeds panel: '+item.id);
      }
      if([7,8,11].includes(shot.id)){
        const hover=query<SVGGElement>('hover');
        if(visible(hover)>.01){
          const svg=stage.querySelector('svg')!;
          const target=anchor(shot.id===7?'csv-183-weight':shot.id===8?'csv-177-horsepower':'csv-16-origin');
          const a=bounds(target,svg),b=bounds(hover,svg);
          if(a.x<b.x+b.w&&a.x+a.w>b.x&&a.y<b.y+b.h&&a.y+a.h>b.y)issues.push('Diagnosis covers referenced value');
        }
      }
      let scale;
      if(shot.id===7){
        const counter=query<SVGTextElement>('weight-counter'),dot=query<SVGCircleElement>('weight-slider');
        scale={value:counter.textContent,x:Number(dot.getAttribute('cx')),beam:query<SVGGElement>('range-beam').getAttribute('transform')};
        const strip=query<SVGGElement>('range'),rect=strip.querySelector('rect')!;
        if(visible(strip)>.01){
          const a=bounds(counter,strip),b=rect.getBBox();
          if(a.x<b.x||a.x+a.w>b.x+b.width||a.y<b.y||a.y+a.h>b.y+b.height)issues.push('Moving scale label exceeds panel');
        }
      }
      const categories=shot.id===10?annotations.map(el=>({kind:el.dataset.annotation,target:el.dataset.target||el.dataset.to,opacity:visible(el),ready:glyphReady(anchor((el.dataset.target||el.dataset.to)!))})):undefined;
      const loading=shot.id===5?{width:Number(query<SVGRectElement>('loading-fill').getAttribute('width')),revealOpacity:visible(query<SVGGElement>('reveal'))}:undefined;
      const categoryPanels=shot.id===11?{suggestion:visible(query<SVGGElement>('hover')),technical:visible(query<SVGGElement>('technical-card'))}:undefined;
      if(categoryPanels&&categoryPanels.suggestion<.001&&categoryPanels.technical>.001)issues.push('Technical card outlives categorical suggestion');
      console.log('TABLINT_AUDIT '+JSON.stringify({frame:timing.scenes[shot.id-1].startFrame+Math.round(movieSeconds*30),scene:shot.id,source:local,issues,scale,categories,loading,categoryPanels}));
    }
  };
}

import React,{useLayoutEffect,useMemo,useRef,useState} from 'react';
import {AbsoluteFill,Audio,Sequence,continueRender,delayRender,getInputProps,staticFile,useCurrentFrame} from 'remotion';
import {loadFont} from '@remotion/fonts';
import {film,layers} from './tabl-int/assets';
import {createEngine} from './tabl-int/engine';
import {timing} from './tabl-int/timing';

export function sourceTime(index:number,frame:number){
  const scene=timing.scenes[index];
  for(const segment of scene.segments){
    if(frame<segment.frame)return segment.sourceFrom;
    if(frame<segment.frame+segment.frames){
      const p=(frame-segment.frame)/segment.frames;
      return segment.sourceFrom+(segment.sourceTo-segment.sourceFrom)*p;
    }
  }
  return scene.segments[scene.segments.length-1].sourceTo;
}

const fontReady=loadFont({family:'Inter',url:staticFile('fonts/inter-latin-wght-normal.woff2'),weight:'100 900'});
const Scene:React.FC<{index:number}>=({index})=>{
  const frame=useCurrentFrame(),shot=film.shots[index];
  const stage=useRef<HTMLDivElement>(null),engine=useRef<ReturnType<typeof createEngine>|null>(null);
  const [handle]=useState(()=>delayRender('Preparing TabLint SVG typography'));
  const [ready,setReady]=useState(false);
  const markup=useMemo(()=>({__html:layers[index].replace(/id="([^"]+)"/g,(_,id)=>`id="s${index}-${id}"`).replace(/url\(#([^)]+)\)/g,(_,id)=>`url(#s${index}-${id})`)}),[index]);
  useLayoutEffect(()=>{let active=true;Promise.resolve(fontReady).then(()=>document.fonts.ready).then(()=>{if(active)setReady(true);});return ()=>{active=false;};},[]);
  useLayoutEffect(()=>{
    if(!ready)return;
    if(!engine.current)engine.current=createEngine(stage.current!,shot,`s${index}-`,getInputProps().audit===true);
    engine.current(shot.start+sourceTime(index,frame),frame/30);
    continueRender(handle);
  },[ready,frame,handle,shot,index]);
  const p=index===0?1:Math.max(0,Math.min(1,frame/timing.transitionFrames));
  return <AbsoluteFill style={{background:'#f6f8f6',opacity:p*p*(3-2*p)}}><div ref={stage} style={{width:1920,height:1080}} dangerouslySetInnerHTML={markup}/></AbsoluteFill>;
};

export const OpenFile=()=> <Scene index={0}/>;
export const InspectFields=()=> <Scene index={1}/>;
export const CompareRows=()=> <Scene index={2}/>;
export const RunTabLint=()=> <Scene index={3}/>;
export const Analysis=()=> <Scene index={4}/>;
export const RowContext=()=> <Scene index={5}/>;
export const WeightDiagnosis=()=> <Scene index={6}/>;
export const MissingValue=()=> <Scene index={7}/>;
export const ApplyFix=()=> <Scene index={8}/>;
export const Categories=()=> <Scene index={9}/>;
export const FlagReview=()=> <Scene index={10}/>;
export const OpenLedger=()=> <Scene index={11}/>;
export const UndoReapply=()=> <Scene index={12}/>;
export const TabLintClose=()=> <Scene index={13}/>;

export const TabLintDemo:React.FC=()=> <AbsoluteFill style={{background:'#f6f8f6'}}>
  <Sequence from={0} durationInFrames={170} name="01 Open file"><OpenFile/></Sequence>
  <Sequence from={152} durationInFrames={84} name="02 Inspect fields"><InspectFields/></Sequence>
  <Sequence from={218} durationInFrames={163} name="03 Compare rows"><CompareRows/></Sequence>
  <Sequence from={363} durationInFrames={171} name="04 Run TabLint"><RunTabLint/></Sequence>
  <Sequence from={516} durationInFrames={109} name="05 Analysis"><Analysis/></Sequence>
  <Sequence from={607} durationInFrames={133} name="06 Row context"><RowContext/></Sequence>
  <Sequence from={722} durationInFrames={311} name="07 Weight diagnosis"><WeightDiagnosis/></Sequence>
  <Sequence from={1015} durationInFrames={172} name="08 Missing value"><MissingValue/></Sequence>
  <Sequence from={1169} durationInFrames={236} name="09 Apply fix"><ApplyFix/></Sequence>
  <Sequence from={1387} durationInFrames={160} name="10 Categorical columns"><Categories/></Sequence>
  <Sequence from={1529} durationInFrames={366} name="11 Flag for review"><FlagReview/></Sequence>
  <Sequence from={1877} durationInFrames={268} name="12 Open ledger"><OpenLedger/></Sequence>
  <Sequence from={2127} durationInFrames={404} name="13 Undo and reapply"><UndoReapply/></Sequence>
  <Sequence from={2513} durationInFrames={120} name="14 TabLint"><TabLintClose/></Sequence>
  <Audio src={staticFile('tabl-int-score.wav')}/>
</AbsoluteFill>;

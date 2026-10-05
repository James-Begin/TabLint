const {bundle}=require('@remotion/bundler');
const {selectComposition,renderMedia}=require('@remotion/renderer');
const fs=require('fs');const path=require('path');
(async()=>{
 const directory=path.resolve('review');
 const output=fs.createWriteStream(path.join(directory,'frame-audit.jsonl'));
 const inputProps={audit:true};
 const serveUrl=await bundle({entryPoint:path.resolve('src/index.ts'),rspack:true});
 const composition=await selectComposition({serveUrl,id:'TabLintDemo',inputProps});
 let last=-1;
 await renderMedia({serveUrl,composition,inputProps,outputLocation:path.resolve('../demo/showcase/TabLint-demo.mp4'),codec:'h264',crf:18,pixelFormat:'yuv420p',audioCodec:'aac',concurrency:4,
  onBrowserLog(log){if(log.text.startsWith('TABLINT_AUDIT '))output.write(log.text.slice(14)+'\n');},
  onProgress(p){const mark=Math.floor(p.progress*10);if(mark!==last){last=mark;console.log('Render progress',mark*10+'%');}}
 });
 await new Promise(resolve=>output.end(resolve));
 console.log('Rendered',composition.durationInFrames,'frames with SVG audit logs.');
})().catch(e=>{console.error(e);process.exit(1)});

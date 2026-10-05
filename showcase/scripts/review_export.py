from pathlib import Path
import os,subprocess,json,math,hashlib
import xml.etree.ElementTree as ET
from audit_timing_continuity import audit as audit_continuity
from PIL import Image,ImageDraw
root=Path(__file__).resolve().parents[1];rev=root/'review';video=root.parent/'demo/showcase/TabLint-demo.mp4'
ff=Path(os.environ.get('FFMPEG', 'ffmpeg'));env=os.environ.copy()
continuity=audit_continuity(rev)
t=json.loads((rev/'timing.json').read_text());audit=[json.loads(x) for x in (rev/'frame-audit.jsonl').read_text().splitlines() if x.strip()]
frames={r['frame'] for r in audit};assert set(range(t['totalFrames']))<=frames
assert not [r for r in audit if r['issues']]
scale=sorted([r for r in audit if r['scene']==7],key=lambda r:r['frame'])
assert all(float(a['scale']['x'])<=float(b['scale']['x']) for a,b in zip(scale,scale[1:]))
assert scale[0]['scale']['value']=='1,877' and scale[-1]['scale']['value']=='4,354'
assert all(float(r['scale']['beam'].split('(')[1].split(' ')[0])>=0 for r in scale)
assert all(c['ready'] or c['opacity']==0 for r in audit for c in r.get('categories',[]))
assert all(r['categoryPanels']['suggestion']>.001 or r['categoryPanels']['technical']<=.001 for r in audit if r.get('categoryPanels'))
assert all(p['frames']==54 for scene in t['scenes'] for p in scene['segments'] if p['readingPause'])
assert not any(p['readingPause'] for p in t['scenes'][4]['segments'])
loading=sorted([r for r in audit if r['scene']==5],key=lambda r:r['frame'])
assert all(a['loading']['width']<=b['loading']['width'] for a,b in zip(loading,loading[1:]))
assert all(r['loading']['width']>=680.99 for r in loading if r['loading']['revealOpacity']>.001)
source=json.loads((rev/'source-shots.json').read_text())
camera=next(m for m in source['shots'][6]['motions'] if m['id']=='camera')
assert all(k['s']==1.9 and k['x']==90 and k['y']==-220 for k in camera['keys'] if k['t']>=1.2/9.4)
for shot,removed in [(8,'warning-track'),(14,'workspace')]:
 tree=ET.fromstring((rev/f'shot-{shot:02d}-layers.svg').read_text())
 assert not any(n.get('id')==removed for n in tree.iter())
# Decode every encoded frame; retain full frame-by-frame contact sheets and detail windows.
windows={f'transition-{i:02d}':list(range(s['startFrame']-3,s['startFrame']+27)) for i,s in enumerate(t['scenes'][1:],1)}
s=t['scenes'][6];scale_start=s['startFrame']+18+round(4.8*30)
windows['scale-entrance']=list(range(scale_start,scale_start+90))
s=t['scenes'][9];windows['categorical-entrance']=list(range(s['startFrame'],s['startFrame']+100))
s=t['scenes'][6];windows['weight-closeup']=list(range(s['startFrame']+30,s['startFrame']+180))
s=t['scenes'][7];windows['missing-value-popup']=list(range(s['startFrame'],s['startFrame']+110))
s=t['scenes'][13];windows['clean-outro']=list(range(s['startFrame']-10,s['startFrame']+110))
s=t['scenes'][4];windows['loading-to-findings']=list(range(s['startFrame'],t['scenes'][5]['startFrame']+45))
windows['opening']=list(range(0,t['scenes'][2]['startFrame']+30))
s=t['scenes'][10];exit_start=s['startFrame']+s['segments'][2]['frame']
windows['categorical-popup-exit']=list(range(exit_start-12,exit_start+108))
selected={frame for window in windows.values() for frame in window};thumbs={}
folder=rev/'all-frames';folder.mkdir(exist_ok=True)
width,height=320,180;stride=width*height*3
decoded_dir=rev/'decoded-frames';decoded_dir.mkdir(exist_ok=True)
result=subprocess.run([str(ff),'-v','error','-i',str(video),'-vf',f'scale={width}:{height}','-an','-y',str(decoded_dir/'frame-%04d.png')],env=env,capture_output=True,text=True)
assert result.returncode==0 and not result.stderr,(result.returncode,result.stderr)
canvas=None;draw=None;decoded=0
for image_path in sorted(decoded_dir.glob('frame-*.png')):
 im=Image.open(image_path).convert('RGB')
 if decoded in selected:thumbs[decoded]=im.copy()
 if decoded%30==0:
  canvas=Image.new('RGB',(1920,990),'#f0f4f1');draw=ImageDraw.Draw(canvas)
 n=decoded%30;x=n%6*320;y=n//6*198;canvas.paste(im,(x,y));draw.text((x+6,y+182),f'{decoded:04d} / {decoded/30:.3f}s',fill='#15352b')
 if n==29:canvas.save(folder/f'second-{decoded//30:03d}.jpg',quality=92)
 decoded+=1
if decoded%30:canvas.save(folder/f'second-{(decoded-1)//30:03d}.jpg',quality=92)

assert decoded==t['totalFrames']
for name,window in windows.items():
 for page in range(math.ceil(len(window)/30)):
  tile=Image.new('RGB',(1920,990),'#f0f4f1');d=ImageDraw.Draw(tile)
  for n,frame in enumerate(window[page*30:(page+1)*30]):
   x=n%6*320;y=n//6*198;tile.paste(thumbs[frame],(x,y));d.text((x+6,y+182),f'{frame:04d} / {frame/30:.3f}s',fill='#15352b')
  tile.save(rev/f'{name}-{page+1:02d}.jpg',quality=95)
manifest={'file':video.name,'sha256':hashlib.sha256(video.read_bytes()).hexdigest(),'runtime_seconds':t['runtime'],'frames':decoded,'width':1920,'height':1080,'fps':30,'revision':15,'checks':{'all_frames_decoded':decoded,'all_frames_layout_audited':len(frames),'layout_issues':0,'reading_pause_seconds':1.8,'category_highlights_wait_for_glyphs':True,'scale_value_and_position_monotonic':True,'scale_tilts_only_toward_heavier_side':True},'categorical_example':'Illustrative suggestion; this specific USA to Japan suggestion is not a captured model run.'}
manifest['checks'].update(weight_closeup_held_through_diagnosis=True,missing_value_connector_removed=True,closing_workspace_removed=True)
manifest['checks'].update(loading_scene_seconds=t['scenes'][4]['contentFrames']/30,loading_has_no_reading_holds=True,findings_follow_full_loading_bar=True)
manifest['checks'].update(opening_data_scenes_seconds=t['scenes'][2]['startFrame']/30,categorical_technical_card_exits_with_suggestion=True)
manifest['checks']['source_time_cuts_skip_no_visible_motion']=continuity['visible_motion_jumps']==0
manifest['music']={'style':'Original electronic groove with warm keys, bass, light drums and stereo synths','bpm':112,'sampled_recordings':False}
(root.parent/'demo/showcase/export.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('All',decoded,'frames decoded and checked; all-frame sheets and sequential detail windows saved.')

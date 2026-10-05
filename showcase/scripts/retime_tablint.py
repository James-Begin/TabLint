from pathlib import Path
import json,copy,xml.etree.ElementTree as ET,math,html
root=Path(__file__).resolve().parents[1]; board=root/'storyboard'; export=root/'review'
export.mkdir(exist_ok=True)
film=json.loads((board/'shots.json').read_text()); layers=[(board/s['asset']).read_text() for s in film['shots']]
D={'x':0,'y':0,'s':1,'sx':1,'sy':1,'r':0,'sk':0,'o':1}
def motion(s,id):return next(m for m in s['motions'] if m['id']==id)
def expanded(m):
 p=copy.copy(D); out=[]
 for k in m['keys']:p={**p,**k};out.append(copy.copy(p))
 return out
# Carry the previous workspace pose into the next scene; no boundary camera reset.
camids=['screen','camera','background','camera','screen','background','camera','camera','camera','camera','camera','screen','screen','workspace']
last=None
for i,s in enumerate(film['shots']):
 m=motion(s,camids[i]);keys=expanded(m)
 if last is not None:
  # Retain authored destination, but match the incoming camera exactly.
  target=keys[1] if len(keys)>1 else keys[0]
  if i==7:target={**keys[-1],'t':.65/s['duration']}
  m['keys']=[{**keys[0],**{k:last[k] for k in D if k!='o'},'t':0}, {**target,'t':.65/s['duration']}]
  # No decorative zoom drift after settling. The final brand reveal remains authored.
  if i==13:m['keys']+=keys[2:]
  else:m['keys']+=[{**target,'t':1}]
 last=expanded(m)[-1]
# Stop extra row offset in scene 2, so CSV glyphs don't snap at the handoff.
s=film['shots'][1];motion(s,'rows')['keys']=[{'t':0,'o':1,'y':0},{'t':1,'o':1,'y':0}]
# Remove the optional label-picker step and the decorative ledger Why badge.
ET.register_namespace('', 'http://www.w3.org/2000/svg')
def edit_svg(index,edit):
    tree=ET.fromstring(layers[index]);edit(tree)
    layers[index]=ET.tostring(tree,encoding='unicode')
    ids={n.get('id') for n in tree.iter() if n.get('id')}
    shot=film['shots'][index]
    shot['motions']=[m for m in shot['motions'] if m['id'] in ids]
    shot['typing']=[t for t in shot['typing'] if t['id'] in ids]
    shot['interactions']=[e for e in shot.get('interactions',[]) if e.get('orbit_id') in ids]
# Opening labels are presentation typography, independently sized from CSV text.
def opening_titles(tree):
    for name,width in [('car-label',184),('horsepower-label',210),('weight-label',136)]:
        group=next(n for n in tree.iter() if n.get('id')==name)
        group.set('data-gap','1')
        rect=next(n for n in group if n.tag.endswith('rect'))
        rect.set('width',str(width));rect.set('height','36')
        text=next(n for n in group if n.tag.endswith('text'))
        text.set('font-size','28');text.set('font-weight','650')
        text.set('x','14');text.set('y','28')
        item=next(t for t in film['shots'][1]['typing'] if t['id']==text.get('id'))
        item.update(font=28,x=14,width=width-28)
        mask=next(n for n in tree.iter() if n.get('id')==item['mask'])
        mask.set('x','13');mask.set('y','0');mask.set('width',str(width-26));mask.set('height','36')
edit_svg(1,opening_titles)
def remove_groups(index,names):
    def edit(tree):
        for parent in tree.iter():
            for node in list(parent):
                if node.get('id') in names:parent.remove(node)
    edit_svg(index,edit)
remove_groups(3,{'picker','label-selection'})
remove_groups(11,{'receipt'})
remove_groups(7,{'track'})
film['shots'][3]['action']='Select TabLint: Check this CSV; begin analysis directly.'
# The loading fill spans the entire track and grows from its left edge.
def loading(tree):
    bar=next(n for n in tree.iter() if n.get('id')=='progress')
    fill=list(bar)[0];fill.set('id','loading-fill');fill.set('width','681')
edit_svg(4,loading)
motion(film['shots'][4],'progress')['keys']=[{'t':0,'sx':1},{'t':1,'sx':1}]
# Hold the Civic close-up throughout its diagnosis; the popup remains at screen scale.
shot=film['shots'][6];original=shot['duration'];shot['duration']=9.4
for m in shot['motions']:
    for k in m['keys']:k['t']=(k['t']*original+(0 if m['id']=='camera' else 2.4))/shot['duration']
for item in shot['typing']:
    if item['start']>=shot['start']:item['start']+=2.4
camera=motion(shot,'camera');incoming=camera['keys'][0]
camera['keys']=[{**incoming,'t':0},{**incoming,'t':.5/9.4},
 {'t':1.2/9.4,'s':1.9,'x':90,'y':-220,'o':1},
 {'t':1,'s':1.9,'x':90,'y':-220,'o':1}]
# The next scene inherits this final pose before its own camera move.
motion(film['shots'][7],'camera')['keys'][0].update(s=1.9,x=90,y=-220)
def raise_weight_popup(tree):
    ns='{http://www.w3.org/2000/svg}'
    hover=next(n for n in tree.iter() if n.get('id')=='hover')
    content=ET.Element(ns+'g',{'transform':'translate(0 -70)'})
    for node in list(hover):hover.remove(node);content.append(node)
    hover.append(content)
edit_svg(6,raise_weight_popup)
# The cell outline appears throughout the close-up, before the hover opens.
focus=motion(shot,'weight-focus');focus['keys'][0]['t']=0
focus['keys'][1]['t']=.3/9.4;focus['keys'][2]['t']=.6/9.4
# Reveal a level strip, then slide the recorded value away from the plausible range.
numberline=motion(shot,'range');numberline['origin']=[960,888]
numberline['keys']=[{'t':0,'o':0,'y':35,'r':0},
 {'t':4.8/9.4,'o':0,'y':35,'r':0},
 {'t':5.1/9.4,'o':1,'y':0,'r':0},
 {'t':8.7/9.4,'o':1,'y':0,'r':0},
 {'t':1,'o':0,'y':20,'r':0}]
def sliding_scale(tree):
    ns='{http://www.w3.org/2000/svg}'
    strip=next(n for n in tree.iter() if n.get('id')=='range')
    backing=list(strip)[0];backing.set('height','150')
    beam=ET.Element(ns+'g',{'id':'range-beam'})
    for node in list(strip)[1:]:strip.remove(node);beam.append(node)
    strip.append(beam)
    band=next(n for n in beam if n.tag==ns+'path' and n.get('stroke')=='#087f78')
    band.set('d','M682.010 888 H769.990')
    green_label=next(n for n in beam if n.tag==ns+'text' and n.get('fill')=='#087f78')
    green_label.set('x','726');green_label.set('text-anchor','middle')
    dot=next(n for n in beam if n.tag==ns+'circle' and n.get('fill')=='#d65961')
    dot.set('id','weight-slider');dot.set('cx','726');dot.set('r','7')
    label=next(n for n in beam if n.tag==ns+'text' and n.get('fill')=='#d65961')
    label.set('id','weight-counter');label.set('x','726');label.set('y','918')
    label.attrib.pop('clip-path',None);label.text='1,877'
edit_svg(6,sliding_scale)
# Category callout text starts only after its referenced CSV value has streamed in.
shot=film['shots'][9];tree=ET.fromstring(layers[9]);parents={c:p for p in tree.iter() for c in p}
labels={'USA':'csv-16-origin','Japan':'csv-20-origin','Europe':'csv-21-origin'}
for item in shot['typing']:
    if item['text'] not in labels:continue
    anchor=next(n for n in tree.iter() if n.get('data-cell')==labels[item['text']])
    parent=anchor
    while parent.get('id') not in {x['id'] for x in shot['typing']}:parent=parents[parent]
    row=next(x for x in shot['typing'] if x['id']==parent.get('id'))
    item['start']=row['start']+row['duration']+.1
# Keep one current replacement card. Preserve undo/reapply as concise history.
def history(index,group):
    def edit(tree):
        parent=next(n for n in tree.iter() if n.get('id')==group)
        for node in list(parent):
            if (float(node.get('x','0'))>=1134 and 553<=float(node.get('y','0'))<=800) or node.get('d','').startswith('M1192 714'):
                parent.remove(node)
        ns='{http://www.w3.org/2000/svg}'
        defs=tree.find(ns+'defs')
        ET.SubElement(parent,ns+'path',{'d':'M1170 570 H1743','stroke':'#dce3e7','fill':'none'})
        for n,(text,y,size,color) in enumerate([('HISTORY',608,15,'#687782'),('Undone · restored 0.0',648,20,'#ad771b'),('Reapplied · current value 70',686,20,'#087f78')]):
            id=f'history-text-{n}';mask=f'history-mask-{n}';clip=f'history-clip-{n}'
            cp=ET.SubElement(defs,ns+'clipPath',{'id':clip,'clipPathUnits':'userSpaceOnUse'})
            ET.SubElement(cp,ns+'rect',{'id':mask,'x':'1190','y':str(y-27),'width':'530','height':'36'})
            node=ET.SubElement(parent,ns+'text',{'id':id,'x':'1192','y':str(y),'font-size':str(size),'font-family':'Inter, Arial, sans-serif','font-weight':'550','fill':color,'clip-path':f'url(#{clip})'})
            node.text=text
            film['shots'][index]['typing'].append({'id':id,'mask':mask,'text':text,'start':59+6.55+n*.22,'duration':max(.22,len(text)/42),'x':1192,'width':530,'font':size,'mono':False,'bold':False})
    edit_svg(index,edit)
history(12,'reapplied');history(13,'workspace')
# A clean background fades over the outgoing ledger; no replica remains behind the title.
remove_groups(13,{'workspace'})
shot=film['shots'][12];old=shot['duration'];shot['duration']=9.6
for m in shot['motions']:
    for k in m['keys']:k['t']*=old/shot['duration']
for id in ['ledger-explainer','ledger-leader']:
    keys=motion(shot,id)['keys'];keys[-2]['t']=9.2/9.6;keys[-1]['t']=1
# The ledger command finishes typing before its press animation begins.
s=film['shots'][11];s['interactions'][0]['at']=1.05
motion(s,'ledger-command')['keys']=[{'t':0,'o':0},{'t':.93/5,'o':0},{'t':1.05/5,'o':1,'s':1},{'t':1.17/5,'s':.994},{'t':1.31/5,'s':1},{'t':1.70/5,'o':1},{'t':1.83/5,'o':0}]
motion(s,'palette')['keys'][-2]['t']=1.75/5;motion(s,'palette')['keys'][-1]['t']=2.15/5
# Add concise, independently animated technical cards beside the diagnostics.
def card(index,kind):
 s=film['shots'][index]
 lines=['HOW TABPFN CHECKS A CELL','Hold this row out of the fit.','Use other columns as inputs.']
 if kind=='numeric':lines+=['Regressor → value distribution','Median + 10th–90th percentiles','Flag values in unlikely tails.']
 else:lines+=['Classifier → P(category | row)','Score the recorded category.','Suggest the most likely class.']
 text=[];defs=[]
 for n,line in enumerate(lines):
  id=f'technical-text-{n}'; mask=f'technical-mask-{n}'; y=34+n*38;size=15 if n==0 else 19
  defs.append(f'<clipPath id="technical-clip-{n}" clipPathUnits="userSpaceOnUse"><rect id="{mask}" x="22" y="{y-24}" width="366" height="32"/></clipPath>')
  text.append(f'<text id="{id}" x="22" y="{y}" font-family="Inter,Arial,sans-serif" font-size="{size}" font-weight="{700 if n in (0,3) else 400}" fill="{"#00867e" if n in (0,3) else "#263b45"}" clip-path="url(#technical-clip-{n})">{html.escape(line)}</text>')
  delay=2.4 if index==6 else 0
  s['typing'].append({'id':id,'mask':mask,'text':line,'start':s['start']+delay+.8+n*.12,'duration':max(.22,len(line)/42),'x':22,'width':366,'font':size,'mono':False,'bold':n in (0,3)})
 top=286 if index==6 else 365
 markup=f'<g id="technical-card"><g transform="translate(345 {top})"><rect width="410" height="252" rx="14" fill="#fff" stroke="#dce5e8"/><rect x="0" y="0" width="4" height="252" rx="2" fill="#00867e"/>'+''.join(text)+'</g></g>'
 layers[index]=layers[index].replace('</defs>',''.join(defs)+'</defs>').replace('</svg>',markup+'</svg>')
 s['motions'].append({'id':'technical-card','origin':[960,540],'keys':[{'t':0,'o':0,'y':12},{'t':(.6+(2.4 if index==6 else 0))/s['duration'],'o':0,'y':12},{'t':(1+(2.4 if index==6 else 0))/s['duration'],'o':1,'y':0},{'t':(s['duration']-.4)/s['duration'],'o':1},{'t':1,'o':0,'y':8}]})
for i,k in [(6,'numeric'),(7,'numeric'),(10,'category')]:card(i,k)
# The classifier explanation belongs to the suggestion and exits with that popup.
s=film['shots'][10];hover=motion(s,'hover')
technical=motion(s,'technical-card')
technical['keys']=technical['keys'][:3]+[
 {'t':hover['keys'][-2]['t'],'o':1,'y':0},
 {'t':hover['keys'][-1]['t'],'o':0,'y':8}]
# Piecewise time maps replace arbitrary idle time with a consistent 1.8-second reading hold.
# A source range plays at normal speed; a repeated source point is a reading hold.
P=1.8
plans=[
 [(0,.75),(.75,.75,.6,False),(.75,3.66),(3.66,3.66,.5,False),(3.66,4)],
 [(0,.65),(.65,.65,16/30,False),(2.6,3)],
 [(0,2.02),(2.02,2.02,P),(3.6,4)],
 [(0,1.15),(1.15,1.15,P),(1.15,2.72)],
 [(0,2.02),(2.6,3)],
 [(0,.93),(.93,.93,P),(5.5,6)],
 [(0,6.68),(6.68,6.68,P),(8.7,9.4)],
 [(0,2.48),(2.48,2.48,P),(5.5,6)],
 [(0,1.0),(1.0,1.0,P),(2.38,4.26),(4.26,4.26,P),(4.8,5)],
 [(0,2.35),(2.35,2.35,P)],
 [(0,2.48),(2.48,2.48,P),(3.5,4.5),(4.5,4.5,P),(4.5,6.57),(6.57,6.57,P)],
 [(0,.95),(.95,.95,P),(.95,4.15),(4.15,4.15,P)],
 [(0,1.56),(1.56,1.56,P),(2.88,5.5),(5.5,5.5,P),(5.5,7.75),(7.75,7.75,P),(9.2,9.6)],
 [(0,1.2),(1.2,1.2,P),(4.6,5)]
]
# Scene 11's final frame is a review-note modal; let it exit smoothly before opening the ledger.
s=film['shots'][10];motion(s,'note')['keys'].pop();motion(s,'note')['keys'] += [{'t':.95,'o':1,'y':0},{'t':1,'o':0,'y':-12}]
plans[10].append((7.6,8))
for index in (7,10):
    shot=film['shots'][index]
    end=max(t['start']+t['duration']-shot['start'] for t in shot['typing'] if 0<=t['start']-shot['start']<2.5)
    read=math.ceil(end*30)/30
    plans[index][0]=(0,read);plans[index][1]=(read,read,P)
timeline=[];start=0
for i,(s,plan) in enumerate(zip(film['shots'],plans)):
 segments=[];frame=18 if i else 0
 for entry in plan:
  a,b=entry[:2];length=round((entry[2] if len(entry)>=3 else b-a)*30)
  segments.append({'frame':frame,'frames':length,'sourceFrom':a,'sourceTo':b,'readingPause':entry[3] if len(entry)>3 else len(entry)==3})
  frame+=length
 timeline.append({'index':i,'startFrame':start,'contentFrames':frame,'durationFrames':frame+(18 if i<13 else 0),'segments':segments})
 start+=frame
payload={'fps':30,'totalFrames':start,'runtime':start/30,'readingPauseSeconds':P,'transitionFrames':18,'loadingStartSeconds':1,'loadingDurationSeconds':1.1,'scenes':timeline}
(root/'src/tabl-int/assets.ts').write_text('export const film = '+json.dumps(film,separators=(',',':'))+';\nexport const layers = '+json.dumps(layers,separators=(',',':'))+';\n')
(root/'src/tabl-int/timing.ts').write_text('export const timing = '+json.dumps(payload,separators=(',',':'))+';\n')
(export/'timing.json').write_text(json.dumps(payload,indent=2));(export/'source-shots.json').write_text(json.dumps(film,indent=2))
for i,layer in enumerate(layers):(export/f'shot-{i+1:02d}-layers.svg').write_text(layer)
print('New runtime:',start/30,'seconds;',start,'frames')
for s,t in zip(film['shots'],timeline): print(s['id'],t['startFrame']/30,t['contentFrames']/30)

# Keep independently editable JSX sequence nodes in sync with the retimed schedule.
import re
component=root/'src/TabLintDemo.tsx';source=component.read_text();counter=iter(timeline)
def sequence(match):
    item=next(counter)
    return f'<Sequence from={{{item["startFrame"]}}} durationInFrames={{{item["durationFrames"]}}}'
component.write_text(re.sub(r'<Sequence from=\{\d+\} durationInFrames=\{\d+\}',sequence,source))

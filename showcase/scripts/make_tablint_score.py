"""Original 112 BPM electronic groove: warm keys, bass, light drums and stereo synths.
Requires numpy; all instruments synthesized here, with no sampled recordings.
"""
from pathlib import Path
import wave
import numpy as np
import json, math
revision=Path(__file__).resolve().parents[1]/'review'
timeline=json.loads((revision/'timing.json').read_text())
board=json.loads((revision/'source-shots.json').read_text())
def movie_time(index,source):
    scene=timeline['scenes'][index]
    for seg in scene['segments']:
        if seg['sourceFrom'] <= source < seg['sourceTo']:
            return (scene['startFrame']+seg['frame']+(source-seg['sourceFrom'])/(seg['sourceTo']-seg['sourceFrom'])*seg['frames'])/30
    return (scene['startFrame']+scene['contentFrames'])/30
sr=44100
duration=timeline['runtime']
out=np.zeros((round(sr*duration),2),dtype=np.float64)
rng=np.random.default_rng(42)
def add(at,signal,pan=0):
    start=int(at*sr)
    end=min(start+len(signal),len(out))
    if end<=start:return
    out[start:end,0]+=signal[:end-start]*np.sqrt((1-pan)/2)
    out[start:end,1]+=signal[:end-start]*np.sqrt((1+pan)/2)
def note(midi,at,length,amp,pan=0,pluck=False):
    t=np.arange(int(length*sr))/sr
    hz=440*2**((midi-69)/12)
    env=(1-np.exp(-t*(75 if pluck else 4)))*np.exp(-t*(2.3 if pluck else .15))
    env*=np.minimum(1,(length-t)/(0.1 if pluck else 1.3))
    signal=(np.sin(2*np.pi*hz*t)+.16*np.sin(2*np.pi*hz*2*t)+.06*np.sin(2*np.pi*hz*3*t))*env*amp
    add(at,signal,pan)
    if pluck:
        add(at+.24,signal*.18,-pan)
        add(at+.48,signal*.08,pan)
beat=60/112;bar_seconds=4*beat
chords=[[50,57,61,64,66],[47,54,57,61,66],[43,50,54,57,62],[45,52,57,59,64]]
def keys(midi,at,amp,pan=0):
    length=1.5;t=np.arange(int(length*sr))/sr;hz=440*2**((midi-69)/12)
    # Rounded electric keys with a soft tine transient and slight chorus.
    env=(1-np.exp(-t*100))*np.exp(-t*2.3)*np.minimum(1,(length-t)/.12)
    body=np.sin(2*np.pi*hz*t)+.28*np.sin(2*np.pi*hz*2*t)*np.exp(-t*5)+.12*np.sin(2*np.pi*(hz*1.002)*t)
    signal=body*env*amp;add(at,signal,pan);add(at+beat*.75,signal*.14,-pan)
def bass(midi,at,length,amp):
    t=np.arange(int(length*sr))/sr;hz=440*2**((midi-69)/12)
    env=np.minimum(t/.012,1)*np.minimum((length-t)/.06,1)
    add(at,(np.sin(2*np.pi*hz*t)+.18*np.sin(2*np.pi*hz*2*t))*env*amp)
def drum(at,kind,amp,pan=0):
    length={'kick':.28,'clap':.14,'hat':.055}[kind];t=np.arange(int(length*sr))/sr
    if kind=='kick':
        signal=np.sin(2*np.pi*(48*t+3.5*(1-np.exp(-t*32))))*np.exp(-t*18)
    else:
        noise=rng.standard_normal(len(t));high=noise-np.convolve(noise,np.ones(9)/9,mode='same')
        if kind=='hat':signal=high*np.exp(-t*95)*np.minimum(t/.001,1)
        else:
            envelope=sum(np.exp(-(t-offset)*65)*(t>=offset) for offset in [0,.009,.019])
            signal=high*envelope*.3
    add(at,signal*amp,pan)
for bar in range(math.ceil(duration/bar_seconds)):
    at=bar*bar_seconds;chord=chords[(bar//2)%4]
    # Change voicing every two bars and introduce the groove after the opening.
    for i,n in enumerate(chord[1:]):note(n,at,bar_seconds+.6,.012,[-.5,-.2,.2,.5][i])
    for offset in [0,1.5,2.75]:
        for i,n in enumerate(chord[1:]):keys(n+12,at+offset*beat+i*.012,.026,(-.25 if i%2 else .25))
    for offset,n,length in [(0,chord[0]-12,.7),(1.5,chord[0]-12,.35),(2,chord[0]-12,.6),(3.5,chord[0],.3)]:
        bass(n,at+offset*beat,length*beat,.05)
    if bar>=2:
        for offset in [0,2,2.75]:drum(at+offset*beat,'kick',.07)
        for offset in [1,3]:drum(at+offset*beat,'clap',.035)
        for i in range(8):drum(at+i*beat/2,'hat',.014 if i%2 else .01,(-.3 if i%2 else .3))
    # Sparse melodic figures leave space for the explanations.
    if bar%4 in [1,3]:
        for i,step in enumerate([3,2,4,2]):
            note(chord[step]+12,at+(i*.75+.25)*beat,.7,.028,(-.45 if i%2 else .45),True)
for at in [movie_time(0,1.95)]+[scene['startFrame']/30+.3 for scene in timeline['scenes'][1:]]:
    t=np.arange(int(.4*sr))/sr
    raw=rng.standard_normal(len(t))
    smooth=np.convolve(raw,np.ones(17)/17,mode='same')
    add(at-.16,smooth*np.sin(np.pi*t/.4)**2*.018)
# UI confirmation tones, matched to the approved interactions.
for index,shot in enumerate(board['shots']):
    for event in shot.get('interactions',[]):
        if event['mode']=='press':
            at=movie_time(index,event['at']+.28)
            note(81,at,.14,.045,-.08,True)
            note(76,at+.06,.18,.023,.08,True)
for at in [movie_time(8,3.26),movie_time(12,6.51)]:
    note(76,at,.6,.065,-.15,True)
    note(79,at+.12,.75,.05,.15,True)
fade=np.minimum(np.arange(len(out))/sr/1.2,1)*np.minimum((len(out)-np.arange(len(out)))/sr/2.0,1)
out*=fade[:,None]
out=np.tanh(out*1.4)
path=Path(__file__).resolve().parents[1]/'public'/'tabl-int-score.wav'
with wave.open(str(path),'wb') as w:
    w.setnchannels(2);w.setsampwidth(2);w.setframerate(sr)
    w.writeframes((out*32767).astype('<i2').tobytes())
print('Wrote',path,'peak',float(np.max(np.abs(out))))

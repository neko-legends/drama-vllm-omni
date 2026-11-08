import json, subprocess, time, wave, statistics, urllib.request

BASE="http://127.0.0.1:8091"
model=json.load(urllib.request.urlopen(BASE+"/v1/models"))["data"][0]["id"]
TEXTS={
 "10w":"The quick brown fox jumps over the lazy dog.",
 "50w":("Our agents read long prompts every day. Prefill is the part that decides whether the answer feels instant, "
        "so we measured it carefully, on the same four machines, with the same harness, twice per size, and wrote "
        "down every number we could reproduce."),
 "long":("She opened the door and the room was exactly as she left it. The lamp still leaned, the chair still faced "
         "the window, and the rain had not stopped. What surprised her was not the silence but how familiar it felt, "
         "as though the house had been waiting patiently for someone to come back and notice it. She set the bag down, "
         "listened to the floorboards settle, and decided that tomorrow she would finally fix the hinge."),
}
out=[]
for label,text in TEXTS.items():
    rows=[]
    for i in range(3):
        body=json.dumps({"model":model,"input":text,"response_format":"wav"}).encode()
        t0=time.time()
        req=urllib.request.Request(BASE+"/v1/audio/speech",data=body,headers={"Content-Type":"application/json"})
        data=urllib.request.urlopen(req,timeout=600).read()
        wall=time.time()-t0
        open(f"/tmp/b-{label}-{i}.wav","wb").write(data)
        w=wave.open(f"/tmp/b-{label}-{i}.wav")
        dur=w.getnframes()/w.getframerate()
        rows.append((wall,dur))
    walls=[r[0] for r in rows]; durs=[r[1] for r in rows]; rtfs=[d/w for w,d in rows]
    out.append({"label":label,"words":len(text.split()),"wall_med":round(statistics.median(walls),3),
                "wall_min":round(min(walls),3),"audio_med":round(statistics.median(durs),3),
                "rtf_med":round(statistics.median(rtfs),2),"rtf_min":round(max(rtfs),2),"runs":rows})
    print(f"{label:5s} words={len(text.split()):3d} wall={out[-1]['wall_med']:6.2f}s audio={out[-1]['audio_med']:6.2f}s RTF={out[-1]['rtf_med']:4.2f}x")
json.dump(out,open("/home/jun/drama-omni/bench-r9700.json","w"),indent=1)

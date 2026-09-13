import sys, pathlib, os, io, contextlib, json, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))   # official_eval.py 就在同一層
import official_eval as OE
from pycocotools.cocoeval import COCOeval
def load(tag):
    z=np.load(os.path.join(OE.RAWD,f"{tag}.npz"), allow_pickle=True)
    return {v:{'boxes':z[f'b{v}'],'scores':z[f's{v}'],'ids':z[f'i{v}']} for v in OE.VIDS}
rows=[]
for tag in ("s1_i640","v1p_base","v3_s2"):
    raws=load(tag)
    for v in OE.VIDS:
        G=OE.gt(v); cm=OE.cat_map(v)
        dets=OE.decode(raws[v], mode='global_topk', conf=0.001, max_det=300)
        res=[{"image_id":i,"category_id":cm[c],"score":s,"bbox":b} for i,c,s,b in dets]
        with contextlib.redirect_stdout(io.StringIO()):
            D=G.loadRes(res)
        present=sorted({a['category_id'] for a in G.loadAnns(G.getAnnIds())})
        for cid in present:
            with contextlib.redirect_stdout(io.StringIO()):
                E=COCOeval(G,D,'bbox'); E.params.maxDets=[5,10,100]; E.params.catIds=[cid]
                E.evaluate(); E.accumulate(); E.summarize()
            s=E.stats
            # AP at IoU 0.5..0.95 個別值
            prec=E.eval['precision']  # [T,R,K,A,M]
            ap_t=[float(np.mean(prec[t,:,0,0,2][prec[t,:,0,0,2]>-1])) if (prec[t,:,0,0,2]>-1).any() else 0.0 for t in range(10)]
            name=G.loadCats(cid)[0]['name']; ngt=len(G.getAnnIds(catIds=[cid]))
            rows.append((tag,v,name,ngt,s[0],s[1],s[2],ap_t))
print(f"{'model':9s} v {'class':26s} {'nGT':>5s} {'AP':>6s} {'AP50':>6s} {'AP75':>6s}  AP@.5 .55 .6 .65 .7 .75 .8 .85 .9 .95")
for tag,v,name,ngt,ap,ap50,ap75,ap_t in rows:
    print(f"{tag:9s} {v} {name:26s} {ngt:5d} {ap:6.3f} {ap50:6.3f} {ap75:6.3f}  "+" ".join(f"{x:.2f}" for x in ap_t))
json.dump([dict(tag=t,vid=v,cls=n,ngt=g,ap=a,ap50=b,ap75=c,ap_t=d) for t,v,n,g,a,b,c,d in rows], open(os.environ.get("SURGVU_OUT", ".") + "/ap_by_iou.json", "w"), indent=1)

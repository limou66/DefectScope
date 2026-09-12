"""Reproduce the small synthetic multi-seed suite. No test-driven parameter search."""
import argparse
import json
from pathlib import Path
import numpy as np
from defectscope.data import generate, CATEGORIES
from defectscope.experiment import evaluate

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seeds",type=int,nargs="+",default=[7,19,42])
    p.add_argument("--extractor",choices=["texture","resnet18"],default="texture")
    p.add_argument("--categories",nargs="+",choices=CATEGORIES,default=list(CATEGORIES))
    p.add_argument("--out",default="runs/suite")
    args=p.parse_args()
    rows=[]
    for seed in args.seeds:
        data=Path("data")/f"synthetic-s{seed}"
        if not data.exists():
            generate(data,seed)
        for category in args.categories:
            for illumination in (False,True):
                run=Path(args.out)/f"{category}-{args.extractor}-s{seed}-illum{int(illumination)}"
                if (run/"metrics.json").exists():
                    meta=json.loads((run/"metrics.json").read_text(encoding="utf-8"))
                else:
                    meta=evaluate(data/category,run,seed,args.extractor,illumination=illumination)
                rows.append(meta)
                print(category,seed,illumination,meta["results"]["gated"]["image_auroc"],flush=True)
    summary=[]
    for category in args.categories:
        for illumination in (False,True):
            group=[r for r in rows if r["category"]==category and r["config"]["illumination"]==illumination]
            for mode in ("global","spatial","fusion","gated"):
                row={"category":category,"illumination":illumination,"mode":mode,"seeds":args.seeds}
                for key in ("image_auroc","pixel_auroc","f1","fpr","shift_fpr"):
                    values=[r["results"][mode][key] for r in group]
                    row[key]={"mean":float(np.mean(values)),"std":float(np.std(values))}
                summary.append(row)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    lines=["# Synthetic multi-seed results","","Population SD across seeds; synthetic results are not industrial benchmark claims.","",
           "| Category | Illumination | Mode | Image AUROC mean ± SD | F1 mean ± SD | Normal FPR | Shift FPR |",
           "|---|---|---|---:|---:|---:|---:|"]
    for r in summary:
        val=lambda k:f"{r[k]['mean']:.3f} ± {r[k]['std']:.3f}"
        lines.append(f"| {r['category']} | {r['illumination']} | {r['mode']} | {val('image_auroc')} | {val('f1')} | {val('fpr')} | {val('shift_fpr')} |")
    (out/"SUMMARY.md").write_text("\n".join(lines),encoding="utf-8")

if __name__=="__main__":
    main()

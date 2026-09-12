import argparse
import json
import os
from pathlib import Path
from .data import CATEGORIES, generate
from .experiment import evaluate
from .model import Detector, MODES
from .reporting import save_prediction

def main():
    parser = argparse.ArgumentParser(description="DefectScope: normal-only visual anomaly inspection")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="create reproducible synthetic images")
    gen.add_argument("--out", required=True); gen.add_argument("--seed", type=int, default=7)
    gen.add_argument("--categories", nargs="+", choices=CATEGORIES, default=list(CATEGORIES))
    run = sub.add_parser("run", help="fit, calibrate and evaluate all four modes")
    run.add_argument("--data", required=True); run.add_argument("--out", required=True)
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--extractor", choices=["texture","resnet18"], default="texture")
    run.add_argument("--size", type=int, default=96); run.add_argument("--patch", type=int, default=6)
    run.add_argument("--memory", type=int, default=128); run.add_argument("--alpha", type=float, default=.05)
    run.add_argument("--illumination", action="store_true", help="robustly align low-frequency lighting")
    pred=sub.add_parser("predict", help="inspect one image")
    pred.add_argument("--model",required=True); pred.add_argument("--image",required=True)
    pred.add_argument("--out",required=True); pred.add_argument("--mode",choices=MODES,default="gated")
    serve=sub.add_parser("serve",help="start local interactive inspection console")
    serve.add_argument("--model",required=True);serve.add_argument("--samples",required=True)
    serve.add_argument("--port",type=int,default=8765)
    args=parser.parse_args()
    os.environ.setdefault("TORCH_HOME", str(Path.cwd()/".cache"/"torch"))
    if args.command=="generate":
        print(generate(args.out,args.seed,args.categories))
    elif args.command=="run":
        result=evaluate(args.data,args.out,args.seed,args.extractor,args.size,args.patch,args.memory,args.alpha,args.illumination)
        print(json.dumps(result["results"],indent=2))
    elif args.command=="predict":
        model=Detector.load(args.model)
        result=model.predict(args.image,args.mode)
        save_prediction(args.image,result,args.out,model.config["size"])
        print(json.dumps({k:v for k,v in result.items() if k not in ("heatmap","global_map","spatial_map")},indent=2))
    elif args.command=="serve":
        from .server import serve
        serve(args.model,args.samples,args.port)

if __name__=="__main__":
    main()

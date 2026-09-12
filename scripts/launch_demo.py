import importlib.util
import urllib.request
import webbrowser
from pathlib import Path
from threading import Timer
from defectscope.data import generate
from defectscope.experiment import evaluate
from defectscope.server import serve

def main():
    url="http://127.0.0.1:8765"
    try:
        with urllib.request.urlopen(url+"/api/info",timeout=2) as response:
            if response.status==200:
                print("A local inspection service is already running:",url)
                webbrowser.open(url)
                return
    except OSError:
        pass
    model=Path("runs/bottle-resnet-s7/model.npz")
    samples=Path("data/mvtec/bottle/test")
    if not model.exists() or not samples.exists() or importlib.util.find_spec("torch") is None:
        data=Path("data/demo")
        model=Path("runs/demo/model.npz")
        samples=data/"board/test"
        if not (data/"dataset.json").exists():
            generate(data,7,["board"])
        if not model.exists():
            evaluate(data/"board","runs/demo",illumination=True)
    print("Opening",url,"; press Ctrl+C here to stop.")
    timer=Timer(2,lambda:webbrowser.open(url));timer.daemon=True;timer.start()
    serve(model,samples)

if __name__=="__main__":
    main()

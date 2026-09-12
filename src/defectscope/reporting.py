import base64
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .features import load_image

def heat_image(array, threshold):
    # Fixed normal-only reference, never normalize each image to its own max.
    t = np.clip(np.asarray(array)/(max(float(threshold), .001)*2), 0, 1)
    rgb = np.stack([np.clip(2*t, 0, 1), np.clip(2-abs(4*t-2), 0, 1), np.clip(1-2*t, 0, 1)], -1)
    return Image.fromarray(np.uint8(rgb*255))

def png_data(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")

def views(path, result, size=96):
    original = Image.fromarray(np.uint8(load_image(path, size)*255))
    heat = heat_image(result["heatmap"], result["pixel_threshold"]).resize((size, size), Image.Resampling.BILINEAR)
    overlay = Image.blend(original, heat, .45)
    mask = Image.fromarray(np.uint8(result["heatmap"] > result["pixel_threshold"])*255).resize((size, size), Image.Resampling.NEAREST)
    normal = Path(result["evidence"]["normal_image"])
    reference = Image.open(normal).convert("RGB").resize((size, size)) if normal.is_file() else Image.new("RGB", (size, size), "#334155")
    draw = ImageDraw.Draw(reference)
    y, x = result["evidence"]["normal_patch_yx"]
    p = size//result["heatmap"].shape[0]
    draw.rectangle((x*p, y*p, (x+1)*p-1, (y+1)*p-1), outline="#00ffcc", width=2)
    return {"original": original, "heatmap": heat, "overlay": overlay, "mask": mask, "reference": reference}

def save_prediction(path, result, output, size=96):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, im in views(path, result, size).items():
        im.resize((384, 384), Image.Resampling.NEAREST).save(output/(name+".png"))
    public = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
    (output/"prediction.json").write_text(json.dumps(public, indent=2, ensure_ascii=False), encoding="utf-8")
    np.save(output/"heatmap.npy", result["heatmap"])

def gallery(items, output, size=96):
    width, rowheight = 5*192, 220
    canvas = Image.new("RGB", (width, rowheight*(len(items)+1)), "#101c2b")
    draw = ImageDraw.Draw(canvas)
    headers = ["INPUT", "GROUND TRUTH", "HEATMAP", "OVERLAY", "NORMAL EVIDENCE"]
    for col, label in enumerate(headers):
        draw.text((col*192+12, 15), label, fill="white")
    for row, (path, maskpath, result) in enumerate(items):
        v = views(path, result, size)
        truth = Image.open(maskpath).convert("RGB") if maskpath else Image.new("RGB", (size, size))
        ims = [v["original"], truth, v["heatmap"], v["overlay"], v["reference"]]
        y = 50 + row*rowheight
        for col, im in enumerate(ims):
            canvas.paste(im.resize((180, 180)), (col*192+6, y))
        draw.text((10, y+184), f"{path.parent.name} | score={result['score']:.3f} | p={result['p_value']:.3f}", fill="white")
    canvas.crop((0, 0, width, 50+len(items)*rowheight)).save(output)

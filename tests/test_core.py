import io
import json
import base64
import threading
import urllib.request
import urllib.error
import numpy as np
import pytest
from defectscope.data import generate, splits, provenance
from defectscope.features import TextureFeatures, load_image
from defectscope.metrics import auroc, conformal_threshold, normal_p_value
from defectscope.model import Detector, coreset, nearest
from defectscope.server import create_server

@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root=tmp_path_factory.mktemp("dataset")/"synthetic"
    generate(root,7,["board"])
    return root/"board"

@pytest.fixture(scope="module")
def trained(dataset,tmp_path_factory):
    train,tune,cal,_,_=splits(dataset)
    model=Detector(memory=32).fit(train,tune,cal)
    path=tmp_path_factory.mktemp("model")/"model.npz"
    model.save(path)
    return model,path

def test_auroc_ties_and_missing_class():
    assert auroc([0,1],[.5,.5])==.5
    assert auroc([0,1,0,1],[0,3,1,2])==1
    assert auroc([0,1],[1,0])==0
    assert auroc([0,0],[1,2]) is None

def test_conformal_rank_boundary():
    assert conformal_threshold(range(39),.05)==37
    assert normal_p_value(np.arange(39),38.5)==.025
    assert normal_p_value(np.arange(39),37.5)==.05
    assert normal_p_value(np.arange(39),37)==.075
    assert np.isinf(conformal_threshold([1,2],.05))

def test_features(dataset):
    f=TextureFeatures()(load_image(dataset/"train/good/000.png"))
    assert f.shape==(16,16,14)
    assert np.isfinite(f).all()
    with pytest.raises(ValueError):
        TextureFeatures(size=95)

def test_nearest_matches_brute_force():
    rng=np.random.default_rng(2)
    q,b=rng.normal(size=(50,7)),rng.normal(size=(25,7))
    distances,indices=nearest(q,b,chunk=11)
    expected=np.sqrt(((q[:,None]-b[None])**2).mean(2))
    np.testing.assert_allclose(distances,expected.min(1),atol=1e-7)
    np.testing.assert_array_equal(indices,expected.argmin(1))

def test_coreset_unique_deterministic():
    x=np.zeros((20,4))
    assert len(set(coreset(x,15,1)))==15
    np.testing.assert_array_equal(coreset(x,15,1),coreset(x,15,1))

def test_no_split_leakage(dataset):
    train,tune,cal,tests,shift=splits(dataset)
    provenance({"train":train,"tune":tune,"cal":cal,"test":[x["path"] for x in tests],"shift":shift},dataset)
    with pytest.raises(ValueError,match="duplicate"):
        provenance({"a":[train[0]],"b":[train[0]]},dataset)
    with pytest.raises(ValueError,match="disjoint"):
        Detector().fit(train,train[:2],cal)

def test_save_load_prediction_parity(dataset,trained):
    model,path=trained
    sample=dataset/"test/hole/000.png"
    before=model.predict(sample)
    after=Detector.load(path).predict(sample)
    assert before["score"]==after["score"]
    np.testing.assert_array_equal(before["heatmap"],after["heatmap"])
    assert after["is_anomaly"]
    assert after["evidence"]["normal_image"].endswith(".png")

def test_calibration_not_used_for_memory(dataset):
    train,tune,cal,_,_=splits(dataset)
    a=Detector(memory=16).fit(train,tune,cal[:20])
    b=Detector(memory=16).fit(train,tune,cal[19:])
    np.testing.assert_array_equal(a.bank,b.bank)
    np.testing.assert_array_equal(a.component_scale,b.component_scale)
    np.testing.assert_array_equal(a.gate,b.gate)

def test_server_prediction_upload_and_invalid_input(dataset,trained,monkeypatch):
    _,path=trained
    import os
    monkeypatch.chdir(dataset.parent)
    server=create_server(path,os.path.relpath(dataset/"test"),0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    url=f"http://127.0.0.1:{server.server_port}"
    def post(payload):
        request=urllib.request.Request(url+"/api/predict",json.dumps(payload).encode(),{"Content-Type":"application/json"})
        return json.loads(urllib.request.urlopen(request).read())
    try:
        info=json.loads(urllib.request.urlopen(url+"/api/info").read())
        assert len(info["samples"])==72
        result=post({"sample_id":0,"mode":"gated"})
        assert result["images"]["overlay"].startswith("data:image/png;base64,")
        raw=(dataset/"test/hole/000.png").read_bytes()
        assert post({"image_base64":base64.b64encode(raw).decode(),"mode":"spatial"})["is_anomaly"]
        with pytest.raises(urllib.error.HTTPError) as exc:
            post({"sample_id":-1})
        assert exc.value.code==400
        with pytest.raises(urllib.error.HTTPError) as exc:
            post({"sample_id":0,"mode":"invalid"})
        assert exc.value.code==400
    finally:
        server.shutdown();server.server_close();thread.join()


def test_illumination_removes_plane_preserves_local_defect():
    from defectscope.illumination import align_illumination
    y,x=np.mgrid[0:96,0:96]
    reference=np.ones((96,96,3),dtype=np.float32)*.5
    shifted=reference+.09+.08*(x/96)[...,None]
    shifted[40:46,40:46]+=.15
    corrected=align_illumination(shifted,reference)
    assert np.abs(corrected[:30]-reference[:30]).mean()<.005
    assert (corrected[40:46,40:46]-reference[40:46,40:46]).mean()>.12

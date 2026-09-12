"""Optional integration test; CI core jobs do not download pretrained weights."""
import os
from pathlib import Path
import numpy as np
import pytest

@pytest.mark.skipif(os.environ.get("DEFECTSCOPE_TEST_DEEP")!="1",reason="set DEFECTSCOPE_TEST_DEEP=1 with official weights cached")
def test_deep_model_roundtrip_on_real_sample():
    from defectscope.model import Detector
    from defectscope.features import load_image
    path=Path("runs/bottle-resnet-s7/model.npz")
    if not path.exists():
        pytest.skip("real-data run not present")
    a=Detector.load(path)
    image=Path("data/mvtec/bottle/test/broken_large/000.png")
    feature=a.extractor(load_image(image,a.config["size"]))
    assert feature.shape==(16,16,192)
    assert np.isfinite(feature).all()
    prediction=a.predict(image)
    assert prediction["is_anomaly"]
    assert prediction["p_value"]==pytest.approx(1/53)
    original=np.load(path,allow_pickle=False)
    np.testing.assert_array_equal(a.reference,original["reference"])

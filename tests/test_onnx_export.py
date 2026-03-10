from pathlib import Path

import numpy as np
import onnx
import pandas as pd

from deployment.onnx_export import _patch_onnx_helper_compat, _patch_onnx_mapping_compat, export_pipeline_to_onnx
from modeling.predictive import SimpleLogisticModel


def test_patch_onnx_mapping_compat_sets_mapping_alias():
    had_mapping = hasattr(onnx, "mapping")
    original_mapping = getattr(onnx, "mapping", None)
    try:
        if had_mapping:
            delattr(onnx, "mapping")
        _patch_onnx_mapping_compat()
        assert hasattr(onnx, "mapping")
        assert onnx.mapping is onnx._mapping
    finally:
        if had_mapping:
            onnx.mapping = original_mapping
        elif hasattr(onnx, "mapping"):
            delattr(onnx, "mapping")


def test_patch_onnx_helper_compat_adds_splitter():
    from onnx import helper

    had_splitter = hasattr(helper, "split_complex_to_pairs")
    original_splitter = getattr(helper, "split_complex_to_pairs", None)
    try:
        if had_splitter:
            delattr(helper, "split_complex_to_pairs")
        _patch_onnx_helper_compat()
        assert helper.split_complex_to_pairs([1 + 2j, 3 + 4j]) == [1.0, 2.0, 3.0, 4.0]
    finally:
        if had_splitter:
            helper.split_complex_to_pairs = original_splitter
        elif hasattr(helper, "split_complex_to_pairs"):
            delattr(helper, "split_complex_to_pairs")


def test_export_pipeline_to_onnx_supports_simple_logistic_model(tmp_path: Path):
    X = pd.DataFrame({"f1": [0.1, 0.2, 0.8, 0.9], "f2": [1.0, 0.9, 0.2, 0.1]})
    y = pd.Series([0, 0, 1, 1])
    model = SimpleLogisticModel(lr=0.1, epochs=50).fit(X, y)

    out = tmp_path / "simple_model.onnx"
    result = export_pipeline_to_onnx(model, out, n_features=2)

    assert result["success"] is True
    assert out.exists()

    import onnxruntime as ort

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    ort_probs = session.run(None, {input_name: X.values.astype(np.float32)})[0]
    py_probs = model.predict_proba(X)
    assert np.allclose(ort_probs, py_probs, atol=1e-5)

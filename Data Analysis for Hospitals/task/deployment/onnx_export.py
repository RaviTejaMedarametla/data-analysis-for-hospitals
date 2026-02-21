from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def export_pipeline_to_onnx(pipeline, output_path: Path, n_features: int) -> bool:
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        onx = convert_sklearn(pipeline, initial_types=initial_type)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(onx.SerializeToString())
        return True
    except Exception:
        return False


def onnx_parity_check(model, onnx_path: Path, X: pd.DataFrame, atol: float = 1e-4) -> dict[str, float]:
    if not onnx_path.exists():
        return {"onnxruntime_available": 0.0, "parity_ok": 0.0, "max_abs_error": float("inf")}
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        arr = X.values.astype(np.float32)
        onnx_pred = sess.run(None, {input_name: arr})[1][:, 1]
        sk_pred = model.predict_proba(X)[:, 1]
        err = float(np.max(np.abs(onnx_pred - sk_pred)))
        return {"onnxruntime_available": 1.0, "parity_ok": float(err <= atol), "max_abs_error": err}
    except Exception:
        return {"onnxruntime_available": 0.0, "parity_ok": 0.0, "max_abs_error": float("inf")}

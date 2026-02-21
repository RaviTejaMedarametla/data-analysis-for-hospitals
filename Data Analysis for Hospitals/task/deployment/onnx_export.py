from __future__ import annotations

from pathlib import Path


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

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def export_pipeline_to_onnx(pipeline: Any, output_path: Path, n_features: int) -> dict[str, Any]:
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        onx = convert_sklearn(pipeline, initial_types=initial_type)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(onx.SerializeToString())
        return {"success": True, "path": str(output_path), "error": None}
    except ImportError as exc:
        logger.error("ONNX export failed due to missing dependency: %s", exc)
        return {"success": False, "path": str(output_path), "error": f"import_error: {exc}"}
    except ValueError as exc:
        logger.error("ONNX export failed due to invalid model signature: %s", exc)
        return {"success": False, "path": str(output_path), "error": f"value_error: {exc}"}
    except Exception as exc:
        logger.exception("Unexpected ONNX export failure")
        return {"success": False, "path": str(output_path), "error": f"unexpected_error: {exc}"}

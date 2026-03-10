from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _patch_onnx_mapping_compat() -> None:
    """Bridge ONNX 1.20+ API change expected by older skl2onnx versions."""
    try:
        import onnx
    except ImportError:
        return

    if not hasattr(onnx, "mapping") and hasattr(onnx, "_mapping"):
        onnx.mapping = onnx._mapping


def _patch_onnx_helper_compat() -> None:
    """Backfill helpers removed from recent ONNX releases."""
    try:
        from onnx import helper
    except ImportError:
        return

    if not hasattr(helper, "split_complex_to_pairs"):

        def split_complex_to_pairs(values):
            pairs = []
            for value in values:
                pairs.extend((float(value.real), float(value.imag)))
            return pairs

        helper.split_complex_to_pairs = split_complex_to_pairs


def _export_simple_logistic_to_onnx(model: Any, output_path: Path, n_features: int) -> None:
    """Export SimpleLogisticModel-compatible weights to ONNX directly."""
    import onnx
    from onnx import TensorProto, helper

    weights = np.asarray(model.weights, dtype=np.float32)
    if weights.shape[0] != n_features + 1:
        raise ValueError("simple_logistic_weights_mismatch")

    coef = weights[1:].reshape(n_features, 1)
    intercept = np.array([weights[0]], dtype=np.float32)
    one = np.array([1.0], dtype=np.float32)

    graph = helper.make_graph(
        nodes=[
            helper.make_node("MatMul", ["float_input", "coef"], ["linear"]),
            helper.make_node("Add", ["linear", "intercept"], ["logits"]),
            helper.make_node("Sigmoid", ["logits"], ["p1"]),
            helper.make_node("Sub", ["one", "p1"], ["p0"]),
            helper.make_node("Concat", ["p0", "p1"], ["probabilities"], axis=1),
        ],
        name="simple_logistic_model",
        inputs=[helper.make_tensor_value_info("float_input", TensorProto.FLOAT, [None, n_features])],
        outputs=[helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [None, 2])],
        initializer=[
            helper.make_tensor("coef", TensorProto.FLOAT, [n_features, 1], coef.flatten().tolist()),
            helper.make_tensor("intercept", TensorProto.FLOAT, [1], intercept.tolist()),
            helper.make_tensor("one", TensorProto.FLOAT, [1], one.tolist()),
        ],
    )

    model_proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model_proto.ir_version = 10
    onnx.checker.check_model(model_proto)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(model_proto.SerializeToString())


def _is_simple_logistic_like(pipeline: Any) -> bool:
    return hasattr(pipeline, "weights") and hasattr(pipeline, "predict_proba")


def export_pipeline_to_onnx(pipeline: Any, output_path: Path, n_features: int) -> dict[str, Any]:
    try:
        if _is_simple_logistic_like(pipeline):
            _export_simple_logistic_to_onnx(pipeline, output_path, n_features)
            return {"success": True, "path": str(output_path), "error": None}

        _patch_onnx_mapping_compat()
        _patch_onnx_helper_compat()
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

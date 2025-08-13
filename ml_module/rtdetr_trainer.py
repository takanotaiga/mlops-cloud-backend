from __future__ import annotations

import os
from typing import List, Dict, Any, Optional


def train_rtdetr(dataset_dir: str, out_dir: str, *, epochs: int = 2, imgsz: int = 640,
                 base_model: str = "rtdetr-l.pt", export_engine: bool = True,
                 export_int8: bool = True) -> Dict[str, Any]:
    import yaml
    from ultralytics import RTDETR
    import glob

    # ✅ Ultralytics 設定書き換え
    ultra_settings_path = "/root/.config/Ultralytics/settings.yaml"
    if os.path.exists(ultra_settings_path):
        with open(ultra_settings_path, "r") as f:
            settings = yaml.safe_load(f)
        settings["datasets_dir"] = "/workspace/src"
        with open(ultra_settings_path, "w") as f:
            yaml.safe_dump(settings, f)

    os.makedirs(out_dir, exist_ok=True)
    data_yaml = os.path.join(dataset_dir, "data.yaml")

    use_pretrained = False
    model_ref = base_model
    try:
        if os.path.exists(base_model):
            use_pretrained = True
        else:
            model_ref = base_model.replace(".pt", ".yaml")
            use_pretrained = False
        model = RTDETR(model_ref)
    except Exception:
        model_ref = base_model.replace(".pt", ".yaml")
        model = RTDETR(model_ref)
        use_pretrained = False

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        name="train_result",
        project=out_dir,
        exist_ok=True,
        pretrained=use_pretrained,
    )

    pt_candidates = glob.glob(os.path.join(out_dir, "train_result*", "weights", "best.pt"))
    best_pt = pt_candidates[0] if pt_candidates else None

    engine_path, onnx_path = None, None
    if export_engine and best_pt:
        try:
            m = RTDETR(best_pt)
            m.export(format="engine", int8=export_int8, data=data_yaml)
        except Exception:
            pass
        engine_candidates = glob.glob(os.path.join(out_dir, "train_result*", "weights", "best.engine"))
        onnx_candidates = glob.glob(os.path.join(out_dir, "train_result*", "weights", "best.onnx"))
        engine_path = engine_candidates[0] if engine_candidates else None
        onnx_path = onnx_candidates[0] if onnx_candidates else None

    return {"pt": best_pt, "engine": engine_path, "onnx": onnx_path}


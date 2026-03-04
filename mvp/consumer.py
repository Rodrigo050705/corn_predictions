import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from db import init_db, has_hash, insert_result
from hashing import sha256_file
from io_utils import list_images, read_rgb, safe_stem
from model import load_meta, build_preprocess, load_densenet121_from_state_dict, predict_softmax
from campp import GradCAMPlusPlus, densenet_target_layer
from masking import cam_to_mask_adaptive, cam_is_flat
from annotate import annotate_with_mask, draw_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, default="input_images")
    ap.add_argument("--output_dir", type=str, default="annotated_images")
    ap.add_argument("--db_path", type=str, default="results.sqlite")
    ap.add_argument("--model_path", type=str, required=True)   # state_dict
    ap.add_argument("--meta_path", type=str, required=True)
    ap.add_argument("--poll_seconds", type=float, default=5.0)
    ap.add_argument("--run_once", action="store_true")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Mask tuning
    ap.add_argument("--mask_q_start", type=float, default=0.92)
    ap.add_argument("--mask_max_area", type=float, default=0.20)
    ap.add_argument("--mask_min_area", type=float, default=0.002)
    ap.add_argument("--highlight_strength", type=float, default=0.55)
    ap.add_argument("--no_contour", action="store_true")
    ap.add_argument("--skip_healthy_mask", action="store_true")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = init_db(Path(args.db_path))
    device = torch.device(args.device)

    meta = load_meta(Path(args.meta_path))
    classes = meta["classes"]
    tfm = build_preprocess(meta)

    model = load_densenet121_from_state_dict(Path(args.model_path), meta, device)
    campp = GradCAMPlusPlus(model, densenet_target_layer(model))

    print("Consuming from:", input_dir.resolve(), flush=True)
    print("Writing to:", output_dir.resolve(), flush=True)
    print("DB:", Path(args.db_path).resolve(), flush=True)
    print("Device:", device, flush=True)

    while True:
        files = list_images(input_dir)

        for p in files:
            sha = sha256_file(p)
            if has_hash(conn, sha):
                continue

            created_at = datetime.now(timezone.utc).isoformat()
            notes = None
            mask_path = None
            annot_path = None

            try:
                rgb = read_rgb(p)
                H, W = rgb.shape[:2]

                x = tfm(Image.fromarray(rgb)).unsqueeze(0).to(device)

                probs_t = predict_softmax(model, x)
                probs = probs_t.numpy()
                pred_idx = int(np.argmax(probs))
                pred_label = classes[pred_idx]
                pred_prob = float(probs[pred_idx])

                # mask default
                mask = np.zeros((H, W), dtype=np.uint8)

                if not (args.skip_healthy_mask and pred_label.strip().lower() == "healthy"):
                    x2 = x.clone().requires_grad_(True)
                    cam = campp(x2, class_idx=pred_idx).numpy().astype(np.float32)

                    if cam_is_flat(cam, min_range=0.15):
                        notes = "CAM too flat; mask suppressed."
                    else:
                        mask = cam_to_mask_adaptive(
                            cam, out_h=H, out_w=W,
                            q_start=args.mask_q_start,
                            max_area_frac=args.mask_max_area,
                            min_area_frac=args.mask_min_area
                        )

                annot = annotate_with_mask(
                    rgb, mask,
                    highlight_strength=args.highlight_strength,
                    draw_contour=(not args.no_contour)
                )
                annot = draw_label(annot, f"{pred_label} ({pred_prob:.2f})")

                base = safe_stem(p)
                cls = pred_label.replace(" ", "_")
                annot_file = output_dir / f"{base}__{cls}__annot.png"
                Image.fromarray(annot).save(annot_file)
                annot_path = str(annot_file)

                probs_json = json.dumps({classes[i]: float(probs[i]) for i in range(len(classes))})

                insert_result(
                    conn=conn,
                    created_at=created_at,
                    input_path=str(p),
                    sha256=sha,
                    pred_label=pred_label,
                    pred_prob=pred_prob,
                    probs_json=probs_json,
                    annotated_path=annot_path,
                    notes=notes,
                )

                print(f"Processed: {p.name} -> {pred_label} ({pred_prob:.2f})", flush=True)

            except Exception as e:
                insert_result(
                    conn=conn,
                    created_at=created_at,
                    input_path=str(p),
                    sha256=sha,
                    pred_label="error",
                    pred_prob=0.0,
                    probs_json="{}",
                    annotated_path=None,
                    notes=f"{type(e).__name__}: {e}",
                )
                print(f"ERROR: {p} -> {type(e).__name__}: {e}", flush=True)

        if args.run_once:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
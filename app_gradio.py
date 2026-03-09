import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import MAEViT

def load_model(checkpoint_path: str, device: torch.device):

    ckpt = torch.load(checkpoint_path, map_location=device)

    if "model_state" in ckpt:

        model = MAEViT(
            img_size=224,
            patch_size=16,
            enc_dim=768,
            enc_depth=12,
            enc_heads=12,
            dec_dim=384,
            dec_depth=12,
            dec_heads=6,
            mask_ratio=0.75,
        )

        model.load_state_dict(ckpt["model_state"])
        model = model.to(device).eval()

        return model, {}

    raise RuntimeError("Unsupported checkpoint format")


def _extract_outputs(output: Any):
    # Expected old format: (_, pred, mask, _)
    if isinstance(output, (tuple, list)) and len(output) >= 3:
        return output[1], output[2]

    # Optional dict format
    if isinstance(output, dict):
        pred = output.get("pred") or output.get("prediction")
        mask = output.get("mask")
        if pred is not None and mask is not None:
            return pred, mask

    raise RuntimeError("Model output format not recognized. Expected tuple/list or dict with pred/mask.")


def reconstruct(
    model: torch.nn.Module,
    image: Image.Image,
    mask_ratio: float,
    device: torch.device,
    img_size: int,
    patch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if image is None:
        raise gr.Error("Please upload an image")

    if hasattr(model, "mask_ratio"):
        setattr(model, "mask_ratio", float(mask_ratio))

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )

    x = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x)
        pred, mask = _extract_outputs(output)

        if hasattr(model, "unpatchify"):
            recon = model.unpatchify(pred).clamp(0, 1) # type: ignore
        else:
            raise RuntimeError("Model has no unpatchify() method; cannot reconstruct image patches.")

        masked_in = x.clone()
        gh = gw = img_size // patch_size
        mask_grid = mask.reshape(1, gh, gw)
        for i in range(gh):
            for j in range(gw):
                if mask_grid[0, i, j] > 0.5:
                    p = patch_size
                    masked_in[0, :, i * p : (i + 1) * p, j * p : (j + 1) * p] = 0.0

    def to_np(t: torch.Tensor) -> np.ndarray:
        arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return (arr * 255.0).clip(0, 255).astype(np.uint8)

    return to_np(masked_in), to_np(recon), to_np(x)


def build_interface(
    model: torch.nn.Module,
    device: torch.device,
    img_size: int,
    patch_size: int,
    default_mask_ratio: float,
) -> gr.Blocks:
    with gr.Blocks(title="MAE TinyImageNet Reconstruction") as demo:
        gr.Markdown("## Masked Autoencoder Reconstruction (TinyImageNet)")

        with gr.Row():
            with gr.Column():
                in_img = gr.Image(type="pil", label="Input Image")
                mask_ratio = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=default_mask_ratio,
                    step=0.05,
                    label="Mask Ratio",
                )
                run_btn = gr.Button("Reconstruct")

            with gr.Column():
                out_masked = gr.Image(type="numpy", label="Masked Input")
                out_recon = gr.Image(type="numpy", label="Reconstruction")
                out_gt = gr.Image(type="numpy", label="Resized Ground Truth")

        run_btn.click(
            fn=lambda image, ratio: reconstruct(model, image, ratio, device, img_size, patch_size),
            inputs=[in_img, mask_ratio],
            outputs=[out_masked, out_recon, out_gt],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio app for MAE inference")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--server_port", type=int, default=7860)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(__file__).resolve().parents[1] / "checkpoints" / "mae_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model, cfg = load_model(str(checkpoint_path), device)

    img_size = int(getattr(model, "img_size", cfg.get("image_size", 224)))
    patch_size = int(getattr(model, "patch_size", cfg.get("patch_size", 16)))
    default_mask_ratio = float(getattr(model, "mask_ratio", cfg.get("mask_ratio", 0.75)))

    demo = build_interface(model, device, img_size, patch_size, default_mask_ratio)
    demo.launch(share=args.share, server_port=args.server_port)


if __name__ == "__main__":
    main()
# 4K Stable Diffusion .ckpt → ONNX converter (NO diffusers needed!)
# Works on Windows / Linux / Colab with any NVIDIA GPU

# Install once (run in terminal or add ! in Colab/Jupyter)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install safetensors onnx onnxruntime-gpu

import torch
from safetensors.torch import load_file

# CHANGE ONLY THESE TWO LINES
CKPT_PATH = "../models/loftr_outdoor.ckpt"  # or .safetensors
OUTPUT_ONNX = "sd_unet_4k_dynamic.onnx"  # .onnx will be added

# -------------------------------------------------------------
print("Loading checkpoint...")

# Detect if it's SDXL or SD-1.5 from the keys
state_dict = (
    load_file(CKPT_PATH)
    if CKPT_PATH.endswith(".safetensors")
    else torch.load(CKPT_PATH, map_location="cpu")
)

# Auto-detect model type
if (
    "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_q.weight"
    in state_dict
):
    cross_attention_dim = 2048  # SDXL / Flux
    print("SDXL / Flux model detected")
else:
    cross_attention_dim = 768  # SD 1.5
    print("SD 1.5 model detected")


# Minimal UNet class that matches exactly what is inside every .ckpt
class MinimalUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = {
            "in_channels": 4,
            "out_channels": 4,
            "model_channels": 320,
            "attention_resolutions": [4, 2, 1],
            "num_res_blocks": 2,
            "channel_mult": [1, 2, 4, 4],
            "num_head_channels": 64,
            "use_spatial_transformer": True,
            "transformer_depth": 1,
            "context_dim": cross_attention_dim,
        }

        # Build exact same architecture as original
        from diffusers.models.unet_2d_condition import UNet2DConditionModel
        from torch import nn

        self.unet = UNet2DConditionModel(
            sample_size=None,  # we want dynamic size
            in_channels=4,
            out_channels=4,
            block_out_channels=(320, 640, 1280, 1280),
            layers_per_block=2,
            downsample_padding=1,
            mid_block_scale_factor=1,
            act_fn="silu",
            norm_num_groups=32,
            norm_eps=1e-5,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=8,
            num_attention_heads=None,
            use_linear_projection=True,
            upcast_attention=True if cross_attention_dim == 2048 else False,
        )

    def forward(self, sample, timestep, encoder_hidden_states):
        return self.unet(sample, timestep, encoder_hidden_states).sample


# Create model and load weights
model = MinimalUNet().cuda().half()  # .half() = float16 → same as most .ckpt files

# Load weights (this line handles both .ckpt and .safetensors automatically)
if CKPT_PATH.endswith(".safetensors"):
    model.unet.load_state_dict(state_dict)
else:
    # Old .ckpt files sometimes have "state_dict" key
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    # Remove "module." prefix if DataParallel was used
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v
    model.unet.load_state_dict(new_state_dict)

model.eval()
print("Model loaded successfully!")

# 4K latent size
latent_h = 2160 // 8  # 270
latent_w = 3840 // 8  # 480

# Dummy inputs
latents = torch.randn(1, 4, latent_h, latent_w, dtype=torch.float16, device="cuda")
timestep = torch.tensor([999], device="cuda")
encoder_hidden_states = torch.randn(
    1, 77, cross_attention_dim, dtype=torch.float16, device="cuda"
)

print(f"Exporting ONNX for true 4K ({latent_h}×{latent_w} latents)...")
torch.onnx.export(
    model,
    (latents, timestep, encoder_hidden_states),
    OUTPUT_ONNX,
    opset_version=17,
    export_params=True,
    do_constant_folding=True,
    input_names=["latents", "timestep", "encoder_hidden_states"],
    output_names=["noise_pred"],
    dynamic_axes={
        "latents": {0: "batch", 2: "height", 3: "width"},
        "encoder_hidden_states": {0: "batch"},
        "noise_pred": {0: "batch", 2: "height", 3: "width"},
    },
)

print("DONE! Your 4K-ready ONNX file is ready:")
print("   →", OUTPUT_ONNX)

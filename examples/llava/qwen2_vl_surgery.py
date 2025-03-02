import argparse
from typing import Dict
import os
import numpy as np

import torch
from gguf import *
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
)

VISION = "clip.vision"


def k(raw_key: str, arch: str) -> str:
    return raw_key.format(arch=arch)


def to_gguf_name(name: str) -> str:
    og = name
    name = name.replace("text_model", "t").replace("visual", "v")
    name = name.replace("blocks", "blk").replace("embeddings.", "")
    name = name.replace("attn.", "attn_")

    # Handle new Qwen2.5 MLP structure
    if "mlp.gate_proj" in name:
        name = name.replace("mlp.gate_proj", "ffn_gate")
    elif "mlp.up_proj" in name:
        name = name.replace("mlp.up_proj", "ffn_up")
    elif "mlp.down_proj" in name:
        name = name.replace("mlp.down_proj", "ffn_down")
    else:
        name = name.replace("mlp.fc1", "ffn_down").replace("mlp.fc2", "ffn_up")

    name = name.replace("proj.", "out.")
    name = name.replace("norm1", "ln1").replace("norm2", "ln2")
    name = name.replace("merger.mlp", 'mm')

    # For RMSNorm, which doesn't have bias
    if "weight_g" in name:
        name = name.replace("weight_g", "weight")

    # Special handling for merger tensors to match clip-debug.cpp expectations
    if "merger.mlp" in name:
        # Extract the layer number
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part == "mlp" and i + 1 < len(parts):
                layer_num = parts[i + 1]
                # Map the merger layers to the expected GGUF tensor names
                # Note: clip-debug.cpp looks for mm.0.* and mm.2.* (not mm.1.*)
                if layer_num == "0":
                    name = name.replace(f"merger.mlp.{layer_num}", "mm.0")
                elif layer_num == "1":
                    name = name.replace(f"merger.mlp.{layer_num}", "mm.2")
                break

    print(f"[to_gguf_name] {og} --> {name}")
    return name


def find_vision_tensors(model, dtype, hidden_size) -> Dict[str, np.ndarray]:
    visual = model.visual
    tensor_map = {}

    for name, ten in visual.state_dict().items():
        ten = ten.numpy()
        if 'qkv' in name:
            if ten.ndim == 2:  # weight
                c3, _ = ten.shape
            else:  # bias
                c3 = ten.shape[0]
            assert c3 % 3 == 0
            c = c3 // 3
            wq = ten[:c]
            wk = ten[c: c * 2]
            wv = ten[c * 2:]
            tensor_map[to_gguf_name(f"visual.{name}").replace("qkv", "q")] = wq
            tensor_map[to_gguf_name(f"visual.{name}").replace("qkv", "k")] = wk
            tensor_map[to_gguf_name(f"visual.{name}").replace("qkv", "v")] = wv
        elif 'merger' in name:
            if name.endswith("ln_q.weight_g"):
                tensor_map['v.post_ln.weight'] = ten
            elif name.endswith("ln_q.bias") and 'weight_g' not in name:
                tensor_map['v.post_ln.bias'] = ten
            else:
                # Handle merger tensors with special attention to naming
                # First, determine if this is a layer 0 or layer 1 tensor
                if "merger.mlp.0" in name:
                    # First layer gets mapped to mm.0.*
                    if "weight" in name:
                        tensor_map["mm.0.weight"] = ten
                    elif "bias" in name:
                        tensor_map["mm.0.bias"] = ten
                elif "merger.mlp.1" in name:
                    # Second layer gets mapped to mm.2.* (not mm.1.*)
                    if "weight" in name:
                        tensor_map["mm.2.weight"] = ten
                    elif "bias" in name:
                        tensor_map["mm.2.bias"] = ten
                else:
                    # For any other tensors, use the standard naming conversion
                    tensor_map[to_gguf_name(name)] = ten
        elif 'patch_embed.proj.weight' in name:
            # Handle different temporal patch sizes more flexibly
            c1, c2, kt, kh, kw = ten.shape
            print(f"Temporal patch size detected: {kt}")
            # Process each temporal slice separately
            for t in range(kt):
                if t == 0:
                    tensor_map["v.patch_embd.weight"] = ten[:, :, t, ...]
                else:
                    tensor_map[f"v.patch_embd.weight.{t}"] = ten[:, :, t, ...]
        else:
            tensor_map[to_gguf_name(f"visual.{name}")] = ten

    for new_name, ten in tensor_map.items():
        if ten.ndim <= 1 or new_name.endswith("_norm.weight"):
            tensor_map[new_name] = ten.astype(np.float32)
        else:
            tensor_map[new_name] = ten.astype(dtype)

    # For Qwen2.5, create a properly sized position embedding tensor
    # Size it based on the model's hidden dimension and expected sequence length
    seq_length = 40 * 40  # Approximate max sequence length
    tensor_map["v.position_embd.weight"] = np.zeros([seq_length, hidden_size], dtype=np.float32)
    print("WARNING: Using zero-initialized position embeddings. This is a placeholder.")

    return tensor_map


def main(args):
    if args.data_type == 'fp32':
        dtype = torch.float32
        np_dtype = np.float32
        ftype = 0
    elif args.data_type == 'fp16':
        dtype = torch.float32
        np_dtype = np.float16
        ftype = 1
    else:
        raise ValueError()

    local_model = False
    model_path = ""
    model_name = args.model_name
    print("model_name: ", model_name)

    # Load the model with the specific Qwen2.5 class
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype, device_map="cpu"
    )
    cfg = model.config
    vcfg = cfg.vision_config

    if os.path.isdir(model_name):
        local_model = True
        if model_name.endswith(os.sep):
            model_name = model_name[:-1]
        model_path = model_name
        model_name = os.path.basename(model_name)
    fname_out = f"{model_name.replace('/', '-').lower()}-vision.gguf"

    fout = GGUFWriter(path=fname_out, arch="clip")
    fout.add_description("image encoder for Qwen2.5VL")

    fout.add_file_type(ftype)
    fout.add_bool("clip.has_text_encoder", False)
    fout.add_bool("clip.has_vision_encoder", True)
    fout.add_bool("clip.has_qwen2vl_merger", True)
    fout.add_bool("clip.is_qwen2_5", True)  # Flag to identify Qwen2.5 models
    fout.add_string("clip.projector_type", "qwen2vl_merger")

    print(cfg.vision_config)
    # SiLU activation
    fout.add_bool("clip.use_silu", True)
    fout.add_bool("clip.use_gelu", False)

    # Add missing keys
    # 1. mm_patch_merge_type - Qwen2.5 uses a flat merge type
    fout.add_string("clip.vision.mm_patch_merge_type", "flat")

    # 2. Calculate image_grid_pinpoints based on model configuration
    # Starting with base_size (patch_size * resolution_factor)
    base_size = vcfg.patch_size * 16  # Standard base factor
    multipliers = [1.0, 1.5, 2.0, 2.5]  # Standard resolution multipliers
    grid_pinpoints = []
    for m in multipliers:
        size = int(base_size * m)
        grid_pinpoints.extend([size, size])  # Add width and height

    print(f"Calculated grid_pinpoints: {grid_pinpoints}")
    fout.add_array("clip.vision.image_grid_pinpoints", grid_pinpoints)

    # 3. feature_layer - Use the fullatt_block_indexes for feature extraction
    if hasattr(vcfg, 'fullatt_block_indexes') and vcfg.fullatt_block_indexes:
        feature_layers = vcfg.fullatt_block_indexes
    else:
        feature_layers = [vcfg.depth]  # Use the last layer as fallback

    print(f"Using feature layers: {feature_layers}")
    fout.add_array("clip.vision.feature_layer", feature_layers)

    # 4. Add window_size from config
    if hasattr(vcfg, 'window_size'):
        print(f"Setting window_size: {vcfg.window_size}")
        fout.add_uint32("clip.vision.window_size", vcfg.window_size)

    # 5. Add fullatt_block_indexes from config
    if hasattr(vcfg, 'fullatt_block_indexes'):
        print(f"Setting fullatt_block_indexes: {vcfg.fullatt_block_indexes}")
        fout.add_array("clip.vision.fullatt_block_indexes", vcfg.fullatt_block_indexes)

    # 6. Add spatial_merge_size from config
    if hasattr(vcfg, 'spatial_merge_size'):
        print(f"Setting spatial_merge_size: {vcfg.spatial_merge_size}")
        fout.add_uint32("clip.vision.spatial_merge_size", vcfg.spatial_merge_size)

    # 7. Add temporal_patch_size from config
    if hasattr(vcfg, 'temporal_patch_size'):
        print(f"Setting temporal_patch_size: {vcfg.temporal_patch_size}")
        fout.add_uint32("clip.vision.temporal_patch_size", vcfg.temporal_patch_size)

    # 8. image_crop_resolution - Calculate based on patch size and expected resolution
    patch_count = 40  # Typical value for high-res image processing
    image_size = vcfg.patch_size * patch_count
    fout.add_uint32("clip.vision.image_crop_resolution", image_size)

    tensor_map = find_vision_tensors(model, np_dtype, vcfg.hidden_size)
    for name, data in tensor_map.items():
        fout.add_tensor(name, data)

    fout.add_uint32("clip.vision.patch_size", vcfg.patch_size)
    fout.add_uint32("clip.vision.image_size", image_size)
    fout.add_uint32(k(KEY_EMBEDDING_LENGTH, VISION), vcfg.hidden_size)
    fout.add_uint32("clip.vision.projection_dim", vcfg.hidden_size)
    fout.add_uint32(k(KEY_ATTENTION_HEAD_COUNT, VISION), vcfg.num_heads)
    fout.add_float32(k(KEY_ATTENTION_LAYERNORM_EPS, VISION), 1e-6)
    fout.add_uint32(k(KEY_BLOCK_COUNT, VISION), vcfg.depth)
    fout.add_uint32(k(KEY_FEED_FORWARD_LENGTH, VISION), vcfg.intermediate_size)
    fout.add_name(model_name)

    # Load the processor using the specific Qwen2.5 processor class
    # Explicitly set use_fast=True to use the faster tokenizer implementation
    # This avoids warnings and will be the default in Transformers v4.48
    if local_model:
        processor = Qwen2_5_VLProcessor.from_pretrained(model_path, use_fast=False)
    else:
        processor = Qwen2_5_VLProcessor.from_pretrained(model_name, use_fast=False)

    # Get the image mean and std values from the processor
    fout.add_array("clip.vision.image_mean", processor.image_processor.image_mean)
    fout.add_array("clip.vision.image_std", processor.image_processor.image_std)

    fout.write_header_to_file()
    fout.write_kv_data_to_file()
    fout.write_tensors_to_file()
    fout.close()
    print("save model as: ", fname_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", nargs='?', default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--data_type", nargs='?', choices=['fp32', 'fp16'], default="fp32")
    args = parser.parse_args()
    main(args)

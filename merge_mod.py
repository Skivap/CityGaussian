#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Merge multiple 3DGS .ply models into one.
# Copyright (C) 2023, Inria (GraphDeco)
# License: research/evaluation use as per LICENSE.md

import os
import sys
import argparse
from pathlib import Path

import torch

from gaussian_renderer import GaussianModel

def _is_dir_like_output(p: Path) -> bool:
    # Treat as directory if it exists as a dir, or if it has no suffix
    # and parent exists (so we can mkdir below).
    return p.is_dir() or p.suffix.lower() != ".ply"

@torch.no_grad()
def merge_gaussians(input_paths, output_path, sh_degree=3, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")

    # Decide final save path
    output_path = Path(output_path)
    if _is_dir_like_output(output_path):
        save_dir = output_path
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "point_cloud.ply"
    else:
        save_path = output_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

    merged = GaussianModel(sh_degree)
    num_loaded = 0

    for ip in input_paths:
        ip = Path(ip)
        if not ip.exists():
            print(f"[WARN] Skipping (not found): {ip}")
            continue
        g = GaussianModel(sh_degree)
        g.load_ply(str(ip))

        # Move internal tensors to device to enable cat()
        # (GaussianModel typically stores tensors as buffers/params)
        # Guard against attributes possibly missing in older forks.
        def _get(attr):
            return getattr(g, attr, None)

        xyz = g.get_xyz
        fdc = _get("_features_dc")
        frest = _get("_features_rest")
        sca = _get("_scaling")
        rot = _get("_rotation")
        opa = _get("_opacity")
        rad = getattr(g, "max_radii2D", None)

        # First model initializes merged buffers
        if num_loaded == 0:
            merged._xyz = xyz
            merged._features_dc = fdc
            merged._features_rest = frest
            merged._scaling = sca
            merged._rotation = rot
            merged._opacity = opa
            if rad is not None:
                merged.max_radii2D = rad
        else:
            merged._xyz = torch.cat([merged._xyz, xyz], dim=0)
            merged._features_dc = torch.cat([merged._features_dc, fdc], dim=0)
            merged._features_rest = torch.cat([merged._features_rest, frest], dim=0)
            merged._scaling = torch.cat([merged._scaling, sca], dim=0)
            merged._rotation = torch.cat([merged._rotation, rot], dim=0)
            merged._opacity = torch.cat([merged._opacity, opa], dim=0)
            if rad is not None and hasattr(merged, "max_radii2D"):
                merged.max_radii2D = torch.cat([merged.max_radii2D, rad], dim=0)

        num_loaded += 1
        print(f"Merged {xyz.shape[0]} points from: {ip}")

    if num_loaded == 0:
        raise FileNotFoundError("No valid input .ply files were found to merge.")

    print(f"Saving merged {merged.get_xyz.shape[0]} points to: {save_path}")
    merged.save_ply(str(save_path))
    print("Done.")

def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Merge multiple 3D Gaussian Splatting .ply models into one output."
    )
    p.add_argument(
        "--input", "-i", action="append", required=True,
        help="Path to an input .ply file. Repeat for multiple parts."
    )
    p.add_argument(
        "--output", "-o", required=True,
        help="Output path."
    )
    p.add_argument(
        "--sh_degree", type=int, default=3,
        help="Spherical Harmonics degree expected by GaussianModel (default: 3)."
    )
    p.add_argument(
        "--device", default="cuda",
        help="Device preference: 'cuda' or 'cpu' (default: cuda). Falls back to CPU if CUDA not available."
    )
    return p.parse_args(argv)

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    merge_gaussians(
        input_paths=args.input,
        output_path=args.output,
        sh_degree=args.sh_degree,
        device=args.device
    )
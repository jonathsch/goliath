# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import logging
import os
import sys
from copy import deepcopy
from typing import List

import dotenv
import mediapy
import numpy as np
import torch as th
from addict import Dict as AttrDict
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

# from ca_code.utils.dataloader import BodyDataset, collate_fn
from ca_code.utils.becominglit_dataloader import BecomingLitDataset, collate_fn
from ca_code.utils.image import linear2srgb
from ca_code.utils.light_decorator import EnvSpinDecorator
from ca_code.utils.module_loader import load_from_config
from ca_code.utils.train import load_checkpoint, to_device

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

# Settings
DRIVING_SUBJECT = "1001"
DRIVING_SEQUENCE = "EMOTIONS"
CAM = "222200037"
# FRAME_SUBSET = list(range(500, 2047))
FRAME_SUBSET = None


def main(config: DictConfig):
    device = th.device("cuda:0")
    remote = os.getenv("REMOTE")

    # if remote:
    #     config.train.run_dir = "/mnt" + config.train.run_dir

    model_dir = config.train.run_dir
    save_dir = f"{model_dir}/cra_{DRIVING_SUBJECT}_{DRIVING_SEQUENCE}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/rgb", exist_ok=True)
    os.makedirs(f"{save_dir}/driving", exist_ok=True)
    os.makedirs(f"{save_dir}/diff", exist_ok=True)
    os.makedirs(f"{save_dir}/spec", exist_ok=True)

    ckpt_path = f"{model_dir}/checkpoints/model.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = f"{model_dir}/checkpoints/latest.pt"

    # Basic data setup
    config.data.cameras_subset = ["222200037"]
    config.data.root_path = os.getenv("BECOMINGLIT_DATASET_PATH")
    config.data.fully_lit_only = True
    config.data.partially_lit_only = False
    del config.data.sequences
    config.data.sequence = DRIVING_SEQUENCE

    driving_data_config = deepcopy(config.data)
    driving_data_config.subject = DRIVING_SUBJECT
    driving_data_config.sequence = DRIVING_SEQUENCE
    # driving_data_config.apply_alpha = False

    driving_dataset = BecomingLitDataset(**driving_data_config, frames_subset=FRAME_SUBSET)
    batch_filter_fn = driving_dataset.batch_filter

    tgt_dataset = BecomingLitDataset(**config.data, frames_subset=FRAME_SUBSET)
    tgt_flame_shape = tgt_dataset.flame_shape_code.clone().to(device)

    static_assets = AttrDict(tgt_dataset.static_assets)

    # building the model
    model = (
        load_from_config(
            config.model,
            assets=static_assets,
        )
        .to(device)
        .eval()
    )

    # loading model checkpoint
    load_checkpoint(
        ckpt_path,
        modules={"model": model},
        strict=False,
    )

    # disabling training-only stuff
    model.learn_blur_enabled = False
    model.cal_enabled = False

    config.dataloader.shuffle = False
    config.dataloader.batch_size = 1
    config.dataloader.num_workers = 4

    loader = DataLoader(
        driving_dataset,
        collate_fn=collate_fn,
        **config.dataloader,
    )

    logger.info(f"Driving subject {DRIVING_SUBJECT}")
    logger.info(f"Driving sequence {DRIVING_SEQUENCE}")
    # logger.info(f"Setting dataset to {config.data.sequence}")

    model_e = EnvSpinDecorator(
        model, envmap_path="/rhome/jschmidt/projects/becominglit/assets/envmaps/shanghai_bund_1k.hdr"
    )

    # forward
    catted_imgs = []
    img_idx = 0
    for i, batch in enumerate(tqdm(loader, total=len(loader))):
        batch = to_device(batch, device)
        batch_filter_fn(batch)
        batch["flame_params"]["shape"] = tgt_flame_shape

        with th.no_grad():
            preds = model_e(**batch, render_aux=True, index=[i])

        # visualizing
        pred_rgb = {k: linear2srgb(v).clamp(0.0, 1.0) for k, v in preds["rgb"].items()}
        gt_rgb = linear2srgb(batch["image"]).clamp(0.0, 1.0)

        rgb = (pred_rgb["full"][0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        diff = (pred_rgb["diff"][0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        spec = (pred_rgb["spec"][0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        gt = (gt_rgb[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")

        Image.fromarray(rgb).save(f"{save_dir}/rgb/{img_idx}.png")
        Image.fromarray(diff).save(f"{save_dir}/diff/{img_idx}.png")
        Image.fromarray(spec).save(f"{save_dir}/spec/{img_idx}.png")
        Image.fromarray(gt).save(f"{save_dir}/driving/{img_idx}.png")

        catted = np.concatenate([gt, rgb], axis=1)
        catted_imgs.append(catted)

        img_idx += 1

        if img_idx > 256:
            break

    catted_imgs = np.stack(catted_imgs, axis=0)
    mediapy.write_video(f"{save_dir}/cat.mp4", catted_imgs, fps=24, crf=15)


if __name__ == "__main__":
    dotenv.load_dotenv()

    config_path: str = sys.argv[1]
    console_commands: List[str] = sys.argv[2:]

    config = OmegaConf.load(config_path)
    config_cli = OmegaConf.from_cli(args_list=console_commands)
    if config_cli:
        logger.info("Overriding with the following args values:")
        logger.info(f"{OmegaConf.to_yaml(config_cli)}")
        config = OmegaConf.merge(config, config_cli)

    main(config)

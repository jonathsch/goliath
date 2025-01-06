# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import logging
import os
import sys
from typing import List

import dotenv
import torch as th
from addict import Dict as AttrDict
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
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


def main(config: DictConfig):
    device = th.device("cuda:0")
    remote = os.getenv("REMOTE")

    if remote:
        config.train.run_dir = "/mnt" + config.train.run_dir

    model_dir = config.train.run_dir
    os.makedirs(f"{model_dir}/tmp", exist_ok=True)
    os.makedirs(f"{model_dir}/nvs", exist_ok=True)

    ckpt_path = f"{model_dir}/checkpoints/600000.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = f"{model_dir}/checkpoints/latest.pt"

    # Setup dataset for target sequence
    config.data.root_path = os.getenv("BECOMINGLIT_DATASET_PATH")
    config.data.fully_lit_only = False
    config.data.partially_lit_only = False
    config.data.sequence = config.self_reenact.tgt_seq

    dataset = BecomingLitDataset(**config.data)
    # batch_filter_fn = dataset.batch_filter

    static_assets = AttrDict(dataset.static_assets)

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

    config.test.data.root_path = os.getenv("BECOMINGLIT_DATASET_PATH")
    dataset = BecomingLitDataset(**config.test.data)
    batch_filter_fn = dataset.batch_filter

    config.dataloader.shuffle = False
    config.dataloader.batch_size = 1
    config.dataloader.num_workers = 4

    loader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        **config.dataloader,
    )

    # Set up metrics
    psnr = PeakSignalNoiseRatio(data_range=(0.0, 1.0)).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0)).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

    # forward
    for i, batch in enumerate(tqdm(loader)):
        batch = to_device(batch, device)
        batch_filter_fn(batch)
        with th.no_grad():
            preds = model(**batch)

        # visualizing
        pred_rgb = linear2srgb(preds["rgb"]).clamp(0.0, 1.0)
        gt_rgb = linear2srgb(batch["image"]).clamp(0.0, 1.0)

        psnr.update(pred_rgb, gt_rgb)
        ssim.update(pred_rgb, gt_rgb)
        lpips.update(pred_rgb, gt_rgb)

        if i % 10 == 0:
            rgb_preds_grid = make_grid(th.cat([gt_rgb, pred_rgb]), nrow=4)
            save_image(rgb_preds_grid, f"{model_dir}/nvs/{i:05d}.png")

    logger.info(f"PSNR: {psnr.compute().item()}, SSIM: {ssim.compute().item()}, LPIPS: {lpips.compute().item()}")
    with open(f"{model_dir}/metrics_self_reenact_{config.self_reenact.tgt_seq}.json", "w") as f:
        json.dump({"psnr": psnr.compute().item(), "ssim": ssim.compute().item(), "lpips": lpips.compute().item()}, f)

    # os.system(
    #     f"ffmpeg -y -framerate 24 -i '{model_dir}/tmp/%d.png' -b:v 8000000 -c:v mpeg4 -g 10 -pix_fmt yuv420p {model_dir}/_self_reenact_{config.self_reenact.tgt_seq}.mp4 -y"
    # )

    model_e = EnvSpinDecorator(
        model,
        envmap_path="./envmaps/metro_noord_1k.hdr",
        ydown=True,
        env_scale=8.0,
    ).to(device)

    # forward
    for i, batch in enumerate(tqdm(loader)):
        batch = to_device(batch, device)
        batch_filter_fn(batch)
        with th.no_grad():
            preds = model_e(**batch, index=[i])

        # visualizing
        rgb_preds_grid = make_grid(linear2srgb(preds["rgb"]), nrow=4)
        save_image(rgb_preds_grid, f"{model_dir}/tmp/{i}.png")

        if i > 256:
            break

    os.system(
        f"ffmpeg -y -framerate 72 -i '{model_dir}/tmp/%d.png' -b:v 8000000 -c:v mpeg4 -pix_fmt yuv420p {model_dir}/_nvs_env.mp4 -y"
    )


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

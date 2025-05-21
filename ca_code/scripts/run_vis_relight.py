# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
import sys
from typing import List
from pathlib import Path
import shutil

import numpy as np
import dotenv
import torch as th
from addict import Dict as AttrDict
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import lovely_tensors as lt

lt.monkey_patch()

from ca_code.utils.dataloader import BodyDataset, collate_fn
from ca_code.utils.becominglit_dataloader import BecomingLitDataset, collate_fn
from ca_code.utils.image import linear2srgb
from ca_code.utils.light_decorator import EnvSpinDecorator, SingleLightCycleDecorator
from ca_code.utils.module_loader import load_from_config
from ca_code.utils.train import load_checkpoint, to_device
# from ca_code.utils.gs_to_mesh import get_mesh

logger = logging.getLogger(__name__)

MODALITY = "full"


def main(config: DictConfig):
    device = th.device("cuda:0")

    # remote = os.getenv("REMOTE")
    # if remote:
    #     config.train.run_dir = "/mnt" + config.train.run_dir

    model_dir = config.train.run_dir
    os.makedirs(f"{model_dir}/tmp", exist_ok=True)

    # ckpt_path = f"{model_dir}/checkpoints/model.pt"
    ckpt_path = f"{model_dir}/checkpoints/model.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = f"{model_dir}/checkpoints/latest.pt"
        # ckpt_path = f"{model_dir}/checkpoints/600000.pt"

    config.data.root_path = os.getenv("BECOMINGLIT_DATASET_PATH")
    config.data.fully_lit_only = True
    config.data.partially_lit_only = False

    if "sequences" in config.data:
        del config.data["sequences"]
        config.data.sequence = "FREE"
    # dataset = BodyDataset(**config.data)
    data_config = config.data

    # dataset = BecomingLitDataset(**data_config)
    # ckpt_path = "/mnt/cluster/valinor/jschmidt/goliath/m--20230714--0903--QVC422--pilot--ProjectGoliath--Head/model/model.pt"
    model_dir = Path(ckpt_path).parent
    print(model_dir)
    save_dir_point = f"{model_dir}/point_{config.data.sequence}_{MODALITY}"
    save_dir_env = f"{model_dir}/env_{config.data.sequence}"
    os.makedirs(save_dir_point, exist_ok=True)
    os.makedirs(save_dir_env, exist_ok=True)

    # config.data.split = "test"
    config.data.fully_lit_only = True
    config.data.partially_lit_only = False
    config.data.cameras_subset = ["222200037"]
    # dataset = BodyDataset(**config.data)
    dataset = BecomingLitDataset(**data_config)
    batch_filter_fn = dataset.batch_filter

    static_assets = AttrDict(dataset.static_assets)

    config.dataloader.shuffle = False
    config.dataloader.batch_size = 1
    config.dataloader.num_workers = 4

    # dataset.cameras = ["401892"]
    # dataset.cameras = ["222200037"]

    static_assets = AttrDict(dataset.static_assets)

    config.dataloader.shuffle = False
    config.dataloader.batch_size = 1
    config.dataloader.num_workers = 4

    # dataset.cameras = ["222200037"]

    loader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        **config.dataloader,
    )

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

    light_positions = np.load(Path.home().joinpath("light_pos.npy"))
    # model_p = SingleLightCycleDecorator(model, light_positions).to(device)

    # # campos_path_x = th.sin(th.linspace(0, 2 * th.pi, 256))
    # # campos_path_y = th.zeros_like(campos_path_x)
    # # campos_path_z = th.cos(th.linspace(0, 2 * th.pi, 256))
    # # campos_path = th.stack([campos_path_x, campos_path_y, campos_path_z], dim=1) * 2.0 # [256, 3]
    # # campos_path = to_device(campos_path, device)

    # forward
    # for i in range(256):
    #     batch = next(iter(loader))
    #     batch = to_device(batch, device)
    #     batch_filter_fn(batch)
    #     with th.no_grad():
    #         preds = model_p(**batch, index=[180 + i])

    # #         # if "hand" in model_dir:
    # #         #     preds["rgb"] = preds["rgb"] / 255.0

    #     # visualizing
    #     rgb_preds_grid = make_grid(linear2srgb(preds["rgb"]), nrow=4)
    #     save_image(rgb_preds_grid, f"{model_dir}/tmp/{i}.png")

    # os.system(
    #     f"ffmpeg -y -framerate 24 -i '{model_dir}/tmp/%d.png' -b:v 8000000 -c:v mpeg4 -g 10 -pix_fmt yuv420p {model_dir}/_point.mp4 -y"
    # )

    # download 1k hdr from https://polyhaven.com/a/metro_noord
    model_e = EnvSpinDecorator(
        model,
        # envmap_path="./envmaps/metro_noord_1k.hdr",
        envmap_path="/rhome/jschmidt/projects/becominglit/assets/envmaps/shanghai_bund_1k.hdr",
        ydown=False,
        env_scale=8.0,
        cycle=256,
    ).to(device)

    # forward
    for i, batch in enumerate(tqdm(loader)):
        batch = to_device(batch, device)
        batch_filter_fn(batch)
        with th.no_grad():
            preds = model_e(**batch, index=[i])

        # visualizing
        # rgb_preds_grid = make_grid(linear2srgb(preds["rgb"]), nrow=4)
        rgb_preds_grid = linear2srgb(preds["rgb"]["full"])
        envbg = linear2srgb(preds["rgb"]["envbg"])
        # save_image(rgb_preds_grid, f"{save_dir_env}/{i}.png")
        save_image(envbg, f"{save_dir_env}/envbg_{i}.png")

        # if i > 360:
        #     break

    # os.system(
    #     f"ffmpeg -y -framerate 24 -i '{save_dir_env}/%d.png' -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -crf 20 -c:v libx264 -g 10 -pix_fmt yuv420p {model_dir}/{config.data.sequence}_env.mp4 -y"
    # )


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

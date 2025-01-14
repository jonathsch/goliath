# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
import sys
from typing import List

import torch as th
from addict import Dict as AttrDict
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import wandb
# from ca_code.utils.dataloader import BodyDataset, collate_fn
from ca_code.utils.becominglit_dataloader import BecomingLitDataset, MultiSequenceBecomingLitDataset, collate_fn
from ca_code.utils.module_loader import build_optimizer
from ca_code.utils.train import load_checkpoint, load_from_config, train

logger = logging.getLogger(__name__)


def main(config: DictConfig):
    device = th.device("cuda:0")

    # train_dataset = BodyDataset(**config.data)
    # train_dataset = BecomingLitDataset(**config.data)
    train_dataset = MultiSequenceBecomingLitDataset(**config.data)
    batch_filter_fn = train_dataset.batch_filter

    static_assets = AttrDict(train_dataset.static_assets)

    model = load_from_config(config.model, assets=static_assets).to(device)
    optimizer = build_optimizer(config.optimizer, model)

    loss_fn = load_from_config(config.loss, assets=static_assets).to(device)

    os.makedirs(config.train.ckpt_dir, exist_ok=True)
    if "ckpt" in config.train:
        logger.info(f"loading checkpoint: {config.train.ckpt}")
        load_checkpoint(**config.train.ckpt, modules={"model": model})
    elif "resume" in config.train:
        logger.info(f"loading latest checkpoint from: {config.train.ckpt_dir}")
        load_checkpoint(config.train.ckpt_dir, modules={"model": model, "optimizer": optimizer})

    logger.info("starting training with the config:")
    logger.info(OmegaConf.to_yaml(config))
    OmegaConf.save(config, f"{config.train.run_dir}/config.yml")

    train_loader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        **config.dataloader,
    )

    wandb_run = wandb.init(
        project="RGCA",
        dir=config.train.run_dir,
        name=config.train.run_id,
        group=config.train.tag,
        tags=[str(config.sid), config.model_name],
        config=OmegaConf.to_container(config),
    ) if config.train.run_id != "debug" else None

    summary_fn = load_from_config(config.summary)

    train(
        model,
        loss_fn,
        optimizer,
        train_loader,
        config,
        summary_fn=summary_fn,
        batch_filter_fn=batch_filter_fn,
        wandb_run=wandb_run,
        saving_enabled=True,
        logging_enabled=True,
        summary_enabled=True,
    )


if __name__ == "__main__":
    config_path: str = sys.argv[1]
    console_commands: List[str] = sys.argv[2:]

    config = OmegaConf.load(config_path)
    config_cli = OmegaConf.from_cli(args_list=console_commands)
    if config_cli:
        logger.info("Overriding with the following args values:")
        logger.info(f"{OmegaConf.to_yaml(config_cli)}")
        config = OmegaConf.merge(config, config_cli)

    main(config)

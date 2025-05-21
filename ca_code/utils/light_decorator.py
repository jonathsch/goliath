# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

import cv2
import numpy as np
import torch as th
import torch.nn.functional as thf
from ca_code.utils import envmap


class EnvSpinDecorator(th.nn.Module):
    def __init__(
        self,
        mod,
        envmap_path,
        envmap_dist=10000.0,
        env_scale=18.0,
        cycle=256,
        sigma_step=0.2,
        miplevel=4,
        ydown=False,
    ):
        super(EnvSpinDecorator, self).__init__()
        self.mod = mod
        self.envmap_dist = envmap_dist

        self.env_scale = env_scale
        self.cycle = cycle
        self.sigma_step = sigma_step
        self.miplevel = miplevel
        self.ydown = ydown

        self._set_lightmap(envmap_path)

        L = 16
        theta, phi = np.meshgrid(
            (np.arange(L, dtype=np.float32) + 0.5) * np.pi / L,
            (np.arange(-L, L, dtype=np.float32) + 0.5) * np.pi / L,
            indexing="ij",
        )
        sph = np.stack(
            [np.sin(theta) * np.sin(phi), np.cos(theta), -np.sin(theta) * np.cos(phi)],
            axis=0,
        ).reshape((3, -1))
        self.register_buffer("sphvec", th.from_numpy(sph))

    def _set_lightmap(self, envmap_path):
        image = cv2.imread(envmap_path, -1)[:, :, ::-1]
        if self.ydown:
            image = image[::-1, ::-1]
        image = cv2.resize(image, (1024, 512), interpolation=cv2.INTER_AREA)
        image = np.ascontiguousarray(image)

        self.image = th.from_numpy(image).float()  # H x W x 3
        self.image = self.image.permute((2, 0, 1)).contiguous()  # 3 x H x W

        multisin = th.sin((th.arange(self.image.shape[1]) + 0.5) * np.pi / self.image.shape[1])[None, None, :, None]
        mipmap = [self.image[None].clone()]
        image = self.image[None].clone() * multisin
        num_sample = 4096
        for i in range(self.miplevel - 1):
            sigma = (i + 1) * self.sigma_step
            image = thf.interpolate(image, None, scale_factor=0.5, mode="area").cuda()
            height = image.shape[2]
            width = image.shape[3]
            theta, phi = th.meshgrid(
                (th.arange(height, dtype=th.float32) + 0.5) * np.pi / height,
                (th.arange(-width // 2, width // 2, dtype=th.float32) + 0.5) * np.pi * 2 / width,
            )

            vec = th.stack(
                [
                    th.sin(theta) * th.sin(phi),
                    th.cos(theta),
                    -th.sin(theta) * th.cos(phi),
                ],
                dim=0,
            ).cuda()[None]
            convolve = envmap.prefilterEnvmapSG(sigma, vec, image, num_sample)
            mipmap.append(convolve.data.cpu())
        for i, p in enumerate(mipmap):
            self.register_buffer(f"mipmap_{i}", p)

    def mipmap(self, bsize, device, scale=1.0):
        return [getattr(self, f"mipmap_{i}").expand(bsize, -1, -1, -1).to(device) * scale for i in range(self.miplevel)]

    def forward(self, **data):
        light_intensity = []
        envbg = []
        light_dir = []
        envmaps = []

        device = data["campos"].device
        batch_size = data["campos"].size(0)

        lightrots = th.zeros(batch_size, 3, 3).float().to(device)
        norm_scale = []
        for i in range(batch_size):
            index = data["index"][i]
            rot_y = 2.0 * np.pi * -index / self.cycle - (np.pi / 2.0)
            axis = th.Tensor([0.0, 1.0, 0.0])
            quat = th.Tensor(axis.tolist() + [rot_y]).float()
            quat = thf.normalize(quat[0:3], dim=0) * quat[3]
            rot_mat = envmap.rvec_to_R(quat)

            rotmat_z_180 = th.tensor([[1, 0, 0], [0, 1, 0], [0, 0, -1]], device=rot_mat.device, dtype=th.float32)
            new_env = envmap.rotate_envmap_mat(self.image, rot_mat)

            lightrots[i] = rot_mat
            perc90 = np.percentile(self.image.data.cpu().numpy(), 90)
            envbg.append(
                envmap.rotate_envmap_mat(new_env, rotmat_z_180) / (perc90 if perc90 > 0 else new_env.max().item()) * 255
            )

            new_env = thf.interpolate(new_env[None], (16, 32), mode="bilinear", antialias=True)[0]

            new_env_sin = (
                new_env * th.sin((th.arange(new_env.shape[1]) + 0.5) * np.pi / new_env.shape[1])[None, :, None]
            )
            new_env = self.env_scale * new_env / new_env_sin.sum()
            norm_scale.append(self.env_scale / new_env_sin.sum())

            envmaps.append(new_env)

            light_intensity.append(new_env.view(3, -1).t())
            light_dir.append(self.sphvec.view(3, -1).t())

        envmaps = th.stack(envmaps, dim=0).float().to(device)
        envbg = th.stack(envbg, dim=0).float().to(device)
        light_intensity = th.stack(light_intensity, dim=0).float().to(device)
        light_dir = th.stack(light_dir, dim=0).float().to(device)

        data["preconv_envmap"] = self.mipmap(batch_size, device, 2.0 * np.pi * norm_scale[0])
        data["sigma_step"] = self.sigma_step
        data["envmap"] = envmaps
        data["lightrot"] = lightrots
        data["light_intensity"] = light_intensity
        data["light_pos"] = self.envmap_dist * light_dir
        data["envbg"] = envbg / 255.0
        data["light_type"] = "envmap"
        data["n_lights"] = light_intensity.shape[1] * th.ones(batch_size, 1).to(device)
        data["is_fullylit_frame"] = th.zeros(1).to(device)

        preds = self.mod(**data)
        preds["envbg"] = envbg
        return preds


class SingleLightCycleDecorator(th.nn.Module):
    def __init__(self, mod: th.nn.Module, light_positions: np.ndarray, cycle: int = 256) -> None:
        super().__init__()
        self.mod = mod

        self.light_path = get_light_interpoation_path(cycle, light_positions)
        self.cycle = len(self.light_path)

    def forward(self, **data: Dict[str, Any]) -> Dict[str, Any]:
        device = data["campos"].device
        batch_size = data["campos"].size(0)

        light_intensity = []
        light_pos = []

        for i in range(batch_size):
            index = data["index"][i] % self.cycle
            # trans = None
            # if "head_pose" in data:
            #     trans = data["head_pose"][i].data.cpu().numpy()[:3, 3]
            # elif "pose" in data:
            #     trans = data["pose"][i, :3].data.cpu().numpy()

            # angle = (abs(index) / self.cycle) * 2 * np.pi
            # if self.light_rotate_axis == 0:
            #     cur_lpos = np.asarray([0.0, 1100.0 * np.sin(angle), 1100.0 * np.cos(angle)]).astype(np.float32)
            # elif self.light_rotate_axis == 1:
            #     cur_lpos = np.asarray([-1.0 * 1100.0 * np.sin(angle), 300.0, 1100.0 * np.cos(angle)]).astype(np.float32)
            # else:
            #     cur_lpos = np.asarray([1100.0 * np.cos(angle), 1100.0 * np.sin(angle), 0.0]).astype(np.float32)

            # cur_lpos = 1100.0 * cur_lpos / np.linalg.norm(cur_lpos)
            # if trans is not None:
            #     cur_lpos += trans

            # cur_lpos = th.from_numpy(cur_lpos).to(device)
            cur_lpos = self.light_path[index] * 1000.0
            light_pos.append(th.as_tensor(cur_lpos, device=device, dtype=th.float32))

            light_intensity.append(th.Tensor([1.0]).to(device))

        light_intensity = th.stack(light_intensity, dim=0).float()
        data["light_intensity"] = light_intensity[..., None]

        light_pos = th.stack(light_pos, dim=0).float()
        data["light_pos"] = light_pos[:, None]
        data["n_lights"] = th.ones(batch_size, device=device).int()
        data["is_fullylit_frame"] = th.zeros(1).to(device)

        return self.mod(**data)


def get_light_interpoation_path(num_frames, light_positions):
    n_interp_between_steps = 20
    n_steps = num_frames // n_interp_between_steps
    num_frames = n_steps * n_interp_between_steps

    0 - 4
    4 - 9
    9 - 39
    39 - 34
    34 - 30
    30 - 0
    interpolation_weight = np.linspace(0, 1, num_frames // 8)

    light_path = []

    light_indices = [0, 3, 6, 9, 39, 36, 33, 30, 0]
    for i in range(8):
        light1 = light_indices[i]
        light2 = light_indices[i + 1]
        for j in range(num_frames // 8):
            light_pos = (
                light_positions[light1] * (1 - interpolation_weight[j])
                + light_positions[light2] * (interpolation_weight[j])
            )
            light_path.append(light_pos)

    light_path = np.stack(light_path, axis=0)

    return th.from_numpy(light_path)

# Copyright (c) Jonathan Schmidt, Inc. and affiliates.
# All rights reserved.

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pillow_avif  # noqa
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch3d.io import load_ply
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import pil_to_tensor

from ca_code.utils.image import srgb2linear

# There are a lot of frame-wise assets. Avoid re-fetching those when we
# switch cameras
CACHE_LENGTH = 16
NUM_SEQUENCES = 8

logger = logging.getLogger(__name__)


class BecomingLitDataset(Dataset):
    def __init__(
        self,
        root_path: Path,
        subject: str,
        sequence: str,
        downscale_factor: int = 2,
        fully_lit_only: bool = False,
        partially_lit_only: bool = False,
        cameras_subset: Optional[Iterable[str]] = None,
        frames_subset: Optional[Iterable[int]] = None,
        lights_subset: Optional[Iterable[int]] = None,
    ):
        self.root_path: Path = Path(root_path)
        self.subject: str = str(subject)
        self.sequence: str = sequence
        self.fully_lit_only: bool = fully_lit_only
        self.partially_lit_only: bool = partially_lit_only

        self.downscale_factor: int = downscale_factor
        self.downscale_suffix = f"_{downscale_factor}" if downscale_factor > 1 else ""

        self.cameras_subset = set(cameras_subset or {})
        self.cameras = list(self.get_camera_calibration().keys())

        self.frames_subset = set(frames_subset or {})
        self.frames_subset = set(map(int, self.frames_subset))

    @property
    def seq_folder(self) -> Path:
        return self.root_path / self.subject / self.sequence

    @property
    def subject_folder(self) -> Path:
        return self.root_path / self.subject

    @lru_cache(maxsize=NUM_SEQUENCES)
    def get_camera_calibration(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        with open(self.subject_folder / "camera_calibration.json") as f:
            camera_calibration = json.load(f)

        # Intrinsics (shared across cameras)
        intrinsic = camera_calibration["cam_data"]
        K = torch.tensor([[intrinsic["fx"], 0, intrinsic["cx"]], [0, intrinsic["fy"], intrinsic["cy"]], [0, 0, 1]])
        K[:2, :] /= self.downscale_factor  # Downscale intrinsics (pixel units)

        # World to camera matrices
        w2c = {cid: torch.tensor(w2c) for cid, w2c in camera_calibration["world_to_cam"].items()}
        logger.info(f"Found {len(w2c)} cameras in the calibration file")

        # NOTE: Camera extrinsics are in OpenCV convention, which fits the gsplat backend. No need for conversion.

        # Scale translation from meters to millimeters
        for cid, mtx in w2c.items():
            mtx[:3, 3] *= 1000

        if self.cameras_subset:
            w2c = {cid: w2c for cid, w2c in w2c.items() if cid in self.cameras_subset}
            logger.info(f"Left with {len(w2c)} cameras after filtering for passed camera subset")

        return {cid: (K, w2c) for cid, w2c in w2c.items()}

    @lru_cache(maxsize=NUM_SEQUENCES*CACHE_LENGTH)
    def get_camera_parameters(self, cam_id: str) -> Dict[str, Any]:
        K, w2c = self.get_camera_calibration()[cam_id]
        R = w2c[:3, :3]
        focal = K[:2, :2]
        princpt = K[:2, 2]

        return {
            "Rt": w2c,
            "K": K,
            "campos": torch.linalg.inv(w2c)[:3, 3],
            "camrot": R,
            "focal": focal,
            "princpt": princpt,
        }

    @lru_cache(maxsize=NUM_SEQUENCES)
    def get_camera_list(self) -> List[str]:
        return self.cameras

    def filter_frame_list(self, frame_list: List[int]):
        frames = set(frame_list)

        vert_files = [
            p for p in (self.seq_folder / "flame_tracking" / "vertices").iterdir() if p.is_file and p.suffix == ".ply"
        ]
        vert_frames = set([int(p.stem.split("_")[-1]) for p in vert_files])

        frames = frames.intersection(vert_frames)
        if self.frames_subset:
            frames = frames.intersection(self.frames_subset)

        return sorted(list(frames))

    @lru_cache(maxsize=2)
    def get_frame_list(self, fully_lit_only: bool = False, partially_lit_only: bool = False) -> List[int]:
        # fully lit only and partially lit only cannot be enabled at the same time
        assert not (fully_lit_only and partially_lit_only)

        # df = pd.read_csv(self.seq_folder / "light_pattern_per_frame.json")

        # if not (fully_lit_only or partially_lit_only):
        #     frame_list = df.frame_id.tolist()
        #     return self.filter_frame_list(frame_list)

        # if fully_lit_only:
        #     frame_list = df[df.light_pattern == 0].frame_id.tolist()
        #     return self.filter_frame_list(frame_list)
        # else:  # partially lit only
        #     frame_list = df[df.light_pattern != 0].frame_id.tolist()
        #     return self.filter_frame_list(frame_list)
        with open(self.seq_folder / "light_pattern_per_frame.json", mode="r") as f:
            light_pattern = json.load(f)

        if not (fully_lit_only or partially_lit_only):
            # frame_list = df.frame_id.tolist()
            frame_list = [int(fid) for (fid, _) in light_pattern]
            return self.filter_frame_list(frame_list)

        if fully_lit_only:
            # frame_list = df[df.light_pattern == 0].frame_id.tolist()
            frame_list = [int(fid) for (fid, l_idx) in light_pattern if l_idx == 0]
            return self.filter_frame_list(frame_list)
        else:  # partially lit only
            # frame_list = df[df.light_pattern != 0].frame_id.tolist()
            frame_list = [int(fid) for (fid, l_idx) in light_pattern if l_idx != 0]
            return self.filter_frame_list(frame_list)

    def load_image(self, frame_id: int, cam_id: str) -> Image.Image:
        img_path = self.seq_folder / f"img_cc{self.downscale_suffix}" / f"cam_{cam_id}" / f"frame_{frame_id:06d}.jpg"
        return pil_to_tensor(Image.open(img_path))

    def load_alpha(self, frame_id: int, cam_id: str) -> Image.Image:
        alpha_path = self.seq_folder / "flame_tracking" / "alpha_masks_birefnet" / f"cam_{cam_id}" / f"frame_{frame_id:06d}_union.jpg"
        return pil_to_tensor(Image.open(alpha_path))

    @lru_cache(maxsize=CACHE_LENGTH)
    def load_registration_vertices(self, frame: int) -> torch.Tensor:
        ply_path = self.seq_folder / "flame_tracking" / "vertices" / f"frame_{frame:06d}.ply"
        with open(ply_path, "rb") as f:
            verts, _ = load_ply(f)
        return verts

    @lru_cache(maxsize=NUM_SEQUENCES)
    def load_registration_vertices_mean(self) -> np.ndarray:
        mean_path = self.seq_folder / "flame_tracking" / "vertices" / "vertices_mean.npy"
        return np.load(mean_path)

    @lru_cache(maxsize=NUM_SEQUENCES)
    def load_registration_vertices_variance(self) -> float:
        verts_path = self.seq_folder / "flame_tracking" / "vertices" / "vertices_var.txt"
        with open(verts_path, "r") as f:
            return float(f.read())

    @lru_cache(maxsize=NUM_SEQUENCES)
    def load_color_mean(self) -> torch.Tensor:
        jpg_path = self.seq_folder / "flame_tracking" / "color_mean.jpg"
        color_mean = Image.open(jpg_path)
        color_mean = color_mean.resize((1024, 1024))
        return pil_to_tensor(color_mean)

    @lru_cache(maxsize=NUM_SEQUENCES)
    def load_color_variance(self) -> float:
        # color_var_path = self.seq_folder / "flame_tracking" / "color_variance.txt"
        # with open(color_var_path, "r") as f:
        #     return float(f.read())
        return 458.0  # TODO: Fix this

    @lru_cache(maxsize=NUM_SEQUENCES)
    def load_color(self, frame: int) -> Optional[torch.Tensor]:
        jpg_path = self.seq_folder / "flame_tracking" / "color_mean.jpg"
        color = Image.open(jpg_path)
        color = color.resize((1024, 1024))
        return pil_to_tensor(color)

    @lru_cache(maxsize=CACHE_LENGTH)
    def load_background(self, camera: str) -> torch.Tensor:
        png_path = self.subject_folder / "BACKGROUND" / f"cam_{camera}_cc.png"
        return pil_to_tensor(Image.open(png_path))

    @lru_cache(maxsize=NUM_SEQUENCES)
    def load_light_pattern(self) -> List[Tuple[int]]:
        light_pattern_path = self.seq_folder / "light_pattern_per_frame.json"
        with open(light_pattern_path, "r") as f:
            return json.load(f)

    @lru_cache(maxsize=NUM_SEQUENCES)
    def load_light_pattern_meta(self) -> Dict[str, Any]:
        light_pattern_path = self.subject_folder / "calibration" / "light_pattern_metadata.json"
        with open(light_pattern_path, "r") as f:
            return json.load(f)

    @lru_cache(maxsize=CACHE_LENGTH)
    def load_head_pose(self, frame: int) -> np.ndarray:
        pose_path = self.seq_folder / "flame_tracking" / "head_poses" / f"frame_{frame:06d}.npy"
        pose = np.load(pose_path)  # (4, 4)
        return pose.astype(np.float32)

    def batch_filter(self, batch):
        batch["image"] = batch["image"].float()
        batch["background"] = batch["background"].float()

        # NOTE: Optional black level subtraction goes here

        batch["image"] = srgb2linear((batch["image"] / 255.0)).clamp(0, 1)
        batch["background"] = srgb2linear((batch["background"] / 255.0)).clamp(0, 1)

    @property
    def static_assets(self) -> Dict[str, Any]:
        assets = self._static_get_fn()
        shared_assets = self.load_shared_assets()
        return {
            **shared_assets,
            **assets,
        }

    @lru_cache(maxsize=NUM_SEQUENCES)
    def load_shared_assets(self) -> Dict[str, Any]:
        topology = torch.load(self.root_path / "topology.pt")
        return {"topology": topology}

    def _static_get_fn(self) -> Dict[str, Any]:
        reg_verts_mean = self.load_registration_vertices_mean()
        reg_verts_var = self.load_registration_vertices_variance()
        light_pattern = self.load_light_pattern()
        light_pattern_meta = self.load_light_pattern_meta()
        color_mean = self.load_color_mean()
        color_var = self.load_color_variance()
        krt = self.get_camera_calibration()
        return {
            "camera_ids": list(krt.keys()),
            "verts_mean": reg_verts_mean,
            "verts_var": reg_verts_var,
            "color_mean": color_mean,
            "color_var": color_var,
            "light_pattern": light_pattern,
            "light_pattern_meta": light_pattern_meta,
        }

    def _get_fn(self, frame: int, camera: str) -> Dict[str, Any]:
        is_fully_lit_frame: bool = frame in self.get_frame_list(fully_lit_only=True)
        image = self.load_image(frame, camera).float() / 255.0
        alpha = self.load_alpha(frame, camera).float() / 255.0
        if image.size() != alpha.size():
            alpha = F.interpolate(alpha[None], size=(image.shape[1], image.shape[2]), mode="bilinear")[0]
        image = image * alpha
        image = (image * 255.0).clamp(0, 255).byte()

        head_pose = self.load_head_pose(frame)
        # kpts = self.load_3d_keypoints(frame)
        reg_verts = self.load_registration_vertices(frame)
        # reg_verts_mean = self.load_registration_vertices_mean()
        # reg_verts_var = self.load_registration_vertices_variance()
        # template_mesh = self.load_template_mesh()

        # TODO: precompute some of them
        light_pattern = self.load_light_pattern()
        light_pattern = {f[0]: f[1] for f in light_pattern}
        light_pattern_meta = self.load_light_pattern_meta()
        light_pos_all = torch.FloatTensor(light_pattern_meta["light_positions"]) * 1000 # meters to millimeters
        n_lights_all = light_pos_all.shape[0]
        lightinfo = torch.IntTensor(light_pattern_meta["light_pattern"][light_pattern[frame]]["light_index_durations"])
        n_lights = lightinfo.shape[0]
        light_pos = light_pos_all[lightinfo[:, 0]]
        light_intensity = lightinfo[:, 1:].float() / 5555.0
        light_pos = F.pad(light_pos, (0, 0, 0, n_lights_all - n_lights), "constant", 0)
        light_intensity = F.pad(light_intensity, (0, 0, 0, n_lights_all - n_lights), "constant", 0)

        # segmentation_parts = self.load_segmentation_parts(frame, camera)
        # c, h, w = segmentation_parts.shape
        # # import ipdb; ipdb.set_trace()
        # if h == 1024:
        #     with torch.no_grad():
        #         segmentation_parts = F.interpolate(segmentation_parts[None, :, :, :], scale_factor=2, mode='bilinear')[0]
        # segmentation_fgbg = segmentation_parts != 0.0
        # color_mean = self.load_color_mean()
        # color_var = self.load_color_variance()
        color = self.load_color(frame)
        # scan_mesh = self.load_scan_mesh(frame)
        # background = self.load_background(camera)[:3]
        # if image.size() != background.size():
        #     background = F.interpolate(background[None], size=(image.shape[1], image.shape[2]), mode="bilinear")[0]
        background = torch.zeros_like(image)

        camera_parameters = self.get_camera_parameters(camera)

        row = {
            "camera_id": camera,
            "frame_id": frame,
            "is_fully_lit_frame": is_fully_lit_frame,
            "head_pose": head_pose,
            "image": image,
            "registration_vertices": reg_verts,
            "light_pos": light_pos,
            "light_intensity": light_intensity,
            "n_lights": n_lights,
            "color": color,
            "background": background,
            # "keypoints_3d": kpts,
            # "registration_vertices_mean": reg_verts_mean,
            # "registration_vertices_variance": reg_verts_var,
            # "template_mesh": template_mesh,
            # "light_pattern": light_pattern,
            # "light_pattern_meta": light_pattern_meta,
            # "segmentation_parts": segmentation_parts,
            # "segmentation_fgbg": segmentation_fgbg,
            # "color_mean": color_mean,
            # "color_variance": color_var,
            # "scan_mesh": scan_mesh,
            **camera_parameters,
        }
        return row

    def get(self, frame: int, camera: str) -> Dict[str, Any]:
        sample = self._get_fn(frame, camera)
        missing_assets = [k for k, v in sample.items() if v is None]
        if len(missing_assets) != 0:
            logger.warning(
                f"sample was missing these assets: {missing_assets} with idx frame_id=`{frame}`, camera_id=`{camera}` {sample['n_lights']}"
            )
            return None
        else:
            return sample

    def __getitem__(self, idx):
        frame_list = self.get_frame_list(
            fully_lit_only=self.fully_lit_only,
            partially_lit_only=self.partially_lit_only,
        )

        camera_list = self.get_camera_list()

        frame = frame_list[idx // len(camera_list)]
        camera = camera_list[idx % len(camera_list)]

        data = self.get(frame, camera)
        return data

    def __len__(self):
        return len(
            self.get_frame_list(
                fully_lit_only=self.fully_lit_only,
                partially_lit_only=self.partially_lit_only,
            )
        ) * len(self.get_camera_list())


def worker_init_fn(worker_id: int):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)


def collate_fn(items):
    """Modified form of `torch.utils.data.dataloader.default_collate`
    that will strip samples from the batch if they are ``None``."""
    items = [item for item in items if item is not None]
    return default_collate(items) if len(items) > 0 else None


class MultiSequenceBecomingLitDataset(Dataset):
    def __init__(
        self,
        root_path: Path,
        subject: str,
        sequences: Iterable[str],
        **dataset_args,
    ):
        self.datasets = [
            BecomingLitDataset(root_path=root_path, subject=subject, sequence=seq, **dataset_args) for seq in sequences
        ]
        self.index_ranges = np.cumsum(np.array([len(dataset) for dataset in self.datasets]))

    @property
    def static_assets(self) -> Dict[str, Any]:
        seq_shared_assets = self.datasets[0]._static_get_fn()
        assets = [dataset._static_get_fn() for dataset in self.datasets]

        seq_assets = {
            "light_pattern": [lp for a in assets for lp in a["light_pattern"]],
            "camera_ids": seq_shared_assets["camera_ids"],
            "verts_mean": np.stack([a["verts_mean"] for a in assets], axis=0).mean(axis=0),
            "verts_var": np.max(np.stack([a["verts_var"] for a in assets], axis=0)).item(),
            "color_mean": torch.stack([a["color_mean"].float() for a in assets], dim=0).mean(dim=0).byte(),
            "color_var": np.max(np.array([a["color_var"] for a in assets])).item(),
            "light_pattern_meta": seq_shared_assets["light_pattern_meta"],
        }

        shared_assets = self.datasets[0].load_shared_assets()
        return {
            **shared_assets,
            **seq_assets,
        }

    def batch_filter(self, batch):
        return self.datasets[0].batch_filter(batch)

    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets])

    def __getitem__(self, index):
        dataset_idx = np.searchsorted(self.index_ranges, index, side="right")
        index_within_dataset = index - self.index_ranges[dataset_idx - 1] if dataset_idx > 0 else index
        return self.datasets[dataset_idx].__getitem__(index_within_dataset)


if __name__ == "__main__":
    import os

    import dotenv

    dotenv.load_dotenv()

    dataset_path = os.getenv("BECOMINGLIT_DATASET_PATH")
    print(f"Dataset path: {dataset_path}")
    dataset = BecomingLitDataset(
        root_path=Path(dataset_path),
        subject="1015",
        sequence="HEADROT",
        fully_lit_only=False,
    )

    sample = dataset.static_assets
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)
        elif isinstance(v, np.ndarray):
            print(k, v.shape)
        else:
            print(k, v)

    print("#####")
    multi_dataset = MultiSequenceBecomingLitDataset(
        root_path=Path(dataset_path), subject="1001", sequences=["EXP-1", "EXP-2", "EMOTIONS"]
    )

    print(len(multi_dataset))
    static_assets = multi_dataset[50000]

    for k, v in static_assets.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)
        elif isinstance(v, np.ndarray):
            print(k, v.shape)
        else:
            print(k, type(v))

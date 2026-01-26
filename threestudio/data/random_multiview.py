import math
import random
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

from threestudio import register
from threestudio.utils.base import Updateable
from threestudio.data.uncond import (
    RandomCameraDataModuleConfig,
    RandomCameraDataset,
    RandomCameraIterableDataset,
)
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_rank
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *

@dataclass
class RandomMultiviewCameraDataModuleConfig(RandomCameraDataModuleConfig):
    relative_radius: bool = True
    zoom_range: Tuple[float, float] = (1.0, 1.0)
    n_view: int = 1  # 添加 n_view 参数

class RandomMultiviewCameraIterableDataset(RandomCameraIterableDataset):
    def __init__(self, cfg: Any):
        super().__init__(cfg)
        self.zoom_range = cfg.zoom_range
        self.counter = 0  # 当前视角计数器 (0-3)
        self.current_base_params = None  # 存储基础视角参数

    def collate(self, batch) -> Dict[str, Any]:
        if self.counter == 0:
            # 生成新的基础视角参数
            self.current_base_params = self._sample_base_view()

        # 计算当前视角的旋转偏移
        offset_deg = 90 * self.counter
        data = self._generate_view(offset_deg)

        # 更新计数器
        self.counter = (self.counter + 1) % 4

        return data

    def _sample_base_view(self) -> Dict[str, Any]:
        """采样基础视角的所有参数"""
        real_batch_size = 1  # 每次处理单个视角

        # 采样基础参数
        elevation_deg, elevation, azimuth_deg, azimuth, fovy_deg, fovy, camera_distances = self._sample_view_params(
            real_batch_size)

        # 采样扰动参数
        camera_perturb = (
            torch.rand(real_batch_size, 3) * 2 * self.cfg.camera_perturb - self.cfg.camera_perturb
        )
        center_perturb = (
            torch.randn(real_batch_size, 3) * self.cfg.center_perturb
        )
        up_perturb = (
            torch.randn(real_batch_size, 3) * self.cfg.up_perturb
        )

        # 计算基础光照位置
        light_positions = self._sample_light_positions(
            real_batch_size,
            camera_distances,
            elevation,
            azimuth,
            camera_perturb
        )

        return {
            'elevation_deg': elevation_deg,
            'elevation': elevation,
            'azimuth_deg': azimuth_deg,
            'azimuth': azimuth,
            'fovy_deg': fovy_deg,
            'fovy': fovy,
            'camera_distances': camera_distances,
            'camera_perturb': camera_perturb,
            'center_perturb': center_perturb,
            'up_perturb': up_perturb,
            'light_positions': light_positions,
        }

    def _generate_view(self, offset_deg: int) -> Dict[str, Any]:
        """生成指定偏移的视角"""
        params = self.current_base_params
        real_batch_size = 1

        # 计算新的方位角
        new_azimuth_deg = (params['azimuth_deg'] + offset_deg) % 360
        new_azimuth = new_azimuth_deg * math.pi / 180

        # 计算相机位置 (应用基础扰动)
        camera_positions = torch.stack([
            params['camera_distances'] * torch.cos(params['elevation']) * torch.cos(new_azimuth),
            params['camera_distances'] * torch.cos(params['elevation']) * torch.sin(new_azimuth),
            params['camera_distances'] * torch.sin(params['elevation']),
        ], dim=-1) + params['camera_perturb']

        # 计算其他参数 (应用基础扰动)
        center = torch.zeros_like(camera_positions) + params['center_perturb']
        up = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None, :].repeat(real_batch_size, 1)
        up += params['up_perturb']

        # 计算视图矩阵
        lookat = F.normalize(center - camera_positions, dim=-1)
        right = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4 = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w = torch.cat([c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1)
        c2w[:, 3, 3] = 1.0

        # 计算光线
        focal_length = 0.5 * self.height / torch.tan(0.5 * params['fovy'])
        directions = self.directions_unit_focal[None, :, :, :].repeat(real_batch_size, 1, 1, 1)
        directions[:, :, :, :2] = directions[:, :, :, :2] / focal_length[:, None, None, None]

        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)
        proj_mtx = get_projection_matrix(params['fovy'], self.width / self.height, 0.1, 1000.0)
        mvp_mtx = get_mvp_matrix(c2w, proj_mtx)

        # return {
        #     "rays_o": rays_o.squeeze(0),
        #     "rays_d": rays_d.squeeze(0),
        #     "mvp_mtx": mvp_mtx.squeeze(0),
        #     "camera_positions": camera_positions.squeeze(0),
        #     "c2w": c2w.squeeze(0),
        #     "light_positions": params['light_positions'].squeeze(0),
        #     "elevation": params['elevation_deg'].squeeze(0),
        #     "azimuth": torch.tensor(new_azimuth_deg),
        #     "camera_distances": params['camera_distances'].squeeze(0),
        #     "height": self.height,
        #     "width": self.width,
        #     "fovy": params['fovy_deg'].squeeze(0),
        # }

        return {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            "c2w": c2w,
            "light_positions": params['light_positions'],
            "elevation": params['elevation_deg'],
            # "azimuth": torch.tensor(new_azimuth_deg),
            # 修正后代码（推荐方案1）
            "azimuth": new_azimuth_deg.clone().detach(),
            "camera_distances": params['camera_distances'],
            "height": self.height,
            "width": self.width,
            "fovy": params['fovy_deg'],
        }

    def _sample_view_params(self, batch_size):
        """采样视角基本参数"""
        # 仰角采样
        if random.random() < 0.5:
            elevation_deg = torch.rand(batch_size) * (self.cfg.elevation_range[1] - self.cfg.elevation_range[0]) + \
                            self.cfg.elevation_range[0]
            elevation = elevation_deg * math.pi / 180
        else:
            elevation_range_percent = [
                (self.cfg.elevation_range[0] + 90.0) / 180.0,
                (self.cfg.elevation_range[1] + 90.0) / 180.0,
            ]
            elevation = torch.asin(
                2 * (torch.rand(batch_size) * (elevation_range_percent[1] - elevation_range_percent[0])
                     + elevation_range_percent[0])
                - 1.0
            )
            elevation_deg = elevation / math.pi * 180.0

        # 方位角采样
        azimuth_deg = torch.rand(batch_size) * (self.cfg.azimuth_range[1] - self.cfg.azimuth_range[0]) + \
                      self.cfg.azimuth_range[0]
        azimuth = azimuth_deg * math.pi / 180

        # FOV采样
        fovy_deg = torch.rand(batch_size) * (self.cfg.fovy_range[1] - self.cfg.fovy_range[0]) + self.cfg.fovy_range[0]
        fovy = fovy_deg * math.pi / 180

        # 相机距离采样
        camera_distances = torch.rand(batch_size) * (
                self.cfg.camera_distance_range[1] - self.cfg.camera_distance_range[0]) + self.cfg.camera_distance_range[
                               0]
        if self.cfg.relative_radius:
            scale = 1 / torch.tan(0.5 * fovy)
            camera_distances = scale * camera_distances

        # 应用缩放
        zoom = torch.rand(batch_size) * (self.zoom_range[1] - self.zoom_range[0]) + self.zoom_range[0]
        fovy = fovy * zoom
        fovy_deg = fovy_deg * zoom

        return elevation_deg, elevation, azimuth_deg, azimuth, fovy_deg, fovy, camera_distances

    def _sample_light_positions(self, batch_size, camera_distances, elevation, azimuth, camera_perturb):
        """采样光照位置"""
        if self.cfg.light_sample_strategy == "dreamfusion":
            light_distances = (
                torch.rand(batch_size)
                * (self.cfg.light_distance_range[1] - self.cfg.light_distance_range[0])
                + self.cfg.light_distance_range[0]
            )
            light_direction = F.normalize(
                camera_distances * torch.stack([
                    torch.cos(elevation) * torch.cos(azimuth),
                    torch.cos(elevation) * torch.sin(azimuth),
                    torch.sin(elevation),
                ], dim=-1)
                + camera_perturb
                + torch.randn(batch_size, 3) * self.cfg.light_position_perturb,
                dim=-1,
            )
            return light_direction * light_distances[:, None]
        elif self.cfg.light_sample_strategy == "magic3d":
            # Magic3D光照采样逻辑
            pass
        else:
            raise ValueError(f"Unknown light strategy: {self.cfg.light_sample_strategy}")


@register("random-multiview-camera-datamodule")
class RandomMultiviewCameraDataModule(pl.LightningDataModule):
    cfg: RandomMultiviewCameraDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(RandomMultiviewCameraDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = RandomMultiviewCameraIterableDataset(self.cfg)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = RandomCameraDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = RandomCameraDataset(self.cfg, "test")


    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            # very important to disable multi-processing if you want to change self attributes at runtime!
            # (for example setting self.width and self.height in update_step)
            num_workers=0,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate
        )
        # return self.general_loader(self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

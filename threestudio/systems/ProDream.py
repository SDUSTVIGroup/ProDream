import os
from dataclasses import dataclass, field

import torch
import numpy as np
import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import cleanup, get_device, load_module_weights
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image

@threestudio.register("ProDream-system")
class DreamControl(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        # in ['coarse', 'geometry', 'texture']
        stage: str = "coarse"
        visualize_samples: bool = False

        geometry_type_c: str = ""
        geometry_c: dict = field(default_factory=dict)

        material_type_c: str = ""
        material_c: dict = field(default_factory=dict)

        background_type_c: str = ""
        background_c: dict = field(default_factory=dict)

        renderer_type_c: str = ""
        renderer_c: dict = field(default_factory=dict)

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()

        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()

        if self.cfg.geometry_c['shape_init'].startswith('mesh:'):
            background = threestudio.find(self.cfg.background_type_c)(self.cfg.background_c)
            material = threestudio.find(self.cfg.material_type_c)(self.cfg.material_c)
            geometry = threestudio.find(self.cfg.geometry_type_c)(self.cfg.geometry_c).to(get_device())
            geometry.initialize_shape()

            self.cond_render = threestudio.find(self.cfg.renderer_type_c)(self.cfg.renderer_c, geometry=geometry,
                                                                          material=material, background=background)
        else:
            ckpt_path = self.cfg.geometry_c['shape_init']
            background = threestudio.find(self.cfg.background_type_c)(self.cfg.background_c).to(get_device())
            material = threestudio.find(self.cfg.material_type_c)(self.cfg.material_c).to(get_device())
            from threestudio.utils.config import load_config
            prev_cfg = load_config(
                os.path.join(
                    os.path.dirname(ckpt_path),
                    "../configs/parsed.yaml",
                )
            )  # TODO: hard-coded relative path
            prev_geometry_cfg = prev_cfg.system.geometry
            # del prev_geometry_cfg['normal_type']

            # prev_geometry_cfg.update(self.cfg.geometry_convert_override)
            prev_geometry = threestudio.find(prev_cfg.system.geometry_type)(
                prev_geometry_cfg
            ).to(get_device())
            state_dict, epoch, global_step = load_module_weights(
                ckpt_path,
                module_name="geometry",
                map_location="cpu",
            )
            prev_geometry.load_state_dict(state_dict, strict=False)
            prev_geometry.do_update_step(epoch, global_step, on_load_weights=True)

            # for p in prev_geometry.density_network.parameters():
            #     p.requires_grad_(False)

            prev_renderer_cfg = prev_cfg.system.renderer
            self.cond_render = threestudio.find(prev_cfg.system.renderer_type)(
                prev_renderer_cfg, geometry=prev_geometry, material=material, background=background
            ).to(get_device())

            state_dict, epoch, global_step = load_module_weights(
                ckpt_path,
                module_name="renderer",
                map_location="cpu",
            )
            self.cond_render.load_state_dict(state_dict, strict=False)
            self.cond_render.do_update_step(epoch, global_step, on_load_weights=True)

        for p in self.cond_render.parameters():
            p.requires_grad_(False)


    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        cond_out = self.cond_render(**batch, render_rgb=False)
        if self.cfg.stage == "geometry":
            render_out = self.renderer(**batch)
        else:
            render_out = self.renderer(**batch)

        return {
            **render_out,
            'cond_opacity': cond_out['opacity'].repeat(1, 1, 1, 3), 'cond_rgb': cond_out['comp_rgb_fg']
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()

    def visualize_and_save(self, out, keys_to_visualize, save_dir):
        # 创建保存结果的目录
        os.makedirs(save_dir, exist_ok=True)

        for key in keys_to_visualize:
            if key in out and isinstance(out[key], torch.Tensor):
                # 将张量转换为 NumPy 数组并去除批量维度
                tensor = out[key]
                tensor_np = tensor.squeeze(0).cpu().detach().numpy()

                # 处理不同形状的张量
                if tensor_np.shape[-1] == 3:  # 对于 RGB 图像，形状为 (H, W, 3)
                    # 确保数据在 [0, 1] 范围内
                    tensor_np = np.clip(tensor_np, 0, 1)
                    # 将数据转换为 [0, 255] 范围的 uint8 类型
                    tensor_np = (tensor_np * 255).astype(np.uint8)
                elif tensor_np.shape[-1] == 1:  # 对于单通道图像，形状为 (H, W, 1)
                    # 去除最后一个维度
                    tensor_np = tensor_np.squeeze(-1)
                    # 将数据转换为 [0, 255] 范围的 uint8 类型
                    tensor_np = (tensor_np * 255).astype(np.uint8)
                else:
                    print(f"Unsupported tensor shape {tensor_np.shape} for key {key}. Skipping saving.")
                    continue

                # 使用 PIL 保存图像
                try:
                    img = Image.fromarray(tensor_np)
                    save_path = os.path.join(save_dir, f'{key}_step_{self.global_step}.png')
                    img.save(save_path)
                    print(f'Saved {key} to {save_path}')
                except Exception as e:
                    print(f"Error saving {key}: {e}")
            else:
                print(f"Key {key} not found in out or is not a valid tensor. Skipping.")

    def save_boundary_and_edge_masks(self, boundary_mask, edge_mask, save_dir, global_step=0):
        # 创建保存结果的目录
        os.makedirs(save_dir, exist_ok=True)

        def process_and_save_mask(mask, mask_name):
            mask_np = mask.squeeze(0).squeeze(-1).cpu().detach().numpy()
            mask_np = (mask_np * 255).astype(np.uint8)
            img = Image.fromarray(mask_np)
            save_path = os.path.join(save_dir, f'{mask_name}_step_{global_step}.png')
            img.save(save_path)
            print(f'Saved {mask_name} to {save_path}')

        process_and_save_mask(boundary_mask, 'boundary_mask')
        process_and_save_mask(edge_mask, 'edge_mask')

    def _compute_valid_mask(self, density: torch.Tensor) -> torch.Tensor:
        non_zero_density = density[density > 0.1]
        if len(non_zero_density) > 0:
            valid_threshold = torch.quantile(non_zero_density, 0.1)
        else:
            valid_threshold = 0
        return density > valid_threshold

    def training_step(self, batch, batch_idx):
        out = self(batch)
        if self.cfg.stage == "geometry":
            guidance_inp = out["comp_normal"]
            guidance_out = self.guidance(
                guidance_inp, out["comp_normal"], out["cond_rgb"], out["cond_opacity"], self.prompt_utils, **batch, rgb_as_latents=False,
                geo_cond=out["comp_rgb"]
            )

        else:
            guidance_inp = out["comp_rgb"]
            guidance_out = self.guidance(
                guidance_inp, out["depth"].repeat(1, 1, 1, 3), out["cond_rgb"], out["cond_opacity"], self.prompt_utils, **batch,
                rgb_as_latents=False,

            # guidance_inp=out["comp_rgb"]
            # guidance_out = self.guidance(
            #     guidance_inp, out["comp_normal"], out["cond_rgb"], out["cond_opacity"], self.prompt_utils, **batch,
            #     rgb_as_latents=False,

            ) if self.cfg.stage == 'coarse' else self.guidance(
                guidance_inp, out["comp_normal"], out["cond_rgb"], out["cond_opacity"], self.prompt_utils, **batch, rgb_as_latents=False
            )

        loss = 0.0

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        if self.cfg.stage == "coarse":
            if self.C(self.cfg.loss.lambda_orient) > 0:
                if "normal" not in out:
                    raise ValueError(
                        "Normal is required for orientation loss, no normal is found in the output."
                    )
                loss_orient = (
                                  out["weights"].detach()
                                  * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
                              ).sum() / (out["opacity"] > 0).sum()
                self.log("train/loss_orient", loss_orient)
                loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

            loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
            self.log("train/loss_sparsity", loss_sparsity)
            loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

            opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            self.log("train/loss_opaque", loss_opaque)
            loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

            # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
            # helps reduce floaters and produce solid geometry
            loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
            self.log("train/loss_z_variance", loss_z_variance)
            loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)
        elif self.cfg.stage == "geometry":
            loss_normal_consistency = out["mesh"].normal_consistency()
            self.log("train/loss_normal_consistency", loss_normal_consistency)
            loss += loss_normal_consistency * self.C(
                self.cfg.loss.lambda_normal_consistency
            )

            if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
                loss_laplacian_smoothness = out["mesh"].laplacian()
                self.log("train/loss_laplacian_smoothness", loss_laplacian_smoothness)
                loss += loss_laplacian_smoothness * self.C(
                    self.cfg.loss.lambda_laplacian_smoothness
                )
        elif self.cfg.stage == "texture":
            pass
        else:
            raise ValueError(f"Unknown stage {self.cfg.stage}")

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
        )

        if self.cfg.visualize_samples:
            self.save_image_grid(
                f"it{self.true_global_step}-{batch['index'][0]}-sample.png",
                [
                    {
                        "type": "rgb",
                        "img": self.guidance.sample(
                            out["cond_rgb"], out["cond_opacity"], self.prompt_utils, **batch, seed=self.global_step
                        )[0],
                        "kwargs": {"data_format": "HWC"},
                    },
                    {
                        "type": "rgb",
                        "img": self.guidance.sample_lora(out["comp_normal"], **batch)[0] if self.cfg.stage in [
                            "geometry", "texture"] else
                        self.guidance.sample_lora(out["depth"].repeat(1, 1, 1, 3), **batch)[0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ],
                name="validation_step_samples",
                step=self.true_global_step,
            )

    def on_validation_epoch_end(self):
        pass

    def test_step_new(self, batch, batch_idx):
        out = self(batch)
        out["comp_rgb_fg"] = out["comp_rgb_fg"] + (1.0 - out["opacity"]) * 1.0
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb_fg"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ],
            name="test_step",
            step=self.true_global_step,
        )

    def test_step(self, batch, batch_idx):
        out = self(batch)
        out["comp_rgb_fg"] = out["comp_rgb_fg"] + (1.0 - out["opacity"]) * 1.0
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb_fg"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )



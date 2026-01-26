import random
from contextlib import contextmanager
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    ControlNetModel,
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    StableDiffusionControlNetPipeline
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.utils.import_utils import is_xformers_available

import threestudio
from threestudio.models.control_lora import ControlLoRA
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseModule
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *
from controlnet_aux import HEDdetector, CannyDetector


import torch
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

class ToWeightsDType(nn.Module):
    def __init__(self, module: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return self.module(x).to(self.dtype)


@threestudio.register("fused-control-guidance")
class FusedControlGuidance(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
        pretrained_model_name_or_path_lora: str = "runwayml/stable-diffusion-v1-5"
        controlnet_name_or_path: str = "lllyasviel/control_v11p_sd15_scribble"
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 7.5
        condition_scale: float = 1.5
        guidance_scale_lora: float = 7.5
        lora_scale: float = 1.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True
        lora_cfg_training: bool = True
        lora_n_timestamp_samples: int = 1

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        view_dependent_prompting: bool = True
        camera_condition_type: str = "extrinsics"

        prompt_processor_type_lora: str = "stable-diffusion-prompt-processor"
        prompt_processor_lora: dict = field(default_factory=dict)

        control_type: str = "hed"

    cfg: Config

    def __init__(self, cfg, save_freq=200):
        super().__init__(cfg)  # 调用父类的 __init__ 方法，传递 cfg 参数
        self.image_save_counter = 0  # 初始化图像保存计数器
        self.save_freq = save_freq

    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }

        pipe_lora_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }

        @dataclass
        class SubModules:
            pipe: Union[StableDiffusionPipeline, StableDiffusionControlNetPipeline]
            pipe_lora: Union[StableDiffusionPipeline, StableDiffusionControlNetPipeline]

        controlnet = ControlNetModel.from_pretrained(
            self.cfg.controlnet_name_or_path,
            torch_dtype=self.weights_dtype,
        )
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            controlnet=controlnet,
            **pipe_kwargs,
        ).to(self.device)

        pipe_lora = StableDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path_lora,
            **pipe_lora_kwargs,
        ).to(self.device)
        prompt_processor_lora = threestudio.find(self.cfg.prompt_processor_type_lora)(self.cfg.prompt_processor_lora)
        self.prompt_utils_lora = prompt_processor_lora()
        n_ch = len(pipe_lora.unet.config.block_out_channels)
        control_ids = [i for i in range(n_ch)]
        cross_attention_dims = {i: [] for i in range(n_ch)}
        for name in pipe_lora.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else pipe_lora.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                control_id = control_ids[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                control_id = list(reversed(control_ids))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                control_id = control_ids[block_id]
            cross_attention_dims[control_id].append(cross_attention_dim)
        cross_attention_dims = tuple([cross_attention_dims[control_id] for control_id in control_ids])

        self.control_lora = ControlLoRA.from_config("./configs/control-lora.yaml")

        self.submodules = SubModules(pipe=pipe, pipe_lora=pipe_lora)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()
                self.pipe_lora.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()
            self.pipe_lora.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)
            self.pipe_lora.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)
            self.pipe_lora.unet.to(memory_format=torch.channels_last)

        del self.pipe.text_encoder
        del self.pipe_lora.text_encoder
        cleanup()

        if self.cfg.control_type == 'hed':
            self.preprocessor = HEDdetector.from_pretrained("lllyasviel/Annotators")
            self.preprocessor.netNetwork.to(self.device)
            self.preprocessor.netNetwork.eval()
            for p in self.preprocessor.netNetwork.parameters():
                p.requires_grad_(False)
        elif self.cfg.control_type == 'canny':
            self.preprocessor = CannyDetector()
        else:
            self.preprocessor = None

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.vae_lora.parameters():
            p.requires_grad_(False)
        for p in self.unet_lora.parameters():
            p.requires_grad_(False)
        for p in self.controlnet.parameters():
            p.requires_grad_(False)

        # Set correct lora layers
        lora_attn_procs = {}
        lora_layers_list = list([list(layer_list) for layer_list in self.control_lora.lora_layers])
        for name in self.unet_lora.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet_lora.config.cross_attention_dim
            if name.startswith("mid_block"):
                control_id = control_ids[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                control_id = list(reversed(control_ids))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                control_id = control_ids[block_id]

            lora_layers = lora_layers_list[control_id]
            if len(lora_layers) != 0:
                lora_layer = lora_layers.pop(0)
                lora_attn_procs[name] = lora_layer

        self.unet_lora.set_attn_processor(lora_attn_procs)

        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )
        self.scheduler.set_timesteps(20)

        self.scheduler_lora = DDIMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path_lora,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )

        self.scheduler_sample = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.scheduler_lora_sample = DPMSolverMultistepScheduler.from_config(
            self.pipe_lora.scheduler.config
        )

        self.pipe.scheduler = self.scheduler
        self.pipe_lora.scheduler = self.scheduler_lora

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value
        self.lora_scale = 0.5

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

        threestudio.info(f"Loaded Stable Diffusion!")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98, lora_scale=0.5, condition_scale=1.5):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
        if lora_scale is not None:
            self.lora_scale = lora_scale
        if condition_scale is not None:
            self.condition_scale = condition_scale

    @property
    def pipe(self):
        return self.submodules.pipe

    @property
    def pipe_lora(self):
        return self.submodules.pipe_lora

    @property
    def unet(self):
        return self.submodules.pipe.unet

    @property
    def unet_lora(self):
        return self.submodules.pipe_lora.unet

    @property
    def vae(self):
        return self.submodules.pipe.vae

    @property
    def vae_lora(self):
        return self.submodules.pipe_lora.vae

    @property
    def controlnet(self):
        return self.submodules.pipe.controlnet

    @torch.cuda.amp.autocast(enabled=False)
    def forward_controlnet(
        self,
        controlnet,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        image_cond: Float[Tensor, "..."],
        condition_scale: float,
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        return controlnet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            controlnet_cond=image_cond.to(self.weights_dtype),
            conditioning_scale=condition_scale,
            return_dict=False,
        )

    @torch.cuda.amp.autocast(enabled=False)
    def forward_control_unet(
        self,
        unet: UNet2DConditionModel,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        cross_attention_kwargs,
        down_block_additional_residuals,
        mid_block_additional_residual,
        class_labels: Optional[Float[Tensor, "B 16"]] = None
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            class_labels=class_labels,
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        unet: UNet2DConditionModel,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        image_cond: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        class_labels: Optional[Float[Tensor, "B 16"]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        _ = self.control_lora(image_cond).control_states
        return unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            class_labels=class_labels,
            cross_attention_kwargs=cross_attention_kwargs,
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 512 512"], geo_cond=None
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        geo_cond = geo_cond * 2.0 - 1.0 if geo_cond is not None else imgs
        posterior_lora = self.vae_lora.encode(geo_cond.to(self.weights_dtype)).latent_dist
        latents_lora = posterior_lora.sample() * self.vae_lora.config.scaling_factor
        return latents.to(input_dtype), latents_lora.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    @contextmanager
    def disable_unet_class_embedding(self, unet: UNet2DConditionModel):
        class_embedding = unet.class_embedding
        try:
            unet.class_embedding = None
            yield unet
        finally:
            unet.class_embedding = class_embedding

    def compute_grad_vsd(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        latents_lora: Float[Tensor, "B 4 64 64"],
        image_cond: Float[Tensor, "B 3 512 512"],
        image_cond_lora: Float[Tensor, "B 3 512 512"],
        text_embeddings_vd: Float[Tensor, "BB 77 768"],
        text_embeddings_lora: Float[Tensor, "BB 77 768"]
    ):
        B = latents.shape[0]

        with torch.no_grad():
            # random timestamp
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [B],
                dtype=torch.long,
                device=self.device,
            )
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            down_block_res_sample, mid_block_res_sample = self.forward_controlnet(
                self.controlnet,
                latent_model_input,
                torch.cat([t] * 2),
                image_cond, self.cfg.condition_scale, text_embeddings_vd
            )
            noise_pred_pretrain = self.forward_control_unet(
                self.unet,
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings_vd,
                cross_attention_kwargs=None,
                down_block_additional_residuals=down_block_res_sample,
                mid_block_additional_residual=mid_block_res_sample
            )

            latents_noisy_lora = self.scheduler_lora.add_noise(latents_lora, noise, t)
            latent_model_input_lora = torch.cat([latents_noisy_lora] * 2, dim=0)
            noise_pred_est = self.forward_unet(
                self.unet_lora,
                latent_model_input_lora,
                torch.cat([t] * 2),
                image_cond_lora,
                encoder_hidden_states=text_embeddings_lora,
                class_labels=None
            )

        (
            noise_pred_pretrain_text,
            noise_pred_pretrain_uncond,
        ) = noise_pred_pretrain.chunk(2)

        # NOTE: guidance scale definition here is aligned with diffusers, but different from other guidance
        noise_pred_pretrain = noise_pred_pretrain_uncond + self.cfg.guidance_scale * (
            noise_pred_pretrain_text - noise_pred_pretrain_uncond
        )

        # TODO: more general cases
        assert self.scheduler.config.prediction_type == "epsilon"
        if self.scheduler_lora.config.prediction_type == "v_prediction":
            alphas_cumprod = self.scheduler_lora.alphas_cumprod.to(
                device=latents_noisy.device, dtype=latents_noisy.dtype
            )
            alpha_t = alphas_cumprod[t] ** 0.5
            sigma_t = (1 - alphas_cumprod[t]) ** 0.5

            noise_pred_est = latent_model_input * torch.cat([sigma_t] * 2, dim=0).view(
                -1, 1, 1, 1
            ) + noise_pred_est * torch.cat([alpha_t] * 2, dim=0).view(-1, 1, 1, 1)

        (
            noise_pred_est_text,
            noise_pred_est_uncond,
        ) = noise_pred_est.chunk(2)

        # NOTE: guidance scale definition here is aligned with diffusers, but different from other guidance
        noise_pred_est = noise_pred_est_uncond + self.cfg.guidance_scale_lora * (
            noise_pred_est_text - noise_pred_est_uncond
        )

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        grad = w * ((noise_pred_pretrain - noise) - self.lora_scale * (noise_pred_est - noise))
        return grad

    def train_lora(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        image_cond: Float[Tensor, "B 3 512 512"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        camera_condition: Float[Tensor, "B 4 4"],
    ):
        B = latents.shape[0]
        latents = latents.detach().repeat(self.cfg.lora_n_timestamp_samples, 1, 1, 1)

        t = torch.randint(
            int(self.num_train_timesteps * 0.0),
            int(self.num_train_timesteps * 1.0),
            [B * self.cfg.lora_n_timestamp_samples],
            dtype=torch.long,
            device=self.device,
        )

        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler_lora.add_noise(latents, noise, t)
        if self.scheduler_lora.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler_lora.config.prediction_type == "v_prediction":
            target = self.scheduler_lora.get_velocity(latents, noise, t)
        else:
            raise ValueError(
                f"Unknown prediction type {self.scheduler_lora.config.prediction_type}"
            )

        text_embeddings, _ = text_embeddings.chunk(2)
        if self.cfg.lora_cfg_training and random.random() < 0.1:
            camera_condition = torch.zeros_like(camera_condition)

        noise_pred = self.forward_unet(
            self.unet_lora,
            noisy_latents,
            t,
            image_cond,
            encoder_hidden_states=text_embeddings.repeat(
                self.cfg.lora_n_timestamp_samples, 1, 1
            ),
            class_labels=None,
        )
        return F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

    def get_latents(
        self, rgb_BCHW: Float[Tensor, "B C H W"], rgb_as_latents=False, geo_cond=None
    ) -> Float[Tensor, "B 4 64 64"]:
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512, geo_cond)
        return latents

    def prepare_image_cond_original(self, cond_rgb, cond_opacity):
        if self.preprocessor is None:
            cond_rgb = cond_rgb.permute(0, 3, 1, 2).detach()
            return F.interpolate(cond_rgb, (512, 512), mode='bilinear', align_corners=False)

        cond_opacity = (cond_opacity[0].detach().cpu().numpy() * 255).astype(np.uint8).copy()
        if self.cfg.control_type == 'canny':
            cond_opacity = cv2.blur(cond_opacity, ksize=(5, 5))
        detected_map_opacity = self.preprocessor(cond_opacity)
        # 在2000轮后才处理和合并RGB轮廓
        if self.image_save_counter >= 50:
            # 处理RGB轮廓
            cond_rgb = (cond_rgb[0].detach().cpu().numpy() * 255).astype(np.uint8).copy()
            if self.cfg.control_type == 'canny':
                cond_rgb = cv2.blur(cond_rgb, ksize=(5, 5))
            detected_map_rgb = self.preprocessor(cond_rgb)
            # 合并轮廓
            combined_map = np.maximum(detected_map_rgb, detected_map_opacity)
        else:
            # 2000轮之前只使用opacity轮廓
            combined_map = detected_map_opacity
        # cond_rgb = (cond_rgb[0].detach().cpu().numpy() * 255).astype(np.uint8).copy()
        # if self.cfg.control_type == 'canny':
        #     cond_rgb = cv2.blur(cond_rgb, ksize=(5, 5))
        # detected_map_rgb = self.preprocessor(cond_rgb)
        # cond_opacity = (cond_opacity[0].detach().cpu().numpy() * 255).astype(np.uint8).copy()
        # if self.cfg.control_type == 'canny':
        #     cond_opacity = cv2.blur(cond_opacity, ksize=(5, 5))
        # detected_map_opacity = self.preprocessor(cond_opacity)
        # # 合并轮廓
        # combined_map = np.maximum(detected_map_rgb, detected_map_opacity)
        control = torch.from_numpy(np.array(combined_map)).float().to(self.device) / 255.
        control = control.unsqueeze(0)
        control = control.permute(0, 3, 1, 2)
        return F.interpolate(control, (512, 512), mode='bilinear', align_corners=False)

    def prepare_image_cond(self, cond_rgb, cond_opacity, current_step=None):
        if self.preprocessor is None:
            cond_rgb = cond_rgb.permute(0, 3, 1, 2).detach()
            return F.interpolate(cond_rgb, (512, 512), mode='bilinear', align_corners=False)

        # 1. 从不透明度图提取粗轮廓（捕获全局形状结构）
        cond_opacity = (cond_opacity[0].detach().cpu().numpy() * 255).astype(np.uint8).copy()
        if self.cfg.control_type == 'canny':
            cond_opacity = cv2.blur(cond_opacity, ksize=(5, 5))
        coarse_contour = self.preprocessor(cond_opacity)

        # 转换预处理器输出为NumPy数组（处理PIL Image对象）
        if hasattr(coarse_contour, 'convert'):  # 检测是否为PIL Image
            coarse_contour = np.array(coarse_contour.convert('L'))
        elif isinstance(coarse_contour, torch.Tensor):
            coarse_contour = coarse_contour.cpu().numpy()
        elif not isinstance(coarse_contour, np.ndarray):
            coarse_contour = np.array(coarse_contour)

        # 2. 获取当前训练迭代步数 - 安全访问模式
        if current_step is None:
            # 尝试从self属性中查找可能的步数（兼容不同调用环境）
            current_iter = getattr(self, 'global_step',
                                   getattr(self, 'step',
                                           getattr(self, 'iteration', 0)))
        else:
            current_iter = current_step

        # 3. 从配置中获取渐进式引导的关键参数（带默认值）
        delta_I = getattr(self.cfg, 'delta_I', 2000)  # 开始细轮廓引导的迭代阈值
        T_ramp = getattr(self.cfg, 'T_ramp', 1000)  # 完全过渡所需的迭代步数

        # 4. 计算混合系数α(t) - 使用线性斜坡函数
        if current_iter <= delta_I:
            alpha = 0.0  # 仅使用粗轮廓
        else:
            alpha = min(1.0, (current_iter - delta_I) / T_ramp)  # 线性渐进到1.0

        # 5. 渐进式轮廓融合
        if alpha > 0:
            # 从RGB提取细轮廓（捕获细节特征）
            cond_rgb = (cond_rgb[0].detach().cpu().numpy() * 255).astype(np.uint8).copy()
            if self.cfg.control_type == 'canny':
                cond_rgb = cv2.blur(cond_rgb, ksize=(5, 5))
            fine_contour = self.preprocessor(cond_rgb)

            # 同样转换fine_contour为NumPy数组
            if hasattr(fine_contour, 'convert'):  # 检测是否为PIL Image
                fine_contour = np.array(fine_contour.convert('L'))
            elif isinstance(fine_contour, torch.Tensor):
                fine_contour = fine_contour.cpu().numpy()
            elif not isinstance(fine_contour, np.ndarray):
                fine_contour = np.array(fine_contour)

            # 确保维度兼容性
            if coarse_contour.shape[:2] != fine_contour.shape[:2]:
                fine_contour = cv2.resize(fine_contour, (coarse_contour.shape[1], coarse_contour.shape[0]))

            # 确保单通道处理
            if coarse_contour.ndim > 2:
                coarse_contour = coarse_contour[..., 0] if coarse_contour.shape[-1] > 1 else coarse_contour.squeeze()
            if fine_contour.ndim > 2:
                fine_contour = fine_contour[..., 0] if fine_contour.shape[-1] > 1 else fine_contour.squeeze()

            # 线性混合：α(t)·x_fine^c + (1-α(t))·x_coarse^c
            combined_map = alpha * fine_contour.astype(np.float32) + (1 - alpha) * coarse_contour.astype(np.float32)
        else:
            # 优化早期阶段：仅使用粗轮廓确保稳定性
            combined_map = coarse_contour.astype(np.float32)

        # 6. 转换为张量并标准化
        if not isinstance(combined_map, np.ndarray):
            combined_map = np.array(combined_map)

        # 确保3通道格式
        if combined_map.ndim == 2:
            combined_map = combined_map[..., np.newaxis]  # 添加通道维度

        if combined_map.shape[-1] == 1:
            combined_map = np.repeat(combined_map, 3, axis=-1)

        # 转换为张量
        combined_map = torch.from_numpy(combined_map).float().to(cond_rgb.device)

        # 调整维度顺序 [B, C, H, W] 并归一化
        if combined_map.dim() == 3:  # [H, W, C]
            combined_map = combined_map.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        elif combined_map.dim() == 2:  # [H, W]
            combined_map = combined_map.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            combined_map = combined_map.repeat(1, 3, 1, 1)  # [1, 3, H, W]

        if combined_map.max() > 1.0:
            combined_map = combined_map / 255.0

        # 调整到扩散模型输入尺寸
        combined_map = F.interpolate(combined_map, (512, 512), mode='bilinear', align_corners=False)

        return combined_map


    def prepare_image_cond_progressive(self, cond_rgb, cond_opacity):
        """
        实现渐进式轮廓引导策略 (Progressive Contour Guidance)
        """
        if self.preprocessor is None:
            cond_rgb = cond_rgb.permute(0, 3, 1, 2).detach()
            return F.interpolate(cond_rgb, (512, 512), mode='bilinear', align_corners=False)
        # 1. 定义超参数
        delta_I = getattr(self.cfg, 'delta_I', 4000)
        # print(delta_I)
        T_ramp = getattr(self.cfg, 'T_ramp', 1)
        # T_ramp_safe = T_ramp if T_ramp != 0 else 1  # 避免除零
        t = self.image_save_counter
        # 2. 图像预处理 (转为 numpy)
        opacity_np = (cond_opacity[0].detach().cpu().numpy() * 255).astype(np.uint8).copy()
        rgb_np = (cond_rgb[0].detach().cpu().numpy() * 255).astype(np.uint8).copy()
        if self.cfg.control_type == 'canny':
            opacity_np = cv2.blur(opacity_np, ksize=(5, 5))
            rgb_np = cv2.blur(rgb_np, ksize=(5, 5))
        # 3. 提取轮廓并确保转换为 NumPy 数组 (修复点在这里)
        detected_map_opacity = self.preprocessor(opacity_np)
        # 使用 np.array() 转换 PIL Image 到 numpy 数组
        x_coarse = np.array(detected_map_opacity).astype(np.float32)
        detected_map_rgb = self.preprocessor(rgb_np)
        # 使用 np.array() 转换并与 opacity 轮廓合并
        x_fine_rgb = np.array(detected_map_rgb).astype(np.float32)
        x_fine = np.maximum(x_fine_rgb, x_coarse)
        # 4. 计算线性混合系数 alpha(t)
        if t <= delta_I:
            alpha = 0.0
        else:
            alpha = min(1.0, (t - delta_I) / T_ramp)
        # 5. 执行渐进式融合: x_hat = alpha * x_fine + (1 - alpha) * x_coarse
        combined_map = alpha * x_fine + (1.0 - alpha) * x_coarse
        combined_map = combined_map.astype(np.uint8)
        # 6. 转回 Tensor 并调整尺寸
        control = torch.from_numpy(combined_map).float().to(self.device) / 255.
        # 检查维度，如果是 Canny 等生成的 HxW，需要变为 1xCxHxW
        if len(control.shape) == 2:
            control = control.unsqueeze(-1)  # 变成 (H, W, 1)
        control = control.unsqueeze(0).permute(0, 3, 1, 2)  # 变成 (1, 1, H, W)
        return F.interpolate(control, (512, 512), mode='bilinear', align_corners=False)


    def prepare_image_cond_o(self, cond_rgb, cond_opacity):
        if self.preprocessor is None:
            cond_rgb = cond_rgb.permute(0, 3, 1, 2).detach()
            return F.interpolate(cond_rgb, (512, 512), mode='bilinear', align_corners=False)
        # 1. 从不透明度图提取粗轮廓（捕获全局形状结构）
        cond_opacity = (cond_opacity[0].detach().cpu().numpy() * 255).astype(np.uint8).copy()
        if self.cfg.control_type == 'canny':
            cond_opacity = cv2.blur(cond_opacity, ksize=(5, 5))
        coarse_contour = self.preprocessor(cond_opacity)
        # 2. 获取当前训练迭代步数（更准确的进度指标）
        current_iter = self.global_step
        # 3. 从配置中获取渐进式引导的关键参数（带默认值）
        delta_I = getattr(self.cfg, 'delta_I', 4000)  # 开始细轮廓引导的迭代阈值
        T_ramp = getattr(self.cfg, 'T_ramp', 1000)  # 完全过渡所需的迭代步数
        # 4. 计算混合系数α(t) - 使用线性斜坡函数
        if current_iter <= delta_I:
            alpha = 0.0  # 仅使用粗轮廓
        else:
            alpha = min(1.0, (current_iter - delta_I) / T_ramp)  # 线性渐进到1.0
        # 5. 渐进式轮廓融合（核心改进）
        if alpha > 0:
            # 从RGB提取细轮廓（捕获细节特征）
            cond_rgb = (cond_rgb[0].detach().cpu().numpy() * 255).astype(np.uint8).copy()
            if self.cfg.control_type == 'canny':
                cond_rgb = cv2.blur(cond_rgb, ksize=(5, 5))
            fine_contour = self.preprocessor(cond_rgb)
            # 确保维度兼容性
            if coarse_contour.shape[:2] != fine_contour.shape[:2]:
                fine_contour = cv2.resize(fine_contour, (coarse_contour.shape[1], coarse_contour.shape[0]))
            # 线性混合：α(t)·x_fine^c + (1-α(t))·x_coarse^c
            combined_map = alpha * fine_contour + (1 - alpha) * coarse_contour
        else:
            # 优化早期阶段：仅使用粗轮廓确保稳定性
            combined_map = coarse_contour
        # 6. 转换为张量并标准化（保持与原始代码兼容）
        if isinstance(combined_map, np.ndarray):
            combined_map = torch.from_numpy(combined_map).float()
        # 确保3通道格式
        if combined_map.dim() == 2:
            combined_map = combined_map.unsqueeze(-1)
        if combined_map.shape[-1] == 1:
            combined_map = combined_map.repeat(1, 1, 3)
        # 调整维度顺序 [B, C, H, W] 并归一化
        combined_map = combined_map.permute(2, 0, 1).unsqueeze(0)
        if combined_map.max() > 1.0:
            combined_map = combined_map / 255.0
        # 调整到扩散模型输入尺寸
        combined_map = F.interpolate(combined_map, (512, 512), mode='bilinear', align_corners=False)
        return combined_map


    def visualize_and_save(self, tensor, save_path, filename):
        # 确保保存路径存在
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 将 Tensor 从 GPU 移动到 CPU 并转换为 NumPy 数组
        tensor = tensor.cpu().detach().squeeze(0)  # 去除批量维度
        array = tensor.permute(1, 2, 0).numpy()  # 调整维度顺序为 (H, W, C)
        # 直接将 0 - 1 范围的数据乘以 255 并转换为 uint8
        array = (array * 255).astype(np.uint8)
        # 转换为 PIL 图像对象
        image = Image.fromarray(array)
        # 保存图像
        full_path = os.path.join(save_path, filename)
        image.save(full_path)
        print(f"Image saved to {full_path}")

    def forward(
        self,
        rgb: Float[Tensor, "B H W C"],
        normal: Float[Tensor, "B H W C"],
        cond_rgb: Float[Tensor, "B H W C"],
        cond_opacity: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        mvp_mtx: Float[Tensor, "B 4 4"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents=False,
        geo_cond=None,
        **kwargs,
    ):
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        if geo_cond is not None:
            geo_cond = geo_cond.permute(0, 3, 1, 2)
        latents, latents_lora = self.get_latents(rgb_BCHW, rgb_as_latents=rgb_as_latents, geo_cond=geo_cond)
        image_cond = self.prepare_image_cond_progressive(cond_rgb, cond_opacity)
        # image_cond = self.prepare_image_cond(cond_rgb, cond_opacity)
        normal_cond = F.interpolate(normal.permute(0, 3, 1, 2), (512, 512), mode='bilinear', align_corners=False)


        if self.image_save_counter % self.save_freq == 0:

            image_cond_filename = f"image_cond_{self.image_save_counter}.png"
            normal_cond_filename = f"normal_cond_{self.image_save_counter}.png"

            # self.visualize_and_save(image_cond, save_path, image_cond_filename)
            # self.visualize_and_save(normal_cond, save_path, normal_cond_filename)
            # self.image_save_counter += 1  # 保存图像后递增计数器
        self.image_save_counter += 1  # 保存图像后递增计数器
        # view-dependent text embeddings
        text_embeddings_vd = prompt_utils.get_text_embeddings(
            elevation,
            azimuth,
            camera_distances,
            view_dependent_prompting=self.cfg.view_dependent_prompting,
        )

        if self.cfg.camera_condition_type == "extrinsics":
            camera_condition = c2w
        elif self.cfg.camera_condition_type == "mvp":
            camera_condition = mvp_mtx
        else:
            raise ValueError(
                f"Unknown camera_condition_type {self.cfg.camera_condition_type}"
            )

        grad = self.compute_grad_vsd(
            latents, latents_lora, image_cond, normal_cond, text_embeddings_vd, text_embeddings_vd
        )

        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # reparameterization trick
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        target = (latents - grad).detach()
        loss_vsd = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        loss_lora = self.train_lora(latents_lora, normal_cond, text_embeddings_vd, camera_condition)

        return {
            "loss_vsd": loss_vsd,
            "loss_lora": loss_lora,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
            lora_scale=C(self.cfg.lora_scale, epoch, global_step),
            condition_scale=C(self.cfg.condition_scale, epoch, global_step)
        )

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def _sample(
        self,
        pipe: StableDiffusionPipeline,
        image_cond: Float[Tensor, "B 3 512 512"],
        sample_scheduler: DPMSolverMultistepScheduler,
        text_embeddings: Float[Tensor, "BB N Nf"],
        num_inference_steps: int,
        guidance_scale: float,
        num_images_per_prompt: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        class_labels: Optional[Float[Tensor, "BB 16"]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> Float[Tensor, "B H W 3"]:
        vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
        height = height or pipe.unet.config.sample_size * vae_scale_factor
        width = width or pipe.unet.config.sample_size * vae_scale_factor
        batch_size = text_embeddings.shape[0] // 2
        device = self.device

        sample_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = sample_scheduler.timesteps
        num_channels_latents = pipe.unet.config.in_channels

        latents = pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            self.weights_dtype,
            device,
            generator,
        )

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = sample_scheduler.scale_model_input(
                latent_model_input, t
            )

            # predict the noise residual
            if class_labels is None:
                down_block_res_sample, mid_block_res_sample = self.forward_controlnet(
                    pipe.controlnet, latent_model_input, t, image_cond, self.cfg.condition_scale, text_embeddings.to(self.weights_dtype)
                )
                # with self.disable_unet_class_embedding(pipe.unet) as unet:
                noise_pred = pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings.to(self.weights_dtype),
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_sample,
                    mid_block_additional_residual=mid_block_res_sample
                ).sample
            else:
                _ = self.control_lora(image_cond).control_states
                noise_pred = pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings.to(self.weights_dtype),
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = sample_scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1 / pipe.vae.config.scaling_factor * latents
        images = pipe.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        images = images.permute(0, 2, 3, 1).float()
        return images

    def sample(
        self,
        cond_rgb: Float[Tensor, "B H W C"],
        cond_opacity: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        seed: int = 0,
        **kwargs,
    ) -> Float[Tensor, "N H W 3"]:
        # view-dependent text embeddings
        text_embeddings_vd = prompt_utils.get_text_embeddings(
            elevation,
            azimuth,
            camera_distances,
            view_dependent_prompting=self.cfg.view_dependent_prompting,
        )

        cross_attention_kwargs = None
        image_cond = self.prepare_image_cond(cond_rgb, cond_opacity)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        return self._sample(
            pipe=self.pipe,
            image_cond=image_cond,
            sample_scheduler=self.scheduler_sample,
            text_embeddings=text_embeddings_vd,
            num_inference_steps=25,
            guidance_scale=self.cfg.guidance_scale,
            height=512,
            width=512,
            cross_attention_kwargs=cross_attention_kwargs,
            generator=generator,
            class_labels=None
        )

    def sample_lora(
        self,
        normal: Float[Tensor, "B H W C"],
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        mvp_mtx: Float[Tensor, "B 4 4"],
        c2w: Float[Tensor, "B 4 4"],
        seed: int = 0,
        **kwargs,
    ) -> Float[Tensor, "N H W 3"]:
        # input text embeddings, view-independent
        text_embeddings = self.prompt_utils_lora.get_text_embeddings(
            elevation, azimuth, camera_distances, view_dependent_prompting=self.cfg.view_dependent_prompting
        )

        normal_cond = F.interpolate(normal.permute(0, 3, 1, 2), (512, 512), mode='bilinear', align_corners=False)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        return self._sample(
            sample_scheduler=self.scheduler_lora_sample,
            image_cond=normal_cond,
            pipe=self.pipe_lora,
            text_embeddings=text_embeddings,
            num_inference_steps=25,
            guidance_scale=self.cfg.guidance_scale_lora,
            height=512,
            width=512,
            class_labels="lora",
            cross_attention_kwargs=None,
            generator=generator,
        )

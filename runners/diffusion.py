import os
import logging
import time
import glob
#from tkinter import E

import blobfile as bf

import numpy as np
import tqdm
import torch
import torch.utils.data as data
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from models.diffusion import Model
from models.improved_ddpm.unet import UNetModel as ImprovedDDPM_Model
from models.guided_diffusion.unet import UNetModel as GuidedDiffusion_Model
from models.guided_diffusion.unet import EncoderUNetModel as GuidedDiffusion_Classifier
from models.guided_diffusion.unet import SuperResModel as GuidedDiffusion_SRModel
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path
from evaluate.fid_score import calculate_fid_given_paths

import torchvision.utils as tvu


def load_data_for_worker(base_samples, batch_size, cond_class):
    with bf.BlobFile(base_samples, "rb") as f:
        obj = np.load(f)
        image_arr = obj["arr_0"]
        if cond_class:
            label_arr = obj["arr_1"]
    buffer = []
    label_buffer = []
    while True:
        for i in range(len(image_arr)):
            buffer.append(image_arr[i])
            if cond_class:
                label_buffer.append(label_arr[i])
            if len(buffer) == batch_size:
                batch = torch.from_numpy(np.stack(buffer)).float()
                batch = batch / 127.5 - 1.0
                batch = batch.permute(0, 3, 1, 2)
                res = dict(low_res=batch)
                if cond_class:
                    res["y"] = torch.from_numpy(np.stack(label_buffer))
                yield res
                buffer, label_buffer = [], []


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == 'cosine':
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2,
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, rank=None):
        self.args = args
        self.config = config
        if rank is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            device = rank
            self.rank = rank
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        model = Model(config)

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = loss_registry[config.model.type](model, x, t, e, b)

                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()

    def prepare_model(self):
        """
        모델(UNet 등)을 생성하고, ckpt_dir(체크포인트)이 있으면 로드한 뒤,
        self.model에 할당해 주는 메서드.
        """
        # (1) 모델 생성
        if self.config.model.model_type == 'improved_ddpm':
            model = ImprovedDDPM_Model(
                in_channels=self.config.model.in_channels,
                model_channels=self.config.model.model_channels,
                out_channels=self.config.model.out_channels,
                num_res_blocks=self.config.model.num_res_blocks,
                attention_resolutions=self.config.model.attention_resolutions,
                dropout=self.config.model.dropout,
                channel_mult=self.config.model.channel_mult,
                conv_resample=self.config.model.conv_resample,
                dims=self.config.model.dims,
                use_checkpoint=self.config.model.use_checkpoint,
                num_heads=self.config.model.num_heads,
                num_heads_upsample=self.config.model.num_heads_upsample,
                use_scale_shift_norm=self.config.model.use_scale_shift_norm
            )
        elif self.config.model.model_type == 'guided_diffusion':
            if self.config.model.is_upsampling:
                model = GuidedDiffusion_SRModel(
                    image_size=self.config.model.large_size,
                    in_channels=self.config.model.in_channels,
                    model_channels=self.config.model.model_channels,
                    out_channels=self.config.model.out_channels,
                    num_res_blocks=self.config.model.num_res_blocks,
                    attention_resolutions=self.config.model.attention_resolutions,
                    dropout=self.config.model.dropout,
                    channel_mult=self.config.model.channel_mult,
                    conv_resample=self.config.model.conv_resample,
                    dims=self.config.model.dims,
                    num_classes=self.config.model.num_classes,
                    use_checkpoint=self.config.model.use_checkpoint,
                    use_fp16=self.config.model.use_fp16,
                    num_heads=self.config.model.num_heads,
                    num_head_channels=self.config.model.num_head_channels,
                    num_heads_upsample=self.config.model.num_heads_upsample,
                    use_scale_shift_norm=self.config.model.use_scale_shift_norm,
                    resblock_updown=self.config.model.resblock_updown,
                    use_new_attention_order=self.config.model.use_new_attention_order,
                )
            else:
                model = GuidedDiffusion_Model(
                    image_size=self.config.model.image_size,
                    in_channels=self.config.model.in_channels,
                    model_channels=self.config.model.model_channels,
                    out_channels=self.config.model.out_channels,
                    num_res_blocks=self.config.model.num_res_blocks,
                    attention_resolutions=self.config.model.attention_resolutions,
                    dropout=self.config.model.dropout,
                    channel_mult=self.config.model.channel_mult,
                    conv_resample=self.config.model.conv_resample,
                    dims=self.config.model.dims,
                    num_classes=self.config.model.num_classes,
                    use_checkpoint=self.config.model.use_checkpoint,
                    use_fp16=self.config.model.use_fp16,
                    num_heads=self.config.model.num_heads,
                    num_head_channels=self.config.model.num_head_channels,
                    num_heads_upsample=self.config.model.num_heads_upsample,
                    use_scale_shift_norm=self.config.model.use_scale_shift_norm,
                    resblock_updown=self.config.model.resblock_updown,
                    use_new_attention_order=self.config.model.use_new_attention_order,
                )
        else:
            # 그 외 기본 모델 (Model) 사용
            model = Model(self.config)

        model = model.to(self.rank)
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}

        if "ckpt_dir" in self.config.model.__dict__.keys():
            ckpt_dir = os.path.expanduser(self.config.model.ckpt_dir)
            states = torch.load(
                ckpt_dir,
                map_location=map_location
            )
            # states = {f"module.{k}":v for k, v in states.items()}
            if self.config.model.model_type == 'improved_ddpm' or self.config.model.model_type == 'guided_diffusion':
                model.load_state_dict(states, strict=True)
                if self.config.model.use_fp16:
                    model.convert_to_fp16()
            else:
                # TODO: FIXME
                # model = torch.nn.DataParallel(model)
                # model.load_state_dict(states[0], strict=True)
                model.load_state_dict(states, strict=True)

            if self.config.model.ema: # for celeba 64x64 in DDIM
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None

            if self.config.sampling.cond_class and not self.config.model.is_upsampling:
                classifier = GuidedDiffusion_Classifier(
                    image_size=self.config.classifier.image_size,
                    in_channels=self.config.classifier.in_channels,
                    model_channels=self.config.classifier.model_channels,
                    out_channels=self.config.classifier.out_channels,
                    num_res_blocks=self.config.classifier.num_res_blocks,
                    attention_resolutions=self.config.classifier.attention_resolutions,
                    channel_mult=self.config.classifier.channel_mult,
                    use_fp16=self.config.classifier.use_fp16,
                    num_head_channels=self.config.classifier.num_head_channels,
                    use_scale_shift_norm=self.config.classifier.use_scale_shift_norm,
                    resblock_updown=self.config.classifier.resblock_updown,
                    pool=self.config.classifier.pool
                )
                ckpt_dir = os.path.expanduser(self.config.classifier.ckpt_dir)
                states = torch.load(
                    ckpt_dir,
                    map_location=map_location,
                )
                # states = {f"module.{k}":v for k, v in states.items()}
                classifier = classifier.to(self.rank)
                # classifier = DDP(classifier, device_ids=[self.rank])
                classifier.load_state_dict(states, strict=True)
                if self.config.classifier.use_fp16:
                    classifier.convert_to_fp16()
                    # classifier.module.convert_to_fp16()
            else:
                classifier = None
        else:
            classifier = None
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            if self.rank == 0:
                print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=map_location))

        model.eval()

        # (4) self.model에 할당
        self.model = model
        self.classifier = classifier
        print("[prepare_model] Model is ready.")

    def sample_npz(self):
        self.sample(npz=True)
        return

    def sample(self, npz=False):
        if self.config.model.model_type == 'improved_ddpm':
            model = ImprovedDDPM_Model(
                in_channels=self.config.model.in_channels,
                model_channels=self.config.model.model_channels,
                out_channels=self.config.model.out_channels,
                num_res_blocks=self.config.model.num_res_blocks,
                attention_resolutions=self.config.model.attention_resolutions,
                dropout=self.config.model.dropout,
                channel_mult=self.config.model.channel_mult,
                conv_resample=self.config.model.conv_resample,
                dims=self.config.model.dims,
                use_checkpoint=self.config.model.use_checkpoint,
                num_heads=self.config.model.num_heads,
                num_heads_upsample=self.config.model.num_heads_upsample,
                use_scale_shift_norm=self.config.model.use_scale_shift_norm
            )
        elif self.config.model.model_type == "guided_diffusion":
            if self.config.model.is_upsampling:
                model = GuidedDiffusion_SRModel(
                    image_size=self.config.model.large_size,
                    in_channels=self.config.model.in_channels,
                    model_channels=self.config.model.model_channels,
                    out_channels=self.config.model.out_channels,
                    num_res_blocks=self.config.model.num_res_blocks,
                    attention_resolutions=self.config.model.attention_resolutions,
                    dropout=self.config.model.dropout,
                    channel_mult=self.config.model.channel_mult,
                    conv_resample=self.config.model.conv_resample,
                    dims=self.config.model.dims,
                    num_classes=self.config.model.num_classes,
                    use_checkpoint=self.config.model.use_checkpoint,
                    use_fp16=self.config.model.use_fp16,
                    num_heads=self.config.model.num_heads,
                    num_head_channels=self.config.model.num_head_channels,
                    num_heads_upsample=self.config.model.num_heads_upsample,
                    use_scale_shift_norm=self.config.model.use_scale_shift_norm,
                    resblock_updown=self.config.model.resblock_updown,
                    use_new_attention_order=self.config.model.use_new_attention_order,
                )
            else:
                model = GuidedDiffusion_Model(
                    image_size=self.config.model.image_size,
                    in_channels=self.config.model.in_channels,
                    model_channels=self.config.model.model_channels,
                    out_channels=self.config.model.out_channels,
                    num_res_blocks=self.config.model.num_res_blocks,
                    attention_resolutions=self.config.model.attention_resolutions,
                    dropout=self.config.model.dropout,
                    channel_mult=self.config.model.channel_mult,
                    conv_resample=self.config.model.conv_resample,
                    dims=self.config.model.dims,
                    num_classes=self.config.model.num_classes,
                    use_checkpoint=self.config.model.use_checkpoint,
                    use_fp16=self.config.model.use_fp16,
                    num_heads=self.config.model.num_heads,
                    num_head_channels=self.config.model.num_head_channels,
                    num_heads_upsample=self.config.model.num_heads_upsample,
                    use_scale_shift_norm=self.config.model.use_scale_shift_norm,
                    resblock_updown=self.config.model.resblock_updown,
                    use_new_attention_order=self.config.model.use_new_attention_order,
                )
        else:
            model = Model(self.config)

        model = model.to(self.rank)
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}

        if "ckpt_dir" in self.config.model.__dict__.keys():
            ckpt_dir = os.path.expanduser(self.config.model.ckpt_dir)
            states = torch.load(
                ckpt_dir,
                map_location=map_location
            )
            # states = {f"module.{k}":v for k, v in states.items()}
            if self.config.model.model_type == 'improved_ddpm' or self.config.model.model_type == 'guided_diffusion':
                model.load_state_dict(states, strict=True)
                if self.config.model.use_fp16:
                    model.convert_to_fp16()
            else:
                # TODO: FIXME
                # model = torch.nn.DataParallel(model)
                # model.load_state_dict(states[0], strict=True)
                model.load_state_dict(states, strict=True)

            if self.config.model.ema: # for celeba 64x64 in DDIM
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None

            if self.config.sampling.cond_class and not self.config.model.is_upsampling:
                classifier = GuidedDiffusion_Classifier(
                    image_size=self.config.classifier.image_size,
                    in_channels=self.config.classifier.in_channels,
                    model_channels=self.config.classifier.model_channels,
                    out_channels=self.config.classifier.out_channels,
                    num_res_blocks=self.config.classifier.num_res_blocks,
                    attention_resolutions=self.config.classifier.attention_resolutions,
                    channel_mult=self.config.classifier.channel_mult,
                    use_fp16=self.config.classifier.use_fp16,
                    num_head_channels=self.config.classifier.num_head_channels,
                    use_scale_shift_norm=self.config.classifier.use_scale_shift_norm,
                    resblock_updown=self.config.classifier.resblock_updown,
                    pool=self.config.classifier.pool
                )
                ckpt_dir = os.path.expanduser(self.config.classifier.ckpt_dir)
                states = torch.load(
                    ckpt_dir,
                    map_location=map_location,
                )
                # states = {f"module.{k}":v for k, v in states.items()}
                classifier = classifier.to(self.rank)
                # classifier = DDP(classifier, device_ids=[self.rank])
                classifier.load_state_dict(states, strict=True)
                if self.config.classifier.use_fp16:
                    classifier.convert_to_fp16()
                    # classifier.module.convert_to_fp16()
            else:
                classifier = None
        else:
            classifier = None
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            if self.rank == 0:
                print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=map_location))

        model.eval()

        if self.args.fid:
            if not os.path.exists(os.path.join(self.args.exp, "fid.npy")):
                self.sample_fid(model, classifier=classifier, npz=npz)
                torch.distributed.barrier()
                if self.rank == 0:
                    print("Begin to compute FID...")
                    fid = calculate_fid_given_paths((self.config.sampling.fid_stats_dir, self.args.image_folder), batch_size=self.config.sampling.fid_batch_size, device=self.device, dims=2048, num_workers=8)
                    print("FID: {}".format(fid))
                    np.save(os.path.join(self.args.exp, "fid"), fid)
        # elif self.args.interpolation:
        #     self.sample_interpolation(model)
        # elif self.args.sequence:
        #     self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, model, classifier=None, npz=False):
        config = self.config
        total_n_samples = config.sampling.fid_total_samples
        world_size = torch.cuda.device_count()
        if total_n_samples % config.sampling.batch_size != 0:
            raise ValueError("Total samples for sampling must be divided exactly by config.sampling.batch_size, but got {} and {}".format(total_n_samples, config.sampling.batch_size))
        if len(glob.glob(f"{self.args.image_folder}/*.png")) == total_n_samples:
            return
        else:
            n_rounds = total_n_samples // config.sampling.batch_size // world_size
        img_id = self.rank * total_n_samples // world_size

        if self.config.model.is_upsampling:
            base_samples_total = load_data_for_worker(self.args.base_samples, config.sampling.batch_size, config.sampling.cond_class)

        with torch.no_grad():
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                # torch.cuda.synchronize()
                # start = torch.cuda.Event(enable_timing=True)
                # end = torch.cuda.Event(enable_timing=True)
                # start.record()

                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )
                noise = x
                
                if self.config.model.is_upsampling:
                    base_samples = next(base_samples_total)
                else:
                    base_samples = None

                if npz:
                    (x, traj, timesteps), classes = self.sample_image(x, model, classifier=classifier, base_samples=base_samples, return_intermediate=True)
                else:
                    x, classes = self.sample_image(x, model, classifier=classifier, base_samples=base_samples)
                
                if npz:
                    for i in range(len(x)):
                        if classes is None:
                            np.savez_compressed(
                                os.path.join(self.args.image_folder, f"{img_id}.npz"),
                                noise=noise[i].cpu(),
                                traj=traj[:, i].cpu(),
                                timesteps=timesteps.cpu(),
                                image=x[i].cpu()
                            )
                        else:
                            np.savez_compressed(
                                os.path.join(self.args.image_folder, f"{img_id}.npz"),
                                noise=noise[i].cpu(),
                                traj=traj[:, i].cpu(),
                                timesteps=timesteps.cpu(),
                                image=x[i].cpu(),
                                clazz=classes[i].cpu(),
                            )
                        img_id += 1
                else:    
                    x = inverse_data_transform(config, x)
                    for i in range(x.shape[0]):
                        if classes is None:
                            path = os.path.join(self.args.image_folder, f"{img_id}.png")
                        else:
                            path = os.path.join(self.args.image_folder, f"{img_id}_{int(classes.cpu()[i])}.png")
                        tvu.save_image(x.cpu()[i], path)
                        img_id += 1
        
    def sample_sequence(self, model, classifier=None):
        config = self.config
        print('sample_sequence randn')
        x = torch.randn(
            8,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            _, x = self.sample_image(x, model, last=False, classifier=classifier)

        x = [inverse_data_transform(config, y) for y in x]

        for i in range(len(x)):
            for j in range(x[i].size(0)):
                tvu.save_image(
                    x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
                )

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        print('sample_interpolation randn')
        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

    def sample_image(self, x, model, classes=None, last=True, classifier=None, base_samples=None, target=None, number=0, return_intermediate=False):
        assert last
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        classifier_scale = self.config.sampling.classifier_scale if self.args.scale is None else self.args.scale
        if classes is not None:
            pass
        elif self.config.sampling.cond_class:
            if self.args.fixed_class is None:
                classes = torch.randint(low=0, high=self.config.data.num_classes, size=(x.shape[0],)).to(x.device)
            else:
                classes = torch.randint(low=self.args.fixed_class, high=self.args.fixed_class + 1, size=(x.shape[0],)).to(x.device)
        else:
            classes = None
        
        if base_samples is None:
            if classes is None:
                model_kwargs = {}
            else:
                model_kwargs = {"y": classes}
        else:
            model_kwargs = {"y": base_samples["y"], "low_res": base_samples["low_res"]}

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps
            def model_fn(x, t, **model_kwargs):
                out = model(x, t, **model_kwargs)
                if "out_channels" in self.config.model.__dict__.keys():
                    if self.config.model.out_channels == 6:
                        return torch.split(out, 3, dim=1)[0]
                return out
            xs, _ = generalized_steps(x, seq, model_fn, self.betas, eta=self.args.eta, classifier=classifier, is_cond_classifier=self.config.sampling.cond_class, classifier_scale=classifier_scale, **model_kwargs)
            x = xs[-1]
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps
            def model_fn(x, t, **model_kwargs):
                out = model(x, t, **model_kwargs)
                if "out_channels" in self.config.model.__dict__.keys():
                    if self.config.model.out_channels == 6:
                        return torch.split(out, 3, dim=1)[0]
                return out
            xs, _ = ddpm_steps(x, seq, model_fn, self.betas, classifier=classifier, is_cond_classifier=self.config.sampling.cond_class, classifier_scale=classifier_scale, **model_kwargs)
            x = xs[-1]

        else:
            from dpm_solver.sampler import NoiseScheduleVP, model_wrapper, DPM_Solver
            from dpm_solver.lagrange_solver import LagrangeSolver
            from dpm_solver.sa_solver import SASolver
            from dpm_solver.uni_pc import UniPC
            from dpm_solver.rbf_solver import RBFSolver
            from dpm_solver.rbf_solver_closed import RBFSolverClosed
            from dpm_solver.rbf_solver_closed_sweep import RBFSolverClosedSweep
            from dpm_solver.rbf_solver_closed_grad import RBFSolverClosedGrad
            from dpm_solver.rbf_solver_numerical import RBFSolverNumerical
            from dpm_solver.rbf_solver_numerical_sweep import RBFSolverNumericalSweep
            from dpm_solver.rbf_solver_quad_sweep import RBFSolverQuadSweep
            from dpm_solver.rbf_solver_quad import RBFSolverQuad
            from dpm_solver.rbf_solver_glq import RBFSolverGLQ
            from dpm_solver.rbf_solver_glq10 import RBFSolverGLQ10
            from dpm_solver.rbf_solver_glq10_lag import RBFSolverGLQ10Lag
            from dpm_solver.rbf_solver_glq10_lag_time import RBFSolverGLQ10LagTime
            from dpm_solver.rbf_solver_glq10_lag_grid import RBFSolverGLQ10LagGrid
            from dpm_solver.rbf_solver_glq10_grad import RBFSolverGLQ10Grad
            from dpm_solver.rbf_solver_glq10_reg import RBFSolverGLQ10Reg
            from dpm_solver.rbf_solver_glq10_hist import RBFSolverGLQ10Hist
            from dpm_solver.rbf_solver_glq10_sepbeta import RBFSolverGLQ10Sepbeta
            from dpm_solver.rbf_ecp_marginal import RBFSolverECPMarginal
            from dpm_solver.rbf_ecp_marginal_spd import RBFSolverECPMarginalSPD
            from dpm_solver.rbf_ecp_marginal_lagp import RBFSolverECPMarginalLagP
            from dpm_solver.rbf_ecp_marginal_lagc import RBFSolverECPMarginalLagC
            from dpm_solver.rbf_ecp_marginal_to1 import RBFSolverECPMarginalTo1
            from dpm_solver.rbf_ecp_marginal_to3 import RBFSolverECPMarginalTo3
            from dpm_solver.rbf_marginal import RBFSolverMarginal
            from dpm_solver.rbf_marginal_spd import RBFSolverMarginalSPD
            from dpm_solver.rbf_marginal_lagp import RBFSolverMarginalLagP
            from dpm_solver.rbf_marginal_lagc import RBFSolverMarginalLagC
            from dpm_solver.rbf_marginal_to1 import RBFSolverMarginalTo1
            from dpm_solver.rbf_marginal_to3 import RBFSolverMarginalTo3
            from dpm_solver.rbf_ecp_marginal_same import RBFSolverECPMarginalSame
            from dpm_solver.general_rbf_solver import GeneralRBFSolver
            from dpm_solver.general_rbf_solver_grad import GeneralRBFSolverGrad
            from dpm_solver.rbf_solver_cpd_target_lg import RBFSolverCPDTargetLG
            from dpm_solver.laplace_solver import LaplaceSolver
            from dpm_solver.dc_solver import DCSolver
            def model_fn(x, t, **model_kwargs):
                out = model(x, t, **model_kwargs)
                # If the model outputs both 'mean' and 'variance' (such as improved-DDPM and guided-diffusion),
                # We only use the 'mean' output for DPM-Solver, because DPM-Solver is based on diffusion ODEs.
                if "out_channels" in self.config.model.__dict__.keys():
                    if self.config.model.out_channels == 6:
                        out = torch.split(out, 3, dim=1)[0]
                return out

            def classifier_fn(x, t, y, **classifier_kwargs):
                logits = classifier(x, t)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                return log_probs[range(len(logits)), y.view(-1)]

            noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas)
            model_fn_continuous = model_wrapper(
                model_fn,
                noise_schedule,
                model_type="noise",
                model_kwargs=model_kwargs,
                guidance_type="uncond" if classifier is None else "classifier",
                condition=model_kwargs["y"] if "y" in model_kwargs.keys() else None,
                guidance_scale=classifier_scale,
                classifier_fn=classifier_fn,
                classifier_kwargs={},
            )

            if self.args.sample_type in ["dpmsolver", "dpmsolver++"]:
                dpm_solver = DPM_Solver(
                    model_fn_continuous,
                    noise_schedule,
                    algorithm_type=self.args.sample_type,
                    correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
                )
                x = dpm_solver.sample(
                    x,
                    steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                    order=self.args.dpm_solver_order,
                    skip_type=self.args.skip_type,
                    method=self.args.dpm_solver_method,
                    lower_order_final=self.args.lower_order_final,
                    denoise_to_zero=self.args.denoise,
                    solver_type=self.args.dpm_solver_type,
                    atol=self.args.dpm_solver_atol,
                    rtol=self.args.dpm_solver_rtol,
                    return_intermediate=return_intermediate
                )
            if self.args.sample_type in ["lagrangesolver"]:
                solver = LagrangeSolver(
                    model_fn_continuous,
                    noise_schedule,
                    algorithm_type=self.args.dpm_solver_type,
                    correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
                )
                x = solver.sample(
                    x,
                    steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                    order=self.args.dpm_solver_order,
                    skip_type=self.args.skip_type,
                    #return_hist=return_hist,
                )    
            if self.args.sample_type in ["sasolver"]:
                solver = SASolver(
                    model_fn_continuous,
                    noise_schedule,
                    algorithm_type=self.args.dpm_solver_type,
                    correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
                )
                x = solver.sample(
                    mode='few_steps',
                    x=x,
                    tau=lambda tau: 0.0,
                    steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                    predictor_order=self.args.dpm_solver_order,
                    corrector_order=self.args.dpm_solver_order+1,
                    skip_type=self.args.skip_type,
                )        
            if self.args.sample_type in ["unipc"]:
                solver = UniPC(
                    model_fn_continuous,
                    noise_schedule,
                    algorithm_type=self.args.dpm_solver_type,
                    correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
                )
                x = solver.sample(
                    x,
                    steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                    order=self.args.dpm_solver_order,
                    skip_type=self.args.skip_type,
                )   

            if self.args.sample_type in ["dcsolver"]:
                solver = DCSolver(
                    model_fn_continuous,
                    noise_schedule,
                    dc_dir=self.args.dc_dir
                )
                if target is not None:
                    solver.ref_ts = target[1]
                    solver.ref_xs = target[0]
                    x = solver.search_dc(
                        x,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,
                        method='multistep',
                    )

                else:
                    x = solver.sample(
                        x,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,
                        method='multistep',
                    )   
            if self.args.sample_type in ["rbfsolver"]:
                solver = RBFSolver(
                    model_fn_continuous,
                    noise_schedule,
                    algorithm_type=self.args.dpm_solver_type,
                    correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
                )
                x = solver.sample(
                    x,
                    steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                    order=self.args.dpm_solver_order,
                    skip_type=self.args.skip_type,
                    gamma=self.args.gamma
                )
            if self.args.sample_type in ["rbfsolverclosed"]:
                solver = RBFSolverClosed(
                    model_fn_continuous,
                    noise_schedule,
                    algorithm_type=self.args.dpm_solver_type,
                    correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
                    scale_dir=self.args.scale_dir
                )
                if target is not None:
                    x = solver.sample_by_target_matching(
                        x,
                        target,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,
                        log_scale_min=self.args.log_scale_min,
                        log_scale_max=self.args.log_scale_max,
                    )
                else:    
                    x = solver.sample(
                        x,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,
                        log_scale=self.args.log_scale,
                    )
            if self.args.sample_type in ["rbfsolverclosedgrad"]:
                solver = RBFSolverClosedGrad(
                    model_fn_continuous,
                    noise_schedule,
                    algorithm_type=self.args.dpm_solver_type,
                    correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
                    scale_dir=self.args.scale_dir
                )
                if target is not None:
                    x = solver.sample_by_target_matching(
                        x,
                        target,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,
                        log_scale_min=self.args.log_scale_min,
                        log_scale_max=self.args.log_scale_max,
                    )
                else:    
                    x = solver.sample(
                        x,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,
                        log_scale=self.args.log_scale,
                    )        
            if self.args.sample_type in ["rbfsolvernumerical"]:
                solver = RBFSolverNumerical(
                    model_fn_continuous,
                    noise_schedule,
                    algorithm_type=self.args.dpm_solver_type,
                    correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
                    scale_dir=self.args.scale_dir
                )
                if target is not None:
                    x = solver.sample_by_target_matching(
                        x,
                        target,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,
                        log_scale_min=self.args.log_scale_min,
                        log_scale_max=self.args.log_scale_max,
                    )
                else:    
                    x = solver.sample(
                        x,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,
                        log_scale=self.args.log_scale,
                    )        
            if self.args.sample_type in ["rbfsolvernumericalsweep"]:
                solver = RBFSolverNumericalSweep(
                    model_fn_continuous,
                    noise_schedule,
                    algorithm_type=self.args.dpm_solver_type,
                    correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
                    scale_dir=self.args.scale_dir
                )
                if target is not None:
                    x = solver.sample_by_target_matching(
                        x,
                        target,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,
                        log_scale_min=self.args.log_scale_min,
                        log_scale_max=self.args.log_scale_max,
                    )
                else:    
                    x = solver.sample(
                        x,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,
                        log_scale=self.args.log_scale,
                    )

            if self.args.sample_type in ["rbfsolverquadsweep"]:
                solver = RBFSolverQuadSweep(
                    model_fn_continuous,
                    noise_schedule,
                    algorithm_type=self.args.dpm_solver_type,
                    correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
                    scale_dir=self.args.scale_dir,
                    exp_num=self.args.exp_num
                )
                if target is not None:
                    x = solver.sample_by_target_matching(
                        x,
                        target,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,
                        log_scale_min=self.args.log_scale_min,
                        log_scale_max1=self.args.log_scale_max1,
                        log_scale_max2=self.args.log_scale_max2,
                        log_scale_num=self.args.log_scale_num,
                        exp_num=exp_num,
                    )
                else:    
                    x = solver.sample(
                        x,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,
                        log_scale=self.args.log_scale,
                    )

            if self.args.sample_type in ["rbfsolverquad"]:
                solver = RBFSolverQuad(
                    model_fn_continuous,
                    noise_schedule,
                    algorithm_type=self.args.dpm_solver_type,
                    correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
                    scale_dir=self.args.scale_dir,
                    exp_num=self.args.exp_num
                )
                if target is not None:
                    x = solver.sample_by_target_matching(
                        x,
                        target,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,
                        log_scale_min=self.args.log_scale_min,
                        log_scale_max=self.args.log_scale_max,
                        log_scale_num=self.args.log_scale_num,
                        exp_num=exp_num,
                    )
                else:    
                    x = solver.sample(
                        x,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,
                        log_scale=self.args.log_scale,
                    )

            if self.args.sample_type in ["rbfsolverglq"]:
                solver = RBFSolverGLQ(
                    model_fn_continuous,
                    noise_schedule,
                    algorithm_type=self.args.dpm_solver_type,
                    correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
                    scale_dir=self.args.scale_dir,
                    exp_num=self.args.exp_num
                )
                if target is not None:
                    x = solver.sample_by_target_matching(
                        x,
                        target,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,
                        log_scale_min=self.args.log_scale_min,
                        log_scale_max=self.args.log_scale_max,
                        log_scale_num=self.args.log_scale_num,
                        exp_num=exp_num,
                    )
                else:    
                    x = solver.sample(
                        x,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,
                        log_scale=self.args.log_scale,
                    )

            if self.args.sample_type in ["rbfsolverglq10"]:
                solver = RBFSolverGLQ10(
                    model_fn_continuous,
                    noise_schedule,
                    algorithm_type=self.args.dpm_solver_type,
                    correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
                    scale_dir=self.args.scale_dir,
                    dataset=self.config.data.dataset
                )
                if target is not None:
                    x = solver.sample_by_target_matching(
                        x,
                        target,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,
                        log_scale_min=self.args.log_scale_min,
                        log_scale_max=self.args.log_scale_max,
                        log_scale_num=self.args.log_scale_num,
                    )
                else:    
                    x = solver.sample(
                        x,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,
                        log_scale=self.args.log_scale,
                    )                

            if self.args.sample_type in ["rbfsolverglq10lag"]:
                solver = RBFSolverGLQ10Lag(
                    model_fn_continuous,
                    noise_schedule,
                    algorithm_type=self.args.dpm_solver_type,
                    correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
                    scale_dir=self.args.scale_dir,
                    dataset=self.config.data.dataset,
                    log_scale_min=self.args.log_scale_min,
                    log_scale_max=self.args.log_scale_max,
                    log_scale_num=self.args.log_scale_num,
                )
                if target is not None:
                    x = solver.sample_by_target_matching(
                        x,
                        target,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,  
                    )
                else:    
                    x = solver.sample(
                        x,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,
                        log_scale=self.args.log_scale,
                    )                        

            if self.args.sample_type in ["rbfsolverglq10lagtime"]:
                solver = RBFSolverGLQ10LagTime(
                    model_fn_continuous,
                    noise_schedule,
                    algorithm_type=self.args.dpm_solver_type,
                    correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
                    scale_dir=self.args.scale_dir,
                    dataset=self.config.data.dataset,
                    log_scale_min=self.args.log_scale_min,
                    log_scale_max=self.args.log_scale_max,
                    log_scale_num=self.args.log_scale_num,
                )
                if target is not None:
                    x = solver.sample_by_target_matching(
                        x,
                        target,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,  
                    )
                else:    
                    x = solver.sample(
                        x,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,
                        log_scale=self.args.log_scale,
                    )                                

            if self.args.sample_type in ["rbfsolverglq10laggrid"]:
                solver = RBFSolverGLQ10LagGrid(
                    model_fn_continuous,
                    noise_schedule,
                    algorithm_type=self.args.dpm_solver_type,
                    correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
                    scale_dir=self.args.scale_dir,
                    dataset=self.config.data.dataset,
                    log_scale_min=self.args.log_scale_min,
                    log_scale_max=self.args.log_scale_max,
                    log_scale_num=self.args.log_scale_num,
                )
                if target is not None:
                    x = solver.sample_by_target_matching(
                        x,
                        target,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,  
                    )
                else:
                    if ecp:    
                        x = solver.sample_ecp(
                            x,
                            steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                            order=self.args.dpm_solver_order,
                            skip_type=self.args.skip_type,
                            log_scale=self.args.log_scale,
                        )                                
                                          
                    else:
                        x = solver.sample(
                            x,
                            steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                            order=self.args.dpm_solver_order,
                            skip_type=self.args.skip_type,
                            log_scale=self.args.log_scale,
                        )                                

            if self.args.sample_type in ["rbfsolverglq10grad"]:
                solver = RBFSolverGLQ10Grad(
                    model_fn_continuous,
                    noise_schedule,
                    algorithm_type=self.args.dpm_solver_type,
                    correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
                    scale_dir=self.args.scale_dir,
                    dataset=self.config.data.dataset
                )
                if target is not None:
                    x = solver.sample_by_target_matching(
                        x,
                        target,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,
                        log_scale_min=self.args.log_scale_min,
                        log_scale_max=self.args.log_scale_max,
                    )
                else:    
                    x = solver.sample(
                        x,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,
                        log_scale=self.args.log_scale,
                    )                        

            if self.args.sample_type in ["rbfsolverglq10reg"]:
                solver = RBFSolverGLQ10Reg(
                    model_fn_continuous,
                    noise_schedule,
                    algorithm_type=self.args.dpm_solver_type,
                    correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
                    scale_dir=self.args.scale_dir,
                    dataset=self.config.data.dataset,
                    reg_weight=self.args.reg_weight
                )
                if target is not None:
                    x = solver.sample_by_target_matching(
                        x,
                        target,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,
                        log_scale_min=self.args.log_scale_min,
                        log_scale_max=self.args.log_scale_max,
                        log_scale_num=self.args.log_scale_num,
                    )
                else:    
                    x = solver.sample(
                        x,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,
                        log_scale=self.args.log_scale,
                    )                        

            if self.args.sample_type in ["rbfsolverglq10hist"]:
                solver = RBFSolverGLQ10Hist(
                    model_fn_continuous,
                    noise_schedule,
                    algorithm_type=self.args.dpm_solver_type,
                    correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
                    scale_dir=self.args.scale_dir,
                    exp_num=self.args.exp_num
                )
                if target is not None and hist is not None:
                    x = solver.sample_by_target_matching(
                        target,
                        hist,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,
                        log_scale_min=self.args.log_scale_min,
                        log_scale_max=self.args.log_scale_max,
                        log_scale_num=self.args.log_scale_num,
                        exp_num=exp_num,
                    )
                else:    
                    x = solver.sample(
                        x,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,
                        log_scale=self.args.log_scale,
                    )                

            if self.args.sample_type in ["rbfsolverglq10sepbeta"]:
                solver = RBFSolverGLQ10Sepbeta(
                    model_fn_continuous,
                    noise_schedule,
                    algorithm_type=self.args.dpm_solver_type,
                    correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
                    scale_dir=self.args.scale_dir,
                    exp_num=self.args.exp_num
                )
                if target is not None:
                    x = solver.sample_by_target_matching(
                        x,
                        target,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,
                        log_scale_min=self.args.log_scale_min,
                        log_scale_max=self.args.log_scale_max,
                        exp_num=exp_num,
                    )
                else:    
                    x = solver.sample(
                        x,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,
                        log_scale=self.args.log_scale,
                    )                

            # if self.args.sample_type in ["rbf_ecp_marginal"]:
            #     solver = RBFSolverECPMarginal(
            #         model_fn_continuous,
            #         noise_schedule,
            #         algorithm_type=self.args.dpm_solver_type,
            #         correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
            #         scale_dir=self.args.scale_dir,
            #     )

            #     if target is not None:
            #         x = solver.sample_by_target_matching(
            #             x,
            #             target,
            #             steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
            #             order=self.args.dpm_solver_order,
            #             skip_type=self.args.skip_type,                       
            #             number=number
            #         )
            #     else:    
            #         x = solver.sample(
            #             x,
            #             steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
            #             order=self.args.dpm_solver_order,
            #             skip_type=self.args.skip_type,
            #         )                

            print(self.args.sample_type)
            if "rbf_marginal" in self.args.sample_type:
                if self.args.sample_type == 'rbf_marginal':
                    SOLVER = RBFSolverMarginal
                elif self.args.sample_type == 'rbf_marginal_spd':
                    SOLVER = RBFSolverMarginalSPD
                elif self.args.sample_type == 'rbf_marginal_lagp':
                    SOLVER = RBFSolverMarginalLagP
                elif self.args.sample_type == 'rbf_marginal_lagc':
                    SOLVER = RBFSolverMarginalLagC
                elif self.args.sample_type == 'rbf_marginal_to1':
                    SOLVER = RBFSolverMarginalTo1
                elif self.args.sample_type == 'rbf_marginal_to3':
                    SOLVER = RBFSolverMarginalTo3
                elif self.args.sample_type == 'rbf_ecp_marginal_same':
                    SOLVER = RBFSolverECPMarginalSame    
                
                solver = SOLVER(
                    model_fn_continuous,
                    noise_schedule,
                    algorithm_type=self.args.dpm_solver_type,
                    correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
                    scale_dir=self.args.scale_dir,
                )

                if target is not None:
                    x = solver.sample_by_target_matching(
                        x,
                        target,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,                       
                        number=number
                    )
                else:    
                    x = solver.sample(
                        x,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,
                    )                

            if "rbf_ecp_marginal" in self.args.sample_type:
                if self.args.sample_type == 'rbf_ecp_marginal':
                    SOLVER = RBFSolverECPMarginal
                elif self.args.sample_type == 'rbf_ecp_marginal_spd':
                    SOLVER = RBFSolverECPMarginalSPD
                elif self.args.sample_type == 'rbf_ecp_marginal_lagp':
                    SOLVER = RBFSolverECPMarginalLagP
                elif self.args.sample_type == 'rbf_ecp_marginal_lagc':
                    SOLVER = RBFSolverECPMarginalLagC
                elif self.args.sample_type == 'rbf_ecp_marginal_to1':
                    SOLVER = RBFSolverECPMarginalTo1
                elif self.args.sample_type == 'rbf_ecp_marginal_to3':
                    SOLVER = RBFSolverECPMarginalTo3
                elif self.args.sample_type == 'rbf_ecp_marginal_same':
                    SOLVER = RBFSolverECPMarginalSame    
                
                solver = SOLVER(
                    model_fn_continuous,
                    noise_schedule,
                    algorithm_type=self.args.dpm_solver_type,
                    correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
                    scale_dir=self.args.scale_dir,
                )

                if target is not None:
                    x = solver.sample_by_target_matching(
                        x,
                        target,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,                       
                        number=number
                    )
                else:    
                    x = solver.sample(
                        x,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,
                    )                    

            if self.args.sample_type in ["rbfsolverclosedsweep"]:
                solver = RBFSolverClosedSweep(
                    model_fn_continuous,
                    noise_schedule,
                    algorithm_type=self.args.dpm_solver_type,
                    correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
                    scale_dir=self.args.scale_dir,
                    exp_num=self.args.exp_num
                )
                if target is not None:
                    x = solver.sample_by_target_matching(
                        x,
                        target,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,
                        log_scale_min=self.args.log_scale_min,
                        log_scale_max1=self.args.log_scale_max1,
                        log_scale_max2=self.args.log_scale_max2,
                        log_scale_num=self.args.log_scale_num,
                        exp_num=exp_num,
                    )
                else:    
                    x = solver.sample(
                        x,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,
                        log_scale=self.args.log_scale,
                    )        

            if self.args.sample_type in ["generalrbfsolver"]:
                solver = GeneralRBFSolver(
                    model_fn_continuous,
                    noise_schedule,
                    algorithm_type=self.args.dpm_solver_type,
                    correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
                    kernel_name=self.args.kernel_name,
                    subintervals=self.args.subintervals,
                    scale_dir=self.args.scale_dir
                )
                if target is not None:
                    x = solver.sample_by_target_matching(
                        x,
                        target,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,
                        log_scale_max=self.args.log_scale_max,
                    )
                else:    
                    x = solver.sample(
                        x,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,
                        log_scale=self.args.log_scale,
                    )
            if self.args.sample_type in ["generalrbfsolvergrad"]:
                solver = GeneralRBFSolverGrad(
                    model_fn_continuous,
                    noise_schedule,
                    algorithm_type=self.args.dpm_solver_type,
                    correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
                    kernel_name=self.args.kernel_name,
                    subintervals=self.args.subintervals,
                    scale_dir=self.args.scale_dir
                )
                if target is not None:
                    x = solver.sample_by_target_matching(
                        x,
                        target,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,
                        log_scale_max=self.args.log_scale_max,
                    )
                else:    
                    x = solver.sample(
                        x,
                        steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                        order=self.args.dpm_solver_order,
                        skip_type=self.args.skip_type,
                        log_scale=self.args.log_scale,
                    )            
            if self.args.sample_type in ["rbfsolvercpdtargetlg"]:
                solver = RBFSolverCPDTargetLG(
                    model_fn_continuous,
                    noise_schedule,
                    algorithm_type=self.args.dpm_solver_type,
                    correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
                )
                x = solver.sample(
                    x,
                    steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                    order=self.args.dpm_solver_order,
                    skip_type=self.args.skip_type,
                    gamma_max=self.args.gamma_max
                )
            if self.args.sample_type in ["laplacesolver"]:
                solver = LaplaceSolver(
                    model_fn_continuous,
                    noise_schedule,
                    algorithm_type=self.args.dpm_solver_type,
                    correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
                )
                x = solver.sample(
                    x,
                    steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                    order=self.args.dpm_solver_order,
                    skip_type=self.args.skip_type,
                    log_gamma=self.args.log_gamma,
                    log_gamma_max=self.args.log_gamma_max
                )    
            # x = x.cpu()
        
        return x, classes

    def test(self):
        pass

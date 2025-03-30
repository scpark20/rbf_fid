import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import random

from main import parse_args_and_config, Diffusion
from datasets import inverse_data_transform

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.benchmark = False

###############################################################################
# 1) Notebook에서 sys.argv를 직접 설정 (argparse 흉내)
###############################################################################
sys.argv = [
    "main.py",
    "--config", "imagenet128_guided.yml",  # 사용하려는 config
    "--sample",
    "--eta", "0",
    "--sample_type", "lagrangesolver",
    "--dpm_solver_type", "data_prediction",
    "--dpm_solver_order", "1",
    "--timesteps", "1000",
    "--skip_type", "logSNR",
    "--scale", "0.0",
    "--thresholding",
    "--ni"
]

###############################################################################
# 2) 인자/설정 로드
###############################################################################
args, config = parse_args_and_config()

###############################################################################
# 3) Diffusion 객체 생성 -> 모델 로딩
###############################################################################
diffusion = Diffusion(args, config, rank=0)
diffusion.prepare_model()
diffusion.model.eval()

###############################################################################
# 4) 배치(25장) 한 번에 샘플링 -> 5x5 그리드(여백 없이) 시각화
###############################################################################
device = diffusion.device

from tqdm import tqdm

def sample_in_batches(diffusion, config, device, total_samples=1024, batch_size=16):
    with torch.no_grad():
        pairs = []
        class_list = []
        for i in tqdm(range(0, total_samples, batch_size)):
            bs = min(batch_size, total_samples - i)
            noise = np.random.randn(bs, config.data.channels, config.data.image_size, config.data.image_size).astype(np.float32)
            noise = torch.tensor(noise, device=device)
            classes = np.random.randint(0, config.data.num_classes, size=(noise.shape[0],))
            classes = torch.tensor(classes).to(device)
            data, _ = diffusion.sample_image(noise, diffusion.model, classes=classes)
            pair = torch.stack([noise.data.cpu(), data.data.cpu()], dim=1)
            pairs.append(pair)
            class_list.append(classes.data.cpu())
    return torch.cat(pairs, dim=0), torch.cat(class_list, dim=0)

total_samples = 256
pairs, classes = sample_in_batches(diffusion, config, device, total_samples=total_samples, batch_size=4)

save_file = f'/data/optimization/euler_NFE=1000_N={total_samples}_imagenet128.pt'
torch.save({'pairs': pairs.data.cpu(),
            'classes': classes.data.cpu()
            }, save_file)
pairs_load = torch.load(save_file)
print(pairs_load['pairs'].shape, pairs_load['classes'].shape)
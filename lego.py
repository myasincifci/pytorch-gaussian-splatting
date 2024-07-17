from typing import List, Optional

import time 
import math
import os
import copy
import random
import torch
import matplotlib.pyplot as plt
from renderer import Renderer
from tqdm import tqdm
from utils import readCamerasFromTransforms, CameraInfo
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np

def image_path_to_tensor(image_path):
    img = Image.open(image_path)
    img_tensor = TF.to_tensor(img)[:3]
    return img_tensor

def init_parameters(N, device):
    # Random gaussians
    bd = 2.3
    be = 0.02

    mu = bd * (torch.rand(N, 3, device=device,) - 0.5)
    scales = be * (torch.ones(N, 3, device=device,))
    d = 3
    cols = torch.ones(N, d, device=device,) * 0.5

    u = torch.rand(N, 1, device=device)
    v = torch.rand(N, 1, device=device)
    w = torch.rand(N, 1, device=device)

    quats = torch.cat(
        [
            torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
            torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
            torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
            torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
        ],
        -1,
    )
    quats = quats.to(device=device)
    opcs = torch.rand((N,), device=device,)

    return {
        'mu': mu, 'scales': scales, 'quats': quats, 'cols': cols, 'opcs': opcs
    }

def init_scene(root: str):
    scene = readCamerasFromTransforms(
        root,
        'transforms_train.json',
        False
    )

    return scene

def get_view(scene: List[CameraInfo], W, H, queue: List):
    if not queue:
        queue = copy.copy(scene)
        random.shuffle(queue)

    view = queue.pop()

    gt_image = TF.resize(image_path_to_tensor(view.image_path), (H, W)).permute(1,2,0)
    viewmat = torch.from_numpy(view.w2c).to(dtype=torch.float)
    focal = 0.5 * float(W) / math.tan(0.5 * view.FovX)
    camera = {'gt_image': gt_image, 'viewmat': viewmat, 'focal': focal, 'H': H, 'W': W}

    return camera

def get_test_view(root, index, W, H):
    scene = readCamerasFromTransforms(
        root,
        'transforms_test.json',
        False
    )

    view = scene[index]

    gt_image = TF.resize(image_path_to_tensor(view.image_path), (H, W)).permute(1,2,0)
    viewmat = torch.from_numpy(view.w2c).to(dtype=torch.float)
    focal = 0.5 * float(W) / math.tan(0.5 * view.FovX)
    camera = {'gt_image': gt_image, 'viewmat': viewmat, 'focal': focal, 'H': H, 'W': W}

    return camera

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N = 10_000
    W = H = 256

    params = init_parameters(N, device)   

    renderer = Renderer(params=params, device=device)

    # Load scene
    scene = init_scene(root='./nerf_example_data/nerf_synthetic/drums/drums')

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(params=renderer.parameters(), lr=1e-3)

    test_view = get_test_view(root='./nerf_example_data/nerf_synthetic/drums/drums', index=0, W=W, H=H)

    plt.ion()
    figure, (ax1, ax2) = plt.subplots(1,2)
    im1 = ax1.matshow(torch.rand(H, W))
    im2 = ax2.matshow(test_view['gt_image'])

    frames = []
    camera_queue = []
    for iter in tqdm(range(5_000)):
        view = get_view(scene, W, H, camera_queue)

        optimizer.zero_grad()

        pred = renderer(view)
        pred[pred.isnan() | pred.isinf()] = 0.

        with torch.no_grad():
            pred_test = renderer(test_view)

            im1.set_data(pred_test.detach().cpu())

            if iter % 10 == 0:
                frames.append((pred_test.detach().cpu().numpy() * 255).astype(np.uint8))

            # im2.set_data(view['gt_image'].detach().cpu())

        figure.canvas.draw()
        figure.canvas.flush_events()
        time.sleep(0.1)

        loss = criterion(pred, view['gt_image'])
        
        loss.backward()

        # torch.nn.utils.clip_grad_value_(renderer.parameters(), clip_value=10.0)
        for param in renderer.parameters():
            param.grad[param.grad.isnan()] = 0.
            param.grad[param.grad.isinf()] = 0.

        optimizer.step()

        print(f'Iter: {iter}, Loss: {loss.item()}, Grad. Norms: {[p.abs().norm().item() for p in renderer.parameters()]}')

    # save them as a gif with PIL
    frames = [Image.fromarray(frame) for frame in frames]
    out_dir = os.path.join(os.getcwd(), "renders")
    os.makedirs(out_dir, exist_ok=True)
    frames[0].save(
        f"{out_dir}/training.gif",
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=5,
        loop=0,)

    plt.matshow(pred.detach().cpu())
    plt.show()

if __name__ == '__main__':
    main()
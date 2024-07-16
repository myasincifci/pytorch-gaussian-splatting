import time 
import math
import copy
import random
import torch
import matplotlib.pyplot as plt
from renderer import Renderer
from tqdm import tqdm
from utils import readCamerasFromTransforms
from PIL import Image
import torchvision.transforms as T

def image_path_to_tensor(image_path):
    img = Image.open(image_path)
    transform = T.ToTensor()
    img_tensor = transform(img)[:3]
    return img_tensor

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N = 10_000
    W = H = 256


    # Random gaussians
    bd = 2.
    be = 0.01

    mu = bd * (torch.rand(N, 3, device=device,) - 0.5)
    scales = be * (torch.rand(N, 3, device=device,))
    d = 3
    cols = torch.rand(N, d, device=device,)

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

    params = {
        'mu': mu, 'scales': scales, 'quats': quats, 'cols': cols, 'opcs': opcs
    }   

    renderer = Renderer(params=params, device=device)
    resize = T.Resize(256)

    # Create GT image
    cameras = readCamerasFromTransforms(
        './nerf_example_data/nerf_synthetic/drums/drums',
        'transforms_train.json',
        False
    )
    viewmat = torch.from_numpy(cameras[10].w2c).to(dtype=torch.float).to(renderer.device)
    focal = 0.5 * float(W) / math.tan(0.5 * cameras[10].FovX)
    camera = {'viewmat': viewmat, 'focal': focal, 'H': H, 'W': W}
    gt_image = image_path_to_tensor(cameras[10].image_path)
    gt_image = resize(gt_image).permute(1,2,0)

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(params=renderer.parameters(), lr=1e-2)

    pred = renderer(camera, gt_image)

    plt.ion()
    figure, (ax1, ax2) = plt.subplots(1,2)
    im1 = ax1.matshow(pred.detach().cpu())
    im2 = ax2.matshow(gt_image.detach().cpu())

    camera_queue = []
    for iter in tqdm(range(100_000)):
        if not camera_queue:
            camera_queue = copy.copy(cameras)
            random.shuffle(camera_queue)

        camera_ = camera_queue.pop()
        viewmat = torch.from_numpy(camera_.w2c).to(dtype=torch.float).to(renderer.device)
        gt_image = image_path_to_tensor(camera_.image_path)
        gt_image = resize(gt_image).permute(1,2,0)
        focal = 0.5 * float(W) / math.tan(0.5 * camera_.FovX)
        camera = {'viewmat': viewmat, 'focal': focal, 'H': H, 'W': W}

        optimizer.zero_grad()

        pred = renderer(camera, gt_image)
        pred[pred.isnan() | pred.isinf()] = 0.

        im1.set_data(pred.detach().cpu())
        im2.set_data(gt_image.detach().cpu())

        figure.canvas.draw()
        figure.canvas.flush_events()
        time.sleep(0.1)

        loss = criterion(pred, gt_image)
        
        loss.backward()

        torch.nn.utils.clip_grad_value_(renderer.parameters(), clip_value=10.0)
        for param in renderer.parameters():
            param.grad[param.grad.isnan()] = 0.
            param.grad[param.grad.isinf()] = 0.

        optimizer.step()

        print(f'Iter: {iter}, Loss: {loss.item()}, Grad. Norms: {[p.abs().norm().item() for p in renderer.parameters()]}')

    plt.matshow(pred.detach().cpu())
    plt.show()

if __name__ == '__main__':
    main()
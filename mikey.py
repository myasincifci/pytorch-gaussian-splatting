import time 
import math
import torch
import matplotlib.pyplot as plt
from renderer import Renderer
from tqdm import tqdm
from torchvision.io import read_image

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N = 10_000
    W = H = 256

    mu = (torch.rand((N, 3), device=device) - 0.5) * 8.
    mu[:,2] = torch.rand((N,)) * 0.001
    scales = torch.rand((N, 3), device=device) * 0.1
    quats = torch.rand((N, 4), device=device)
    cols = torch.rand((N, 3), device=device) 
    opcs = torch.rand((N), device=device)

    params = {
        'mu': mu, 'scales': scales, 'quats': quats, 'cols': cols, 'opcs': opcs
    }            

    renderer = Renderer(params=params, device=device)
    
    # Create GT image
    gt_image = read_image('./mikey_cropped.jpg').permute(1,2,0) / 255
    fov_x = math.pi / 2.0 # Angle of the camera frustum 90°
    focal = 0.5 * float(W) / math.tan(0.5 * fov_x) # Distance to Image Plane
    viewmat = torch.eye(4, device=device)
    viewmat[:3,3] = torch.tensor([0,0,-4])
    camera = {'viewmat': viewmat, 'focal': focal, 'H': H, 'W': W}

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(params=renderer.parameters(), lr=1e-2)

    pred = renderer(camera)

    plt.ion()
    figure, ax = plt.subplots()
    im1 = ax.matshow(pred.detach().cpu())

    for iter in tqdm(range(1_000)):
        optimizer.zero_grad()

        pred = renderer(camera)

        im1.set_data(pred.detach().cpu())
        figure.canvas.draw()
        figure.canvas.flush_events()
        time.sleep(0.1)

        loss = criterion(pred, gt_image)
        
        loss.backward()

        # torch.nn.utils.clip_grad_value_(renderer.parameters(), clip_value=1.0)
        # for param in renderer.parameters():
        #     param.grad[param.grad.isnan()] = 0.

        optimizer.step()

        print(f'Iter: {iter}, Loss: {loss.item()}, Grad. Norms: {[p.abs().norm().item() for p in renderer.parameters()]}')

    plt.matshow(pred.detach().cpu())
    plt.show()

if __name__ == '__main__':
    main()
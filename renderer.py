from typing import Dict
from tqdm import tqdm

import torch
from torch import nn

class Renderer(nn.Module):
    def __init__(self, params: Dict, tile_size: int=16, device: str='cpu') -> None:
        super().__init__()

        self.device = device
        self.params: nn.ParameterDict = nn.ParameterDict(params)
        self.N = len(params['mu'])
        self.tile_size = tile_size

    def forward(self, camera: Dict, gt: torch.Tensor):
        # Project 3D-Gaussians to 2D
        (
            mu_2d,
            cov_2d,
            z,
            cols,
            opcs
        ) = self._project_gaussians(
            mu=self.params['mu'],
            scales=self.params['scales'],
            fx=camera['focal'],
            fy=camera['focal'],
            cx=camera['W']/2,
            cy=camera['H']/2,
            viewmat=camera['viewmat'],
            H=camera['H'],
            W=camera['W']
        )

        # Sort to tiles
        tile_map = self._tile_gaussians(mu_2d, cov_2d, H=camera['H'], W=camera['W'])

        # Rasterize each tile
        img = torch.zeros((camera['H'], camera['W'], 3))
        for y in tqdm(range(len(tile_map))):
            for x in range(len(tile_map[0])):
                if tile_map[y][x]:
                    self._rasterize_tile_fast(
                        tile_map,
                        mu_2d=mu_2d, cov_2d=cov_2d, cols=cols, opcs=opcs,
                        x_coord=x, y_coord=y,
                        H=camera['H'], W=camera['W'],
                        out_img=img
                    )

        return img

    ### Project ##################################################################
    def _P(self, f_x, f_y, h, w, n, f):
        P = torch.tensor([
            [2.*f_x/w, 0., 0., 0.],
            [0., 2.*f_y/h, 0., 0.],
            [0., 0., (f+n)/(f-n), -2*f*n/(f-n)],
            [0., 0., 1., 0.],
        ], device=self.device)

        return P

    def _J(self, f_x, f_y, t_x, t_y, t_z):
        N = len(t_x)
        J = torch.zeros((N,2,3), device=self.device)
        J[:,0,0] = f_x/t_z
        J[:,1,1] = f_y/t_z
        J[:,0,2] = -f_x*t_x/t_z**2
        J[:,1,2] = -f_y*t_y/t_z**2

        return J

    def _quat_to_rot(self, quaternion):
        N = quaternion.shape[0]
        x, y, z, w = quaternion[:,0].clone() ,quaternion[:,1].clone() ,quaternion[:,2].clone() ,quaternion[:,3].clone(),

        R = torch.empty((N,3,3),device=self.device)
        
        R[:,0,0] = 1-2*(y**2+z**2)
        R[:,0,1] = 2*(x*y-w*z)
        R[:,0,2] = 2*(x*z+w*y)

        R[:,1,0] = 2*(x*y+w*z)
        R[:,1,1] = 1-2*(x**2-z**2)
        R[:,1,2] = 2*(y*z-w*x)

        R[:,2,0] = 2*(x*z+w*y)
        R[:,2,1] = 2*(y*z+w*x)
        R[:,2,2] = 1-2*(x**2+y**2)

        return R

    def _project_gaussians(
        self,
        mu,
        scales,
        fx, 
        fy, 
        cx, 
        cy, 
        viewmat,
        H,
        W,
        clip_thresh=0.01,
    ):
        # Project Means
        t = viewmat @ torch.cat((mu.T, torch.ones(1, self.N, device=self.device)), dim=0)
        t_ = self._P(fx, fy, H, W, clip_thresh, 10) @ t

        xys = torch.vstack((
            (W * t_[0] / t_[3]) / 2 + cx, 
            (H * t_[1] / t_[3]) / 2 + cy,
        )).T
        depths = t_[2]

        # Scale + Rot. to Cov.
        R = self._quat_to_rot(self.params['quats']); S = torch.cat([torch.diag(s)[None] for s in self.params['scales']])
        RS =  R @ S
        Sigma = RS @ RS.permute(0,2,1)

        # Project Cov
        J_ = self._J(fx, fy, t[0], t[1], t[2])
        R_cw = viewmat[:3,:3]
        covs = J_ @ R_cw @ Sigma @ R_cw.T @ J_.permute((0,2,1))

        _, ind = torch.sort(depths)

        return xys[ind], covs[ind], depths[ind], self.params['cols'][ind], self.params['opcs'][ind]
    
    ### Tile #######################################################################
    def _get_eigenvalues(self, cov):
        a = cov[:,0,0]; b = cov[:,0,1]; d = cov[:,1,1]

        A = torch.sqrt(a*a - 2*a*d + 4*b**2 + d*d)
        B = a + d

        return 0.5 * (A + B), 0.5 * (-A + B)

    def _get_radii(self, cov):
        eigs = self._get_eigenvalues(cov)
        l1, l2 = eigs[0], eigs[1]
        s = 3
        return s * torch.sqrt(l1), s * torch.sqrt(l2)

    def _get_box(self, mu, cov):
        N = len(mu)

        r1, r2 = self._get_radii(cov)

        B = torch.empty((N, 4, 2))

        B[:,0,0] = mu[:,0] - r1; B[:,0,1] = mu[:,1] + r2
        B[:,1,0] = mu[:,0] + r1; B[:,1,1] = mu[:,1] + r2
        B[:,3,0] = mu[:,0] - r1; B[:,2,1] = mu[:,1] - r2
        B[:,2,0] = mu[:,0] + r1; B[:,3,1] = mu[:,1] - r2
        
        return B

    def _get_orientation(self, cov):
        a = cov[:,0,0]; b = cov[:,0,1]; c = cov[:,1,1]
        eigs = self._get_eigenvalues(cov)
        l1 = eigs[0]

        theta = torch.zeros_like(a)
        theta[(b == 0) & (a >= c)] = torch.pi/2
        theta[b != 0] = torch.atan2(l1 - a, b)

        return theta

    def _get_rotation(self, cov):
        theta = self._get_orientation(cov)

        cos = torch.cos(theta)
        sin = torch.sin(theta)

        R = torch.empty((len(cos),2,2))
        R[:,0,0] = cos
        R[:,0,1] = -sin
        R[:,1,0] = sin
        R[:,1,1] = cos

        return R

    def _get_bounding_boxes(self, xys, covs):
        rot = self._get_rotation(covs)

        box = self._get_box(xys, covs)
        box_mean = box.mean(dim=1, keepdim=True)

        rot_box = (rot @ (box - box_mean).permute((0,2,1))).permute((0,2,1)) + box_mean

        return rot_box

    def _tile_gaussians(self, xys, covs, H, W):
        assert H % self.tile_size == 0
        assert W % self.tile_size == 0
        
        # Compute Bounding-Boxes
        bbs = self._get_bounding_boxes(xys, covs)
        bbs[bbs.isnan()] = -1.

        tile_map = [[[] for tw in range(W//self.tile_size)] for th in range(H//self.tile_size)]

        minmax = torch.cat(((bbs.amin(dim=1) / self.tile_size).floor()[:,:,None], (bbs.amax(dim=1) / self.tile_size).ceil()[:,:,None]), dim=2)
        minmax[:,0].clamp_(0, W//self.tile_size - 1)
        minmax[:,1].clamp_(0, H//self.tile_size - 1)

        for i, g in enumerate(minmax.to(torch.long)):
            x_min, x_max = g[0]
            y_min, y_max = g[1]

            for x in range(x_min, x_max+1):
                for y in range(y_min, y_max+1):
                    tile_map[y][x].append(i)

        return tile_map
    
    ### Rasterize ##################################################################
    def _inv_2d(self, A: torch.Tensor):
        A_inv = A.new_empty(A.shape)
        A_inv[0,0] = A[1,1]
        A_inv[0,1] = -A[0,1]
        A_inv[1,0] = -A[1,0]
        A_inv[1,1] = A[0,0]

        A_inv *= 1/(A[0,0]*A[1,1]-A[0,1]*A[1,0])

        return A_inv

    def _g_fast(self, x, m, S):
        ''' x: (h*w, 2) matrix
            m: (2, 1) mean
            S: (2, 2) cov matrix
        '''
        x = x.T.view(-1, 1, 2)
        m = m.view(1, 1, 2)

        S_inv = torch.inverse(S + 1e-6 * torch.eye(S.size(-1)).cuda()) # self._inv_2d(S)

        x_m = x - m

        A = x_m @ S_inv

        result = torch.exp(-0.5 * A @ x_m.permute(0,2,1))

        return result
    
    def _rasterize_tile_fast(
        self,
        tile_map,
        mu_2d, cov_2d, cols, opcs,
        x_coord, y_coord,
        H, W,
        out_img
    ):
        x_min, x_max = x_coord*self.tile_size, (x_coord+1)*self.tile_size
        y_min, y_max = y_coord*self.tile_size, (y_coord+1)*self.tile_size
        
        x, y = torch.meshgrid(torch.arange(x_min, x_max, device=self.device), torch.arange(y_min, y_max, device=self.device))
        x = x.reshape(1,-1); y = y.reshape(1, -1)

        pixels_xy = torch.cat((x.reshape(1,-1),y.reshape(1,-1)), dim=0)

        tile = tile_map[y_coord][x_coord]

        # mu_2d = mu_2d.clamp(min=0, max=float(H))

        C = cols[None, tile].clamp(min=0.0001, max=1.)
        O = torch.nn.functional.sigmoid(opcs[None, tile])

        P = torch.vmap(self._g_fast, in_dims=0)(
            pixels_xy[None].expand((len(tile), pixels_xy.shape[0], pixels_xy.shape[1])), 
            mu_2d[tile], cov_2d[tile]).permute((1,0,2,3)).squeeze(dim=(2,3))

        P_ = torch.nan_to_num(P)
        O = torch.nn.functional.sigmoid(O)

        A = O * P_
        A[A<0.001] = 0.
        A = A.clip(max=0.99)

        D = (1 - A)
        D = torch.cat((torch.ones(A.shape[0], 1, device=mu_2d.device), D[:,:-1]), dim=1).contiguous()
        D = D
        D = D.cumprod(dim=1)

        RGB = C * A[:,:,None]
        RGB = RGB * D[:,:,None]

        RGB = RGB.sum(dim=1).clamp(min=0.0001, max=1.)
        RGB_ = RGB.view(self.tile_size, self.tile_size, 3).permute(1,0,2)

        out_img[y_min:y_max, x_min:x_max] = RGB_
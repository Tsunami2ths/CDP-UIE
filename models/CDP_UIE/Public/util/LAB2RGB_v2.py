import torch

class Lab2RGB():
    def __init__(self, useGPU=True):
        self.lab_to_inorm = torch.tensor([
        # l_in  a_in     b_in
        [100.0 , 0.0,      0.0],      # l_n
        [0.0,  184.4161,   0.0],      # a_n
        [0.0,  0.0,      202.3354],   # b_n
        ])
        if useGPU and torch.cuda.is_available():
            self.lab_to_inorm = self.lab_to_inorm.cuda()

        self.unkown_const2 = torch.tensor([0.0, -86.1830, -107.8573])
        if useGPU and torch.cuda.is_available():
            self.unkown_const2 = self.unkown_const2.cuda()

        self.lab_to_fxfyfz = torch.tensor([
        #   fx      fy        fz
        [1/116.0, 1/116.0,  1/116.0], # l
        [1/500.0,     0.0,      0.0], # a
        [    0.0,     0.0, -1/200.0], # b
        ])
        if useGPU and torch.cuda.is_available():
            self.lab_to_fxfyfz = self.lab_to_fxfyfz.cuda()

        self.unkown_const = torch.tensor([16.0/116, 16.0/116, 16.0/116])
        if useGPU and torch.cuda.is_available():
            self.unkown_const = self.unkown_const.cuda()

        self.white = torch.tensor([0.950456, 1.0, 1.088754])
        if useGPU and torch.cuda.is_available():
            self.white = self.white.cuda()

        self.xyz_to_rgb = torch.tensor([
        #     r           g          b
        [ 3.2404542, -0.9692660,  0.0556434], # x
        [-1.5371385,  1.8760108, -0.2040259], # y
        [-0.4985314,  0.0415560,  1.0572252], # z
        ]) # inverse of M_{RGB2XYZ}^T
        if useGPU and torch.cuda.is_available():
            self.xyz_to_rgb = self.xyz_to_rgb.cuda()

        # print('init done')
    def lab_inverse_norm(self, lab_n):
        lab_in = torch.matmul(lab_n, self.lab_to_inorm) + self.unkown_const2

        return lab_in

    def lab_to_rgb(self, lab, needInverseNorm=True):
        if len(lab.shape) == 4:
            lab_pixels = lab.permute(0, 2, 3, 1)  # b,c,h,w -> b,h,w,c
        elif len(lab.shape) == 3:
            lab_pixels = lab.permute(1, 2, 0)  # c,h,w -> h,w,c
        else:
            print("len(lab.shape) is 3 or 4, quit function lab_to_rgb")
            return

        if needInverseNorm:
            # inverse norm from [0,1] to [[0,100], [-86.18,98.23], [-107.86,94.48]]
            lab_pixels = self.lab_inverse_norm(lab_pixels)

        fxfyfz_pixels = torch.matmul(lab_pixels, self.lab_to_fxfyfz) + self.unkown_const

        epsilon = 6. / 29
        linear_mask = (fxfyfz_pixels <= epsilon).to(dtype=torch.float)
        exponential_mask = (fxfyfz_pixels > epsilon ).to(dtype=torch.float)

        xyz_pixels = (3 * epsilon ** 2 * (fxfyfz_pixels - 4. / 29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask
        xyz_pixels = torch.multiply(xyz_pixels, self.white)
        rgb_pixels = torch.matmul(xyz_pixels, self.xyz_to_rgb)
        rgb_pixels = torch.clip(rgb_pixels, 0.0, 1.0)
        linear_mask = (rgb_pixels <= 0.0031308).to(dtype=torch.float)
        exponential_mask = (rgb_pixels > 0.0031308).to(dtype=torch.float)
        srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask


        if len(lab.shape) == 4:
            return srgb_pixels.permute(0,3,1,2) # b,h,w,c -> b,c,h,w
        elif len(lab.shape) == 3:
            return srgb_pixels.permute(2,0,1) # h,w,c -> c,h,w

    # lab: [-1,1]
    # rgb: [-1,1]
    def labn12p1_to_rgbn12p1(self, lab):
        #[-1,1] to [0,1]
        lab_021 = (lab + 1)/2
        rgb_021 = self.lab_to_rgb(lab_021)

        #[0,1] to [-1,1]
        return 2*rgb_021 - 1


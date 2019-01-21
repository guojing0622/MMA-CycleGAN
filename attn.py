import torch
import torch.nn as nn

class Multi_Attn(nn.Module):
    "Multi Attention Layer"
    def __init__(self, sampling='down', in_dim=64, pool_ksize=2, down_ch=1, dconv_dim = 1):
        super(Multi_Attn, self).__init__()
        self.dim = in_dim
        self.pool_ksize = pool_ksize
        self.sampling = sampling

        # pool methods
        self.max_pool = nn.MaxPool2d(pool_ksize, pool_ksize)
        self.avg_pool = nn.AvgPool2d(pool_ksize, pool_ksize)
        self.lp_pool = nn.LPPool2d(2,pool_ksize,pool_ksize)
        self.pools = [self.max_pool, self.avg_pool, self.lp_pool]

        self.xin_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim*2,
                                kernel_size= 4, stride=2, padding=1)
        self.yin_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim*2,
                                kernel_size= 4, stride=2, padding=1)

        self.q_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim//down_ch, kernel_size= 1)
        self.k_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim//down_ch, kernel_size= 1)
        self.v1_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim, kernel_size= 1)
        self.v2_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim, kernel_size= 1)

        self.out_deconv1_1 = nn.ConvTranspose2d(in_channels = in_dim, out_channels = in_dim * dconv_dim,
                                              kernel_size=4, stride=2, padding=1)
        self.out_deconv1_2 = nn.ConvTranspose2d(in_channels = in_dim, out_channels = in_dim * dconv_dim,
                                              kernel_size=4, stride=2, padding=1)

        self.out_deconv2_1 = nn.ConvTranspose2d(in_channels = in_dim, out_channels = in_dim * dconv_dim,
                                              kernel_size=4, stride=2, padding=1)
        self.out_deconv2_2 = nn.ConvTranspose2d(in_channels=in_dim, out_channels=in_dim * dconv_dim,
                                              kernel_size=4, stride=2, padding=1)

        self.softmax1 = nn.Softmax(dim=-1)
        self.softmax2 = nn.Softmax(dim=1)

        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.a1 = nn.Parameter(torch.ones(1))
        self.a2 = nn.Parameter(torch.ones(1))
        self.a3 = nn.Parameter(torch.ones(1))

        self.b1 = nn.Parameter(torch.ones(1))
        self.b2 = nn.Parameter(torch.ones(1))
        self.b3 = nn.Parameter(torch.ones(1))



    def forward(self, x, y):
        batch_size, num_c, w, h = x.size()
        if self.sampling == 'down':
            x_pro = self.xin_conv(x)
            y_pro = self.yin_conv(y)
        else:
            x_pro = x
            y_pro = y

        # ***
        x_out = []
        y_out = []
        # ***
        for i in range(len(self.pools)):
            x_pool = self.pools[i](x)
            y_pool = self.pools[i](y)
            _, _, patch_w, patch_h = x_pool.size()
            q_proj = self.q_conv(y_pool).view(batch_size, -1, patch_w*patch_h)
            k_proj = self.k_conv(x_pool).view(batch_size, -1, patch_w*patch_h)
            v1_proj = self.v1_conv(x_pool).view(batch_size, -1, patch_w*patch_h)
            v2_proj = self.v2_conv(y_pool).view(batch_size, -1, patch_w*patch_h)

            energy = torch.bmm(q_proj.permute(0, 2, 1), k_proj)   #qT * k

            attention1 = self.softmax1(energy)   # k effects on qi
            attention2 = self.softmax2(energy)   # q effects on ki

            outx_head = torch.bmm(v1_proj, attention2)   # new x
            outy_head = torch.bmm(v2_proj, attention1.permute(0, 2, 1)) # new y

            x_out.append(outx_head.view(batch_size, num_c, patch_w, patch_h))
            y_out.append(outy_head.view(batch_size, num_c, patch_w, patch_h))

        '''
            if i == 0:
                out1 = out1_head.view(batch_size, num_c, patch_w, patch_h)
                out2 = out2_head.view(batch_size, num_c, patch_w, patch_h)
            else:
                out1 += out1_head.view(batch_size, num_c, patch_w, patch_h)
                out2 += out2_head.view(batch_size, num_c, patch_w, patch_h)

        out1 = out1 / len(self.pools)
        out2 = out2 / len(self.pools)

        '''
        a = self.a1*x_out[0] + self.a2*x_out[1] + self.a3*x_out[2]
        b = self.b1*y_out[0] + self.b2*y_out[1] + self.b3*y_out[2]
        out1 = a / len(self.pools)
        out2 = b / len(self.pools)

        out1 = self.out_deconv1_1(out1)
        if (self.sampling == 'up') and (self.pool_ksize==4):
            out1 = self.out_deconv1_2(out1)

        out2 = self.out_deconv2_1(out2)
        if (self.sampling == 'up') and (self.pool_ksize==4):
            out2 = self.out_deconv2_2(out2)

        out1 = self.gamma1 * out1 + x_pro
        out2 = self.gamma2 * out2 + y_pro

        return out1, out2

import torch.nn as nn
import torch
import torch.nn.functional as F

class TimeEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super(TimeEmb, self).__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim, dim * 16)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(dim * 16, dim * 16)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = t.shape[-1]
        return self.linear2(self.act(self.linear1(t)))

class PosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super(PosEmb, self).__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.SiLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)  # in_channels = out_channels (up) + out_channels (skip)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Down, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))

class UNet(nn.Module):  
    def __init__(
            self, 
            in_channels: int,
            cond_channels: int,
            base_channels: int,
            ) -> None:
        super(UNet, self).__init__()
        self.time_mlp = nn.Sequential(
                    PosEmb(base_channels),
                    TimeEmb(base_channels),
                )

        self.inc = DoubleConv(in_channels + cond_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)
        self.bottleneck = DoubleConv(base_channels * 16, base_channels * 16)
        self.up4 = Up(base_channels * 16, base_channels * 8)
        self.up3 = Up(base_channels * 8, base_channels * 4)
        self.up2 = Up(base_channels * 4, base_channels * 2)
        self.up1 = Up(base_channels * 2, base_channels)
        self.outc = nn.Conv2d(base_channels, in_channels, kernel_size=1)
        nn.init.zeros_(self.outc.weight)
        nn.init.zeros_(self.outc.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, timesteps: torch.Tensor, debug: bool = False) -> torch.Tensor:

        cond = F.interpolate(cond, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, cond], dim=1)
        t_emb = self.time_mlp(timesteps)

        x1 = self.inc(x)
        if debug:
            print("after inc: shape=", x1.shape, "channels=", x1.shape[1])
        x2 = self.down1(x1)
        if debug:
            print("after down1: shape=", x2.shape, "channels=", x2.shape[1])
        x3 = self.down2(x2)
        if debug:
            print("after down2: shape=", x3.shape, "channels=", x3.shape[1])
        x4 = self.down3(x3)
        if debug:
            print("after down3: shape=", x4.shape, "channels=", x4.shape[1])
        x5 = self.down4(x4)
        if debug:
            print("after down4: shape=", x5.shape, "channels=", x5.shape[1])
        x5 = x5 + t_emb[:, :, None, None]
        if debug:
            print("after add t_emb: shape=", x5.shape, "channels=", x5.shape[1])

        x = self.bottleneck(x5)
        if debug:
            print("after bottleneck: shape=", x.shape, "channels=", x.shape[1])
        x = self.up4(x, x4) # x4 is the skip connection from the downsampling path
        if debug:
            print("after up4: shape=", x.shape, "channels=", x.shape[1])
        x = self.up3(x, x3)
        if debug:
            print("after up3: shape=", x.shape, "channels=", x.shape[1])
        x = self.up2(x, x2)
        if debug:
            print("after up2: shape=", x.shape, "channels=", x.shape[1])
        x = self.up1(x, x1)
        if debug:
            print("after up1: shape=", x.shape, "channels=", x.shape[1])
        x = self.outc(x)
        if debug:
            print("after outc: shape=", x.shape, "channels=", x.shape[1])
        return x
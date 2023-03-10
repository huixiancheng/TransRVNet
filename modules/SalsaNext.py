import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.swin_transformer import SwinTransformer


class DyReLU(nn.Module):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLU, self).__init__()
        self.channels = channels
        self.k = k
        self.conv_type = conv_type
        assert self.conv_type in ['1d', '2d']

        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, 2*k)
        self.sigmoid = nn.Sigmoid()

        self.register_buffer('lambdas', torch.Tensor([1.]*k + [0.5]*k).float())
        self.register_buffer('init_v', torch.Tensor([1.] + [0.]*(2*k - 1)).float())

    def get_relu_coefs(self, x):
        theta = torch.mean(x, dim=-1)
        if self.conv_type == '2d':
            theta = torch.mean(theta, dim=-1)
        theta = self.fc1(theta)
        theta = self.relu(theta)
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        return theta

    def forward(self, x):
        raise NotImplementedError

class DyReLUB(DyReLU):
    def __init__(self, channels, reduction=8, k=2, conv_type='2d'):
        super(DyReLUB, self).__init__(channels, reduction, k, conv_type)
        self.fc2 = nn.Linear(channels // reduction, 2*k*channels)

    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x)

        relu_coefs = theta.view(-1, self.channels, 2*self.k) * self.lambdas + self.init_v

        if self.conv_type == '1d':
            # BxCxL -> LxBxCx1
            x_perm = x.permute(2, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # LxBxCx2 -> BxCxL
            result = torch.max(output, dim=-1)[0].permute(1, 2, 0)

        elif self.conv_type == '2d':
            # BxCxHxW -> HxWxBxCx1
            x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # HxWxBxCx2 -> BxCxHxW
            result = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)
        return result

class sa_layer(nn.Module):
    """Constructs a Channel Spatial Group module.
    Args:
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, groups=32):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = nn.Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = nn.Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = nn.Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = nn.Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out

class CAM(nn.Module):
  def __init__(self, inplanes):
    super(CAM, self).__init__()
    self.inplanes = inplanes
    self.pool = nn.MaxPool2d(7, 1, 3)
    self.squeeze = nn.Conv2d(inplanes, inplanes // 16,
                             kernel_size=1, stride=1)
    self.squeeze_bn = nn.BatchNorm2d(inplanes // 16)
    self.relu = nn.ReLU(inplace=True)
    self.unsqueeze = nn.Conv2d(inplanes // 16, inplanes,
                               kernel_size=1, stride=1)
    self.unsqueeze_bn = nn.BatchNorm2d(inplanes)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    # 7x7 pooling
    y = self.pool(x)
    # squeezing and relu
    y = self.relu(self.squeeze_bn(self.squeeze(y)))
    # unsqueezing
    y = self.sigmoid(self.unsqueeze_bn(self.unsqueeze(y)))
    # attention
    return y * x


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate,
                 pooling=True, drop_out=True, cam=False, dy=False, dy1=False, bigpool=False, sca=False):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        self.cam = cam
        self.sca = sca

        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1))
        if dy:
            self.act1 = DyReLUB(out_filters)
        else:
            self.act1 = nn.PReLU()

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(out_filters)
        if dy:
            self.act2 = DyReLUB(out_filters)
        else:
            self.act2 = nn.PReLU()

        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(out_filters)
        if dy:
            self.act3 = DyReLUB(out_filters)
        else:
            self.act3 = nn.PReLU()

        self.conv4 = nn.Conv2d(out_filters, out_filters, kernel_size=(3, 3), dilation=(2, 2), padding=(2, 2))
        self.bn3 = nn.BatchNorm2d(out_filters)
        if dy:
            self.act4 = DyReLUB(out_filters)
        else:
            self.act4 = nn.PReLU()

        self.conv5 = nn.Conv2d(out_filters*3, out_filters, kernel_size=(1, 1))
        self.bn4 = nn.BatchNorm2d(out_filters)
        if dy1:
            self.act5 = DyReLUB(out_filters)
        else:
            self.act5 = nn.PReLU()

        if self.sca:
            self.sanet = sa_layer(out_filters)
        if pooling:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            if cam:
                self.camm = CAM(out_filters)
            if bigpool:
                self.pool = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)
            else:
                self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(x)
        resA = self.bn1(resA)
        resA1 = self.act2(resA)

        resA = self.conv3(resA1)
        resA = self.bn2(resA)
        resA2 = self.act3(resA)

        resA = self.conv4(resA2)
        resA = self.bn3(resA)
        resA3 = self.act4(resA)

        concat = torch.cat((resA1,resA2,resA3),dim=1)
        resA = self.conv5(concat)
        resA = self.bn4(resA)
        if self.sca:
            resA = self.sanet(resA)
        resA = self.act5(resA)
        resA = shortcut + resA

        if self.pooling:
            if self.drop_out:
                resB = self.dropout(resA)
                if self.cam:
                    resB = self.camm(resB)
            else:
                if self.cam:
                    resB = self.camm(resA)
                else:
                    resB = resA
            resB = self.pool(resB)
            return resB, resA
        else:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            return resB

class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, drop_out=True, dy=False, dy1=False):
        super(UpBlock, self).__init__()
        self.drop_out = drop_out
        self.in_filters = in_filters
        self.out_filters = out_filters

        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        self.conv1 = nn.Conv2d(in_filters, out_filters, (3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(out_filters)
        if dy:
            self.act1 = DyReLUB(out_filters)
        else:
            self.act1 = nn.PReLU()

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(out_filters)
        if dy:
            self.act2 = DyReLUB(out_filters)
        else:
            self.act2 = nn.PReLU()

        self.conv3 = nn.Conv2d(out_filters, out_filters, (3, 3), dilation=(2, 2), padding=(2, 2))
        self.bn3 = nn.BatchNorm2d(out_filters)
        if dy:
            self.act3 = DyReLUB(out_filters)
        else:
            self.act3 = nn.PReLU()

        self.conv4 = nn.Conv2d(out_filters*3, out_filters, kernel_size=(1, 1))
        self.bn4 = nn.BatchNorm2d(out_filters)
        if dy1:
            self.act4 = DyReLUB(out_filters)
        else:
            self.act4 = nn.PReLU()

        self.dropout3 = nn.Dropout2d(p=dropout_rate)

    def forward(self, x, skip):
        upA = nn.PixelShuffle(2)(x)
        if self.drop_out:
            upA = self.dropout1(upA)

        upB = torch.cat((upA, skip),dim=1)
        if self.drop_out:
            upB = self.dropout2(upB)

        upE = self.conv1(upB)
        upE = self.bn1(upE)
        upE1 = self.act1(upE)

        upE = self.conv2(upE1)
        upE = self.bn2(upE)
        upE2 = self.act2(upE)

        upE = self.conv3(upE2)
        upE = self.bn3(upE)
        upE3 = self.act3(upE)

        concat = torch.cat((upE1,upE2,upE3),dim=1)
        upE = self.conv4(concat)
        upE = self.bn4(upE)
        upE = self.act4(upE)
        if self.drop_out:
            upE = self.dropout3(upE)
        return upE

class ConvBNLeakyReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=1, stride=1, padding=0, dilation=1, dy=False, *args, **kwargs):
        super(ConvBNLeakyReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size=ks,
                stride=stride,
                padding=padding,
                dilation=dilation
        )
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

class MCRF(nn.Module):
    def __init__(self, C0=5, C1=24):
        super(MCRF, self).__init__()
        self.conv0 = ConvBNLeakyReLU(C0, C1, ks=1)
        self.conv1 = ConvBNLeakyReLU(C0, C1, ks=3, padding=1)
        self.conv2 = ConvBNLeakyReLU(C0, C1, ks=5, padding=2)
        self.conv3 = ConvBNLeakyReLU(C0, C1, ks=7, padding=3)
        self.cat1 = ConvBNLeakyReLU(C1*3, C1, ks=3, padding=1)

    def forward(self, x):
        add1 = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        xc = torch.cat((x1, x2, x3), dim=1)
        x = self.cat1(xc)
        x = x + add1
        return x


class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters, sca=False):
        super(ResContextBlock, self).__init__()
        self.sca = sca
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1))
        self.act1 = nn.PReLU()

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(out_filters)
        self.act2 = nn.PReLU()

        self.conv3 = nn.Conv2d(out_filters, out_filters, (3, 3), dilation=(2, 2), padding=(2, 2))
        self.bn2 = nn.BatchNorm2d(out_filters)
        self.act3 = nn.PReLU()

        if sca:
            self.sca = sa_layer(out_filters, out_filters//4)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(shortcut)
        resA = self.bn1(resA)
        resA1 = self.act2(resA)

        resA = self.conv3(resA1)
        resA = self.bn2(resA)
        resA2 = self.act3(resA)

        output = shortcut + resA2
        if self.sca:
            output = self.sca(output)
        return output


class CTM(nn.Module):
    def __init__(self, C0=5, C1=24, C2=32, C3=48):
        super(CTM, self).__init__()
        self.mcrf1 = MCRF(C0, C1)
        self.mid = ResContextBlock(C1, C2)
        self.mcrf2 = MCRF(C2, C3)
        self.sca = sa_layer(C3, C3 // 4)
    def forward(self, x):
        x = self.mcrf1(x)
        x = self.mid(x)
        x = self.mcrf2(x)
        x = self.sca(x)
        return x

class propose(nn.Module):
    def __init__(self):
        super(propose, self).__init__()
        # self.convd = CTM(1, 4, 8, 16)
        # self.convx = CTM(1, 4, 8, 16)
        # self.convy = CTM(1, 4, 8, 16)
        # self.convz = CTM(1, 4, 8, 16)
        # self.convr = CTM(1, 4, 8, 16)

        self.convd = CTM(1, 8, 16, 32)
        self.convxyz = CTM(3, 8, 16, 32)
        self.convr = CTM(1, 8, 16, 32)

        self.conv = nn.Conv2d(96, 32, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(32)
        self.pr = nn.PReLU()
        self.sca = sa_layer(channel=32, groups=8)

    def forward(self, xyzdr):
        d = self.convd(xyzdr[:, 0, None, :, :])
        # x = self.convd(xyzdr[:, 1, None, :, :])
        # y = self.convd(xyzdr[:, 2, None, :, :])
        # z = self.convd(xyzdr[:, 3, None, :, :])
        xyz = self.convxyz(xyzdr[:, 1:4, :, :])
        r = self.convr(xyzdr[:, 4, None, :, :])
        feature = torch.cat((d, xyz, r), dim=1)
        feature = self.conv(feature)
        feature = self.bn(feature)
        feature = self.sca(feature)
        feature = self.pr(feature)
        return feature



class SalsaNext(nn.Module):
    def __init__(self, window_size, nclasses):
        super(SalsaNext, self).__init__()
        self.nclasses = nclasses

        self.MRCF = propose()
        # self.MRCF = ConvBNLeakyReLU(5, 32, ks=3, padding=1)
        self.resBlock1 = ResBlock(32, 64, 0.2, pooling=True, drop_out=False, cam=False, dy=False)
        self.resBlock2 = ResBlock(64, 128, 0.2, pooling=True, drop_out=True, cam=False, dy=False)
        self.resBlock3 = ResBlock(128, 256, 0.2, pooling=True, drop_out=True, cam=False, dy=False)
        self.resBlock4 = ResBlock(256, 512, 0.2, pooling=True, drop_out=True, cam=False, dy=False)
        self.Decoder = SwinTransformer(window_size=window_size)
        self.norm = nn.LayerNorm(64)
        self.reduce = UpBlock(80, 64, 0.2, drop_out=False)
        self.logits = nn.Conv2d(64, nclasses, kernel_size=(1, 1))

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        downCntx = self.MRCF(x)
        out = []
        down0c, down0b = self.resBlock1(downCntx)
        down1c, down1b = self.resBlock2(down0c)
        out.append(down1b)
        down2c, down2b = self.resBlock3(down1c)
        out.append(down2b)
        down3c, down3b = self.resBlock4(down2c)
        out.append(down3b)

        up4e = self.Decoder(down3c, out)
        up0 = self.norm(up4e).view(-1, h//2, w//2, 64).permute(0, 3, 1, 2).contiguous()
        up0 = self.reduce(up0, down0b)
        logits = self.logits(up0)
        logits = F.softmax(logits, dim=1)

        return logits

if __name__ == '__main__':
    test_model = CAM(128)
    dummy_input = torch.rand(1,128,4,64)
    a =test_model(dummy_input)




import torch
import torch.nn as nn
import torch.nn.functional as F

# Keras BatchNormalization defaults: epsilon=1e-3, momentum=0.99.
# PyTorch's momentum is (1 - keras_momentum), so 0.01 here.
BN_EPS = 1e-3
BN_MOMENTUM = 0.01


def BN(c):
    """Standard per-channel BN with Keras-equivalent eps/momentum."""
    return nn.BatchNorm2d(c, eps=BN_EPS, momentum=BN_MOMENTUM)


class RowBN(nn.Module):
    """
    Reproduces Keras BatchNormalization(axis=1, scale=False) on NHWC data.
    In channels-last, axis=1 is the HEIGHT axis: one mean/var (and one beta)
    per image row, no gamma. Input here is NCHW, so we swap C and H.
    """
    def __init__(self, height):
        super().__init__()
        self.bn = nn.BatchNorm2d(height, eps=BN_EPS, momentum=BN_MOMENTUM, affine=False)
        self.beta = nn.Parameter(torch.zeros(height))

    def forward(self, x):                      # N,C,H,W
        x = x.permute(0, 2, 1, 3)              # N,H,C,W
        x = self.bn(x) + self.beta.view(1, -1, 1, 1)
        return x.permute(0, 2, 1, 3)           # N,C,H,W


def bn_noscale(channels, height, tf_row_bn):
    """BN used where the TF code has axis=1, scale=False."""
    if tf_row_bn:
        return RowBN(height)
    return nn.BatchNorm2d(channels, eps=BN_EPS, momentum=BN_MOMENTUM, affine=False)


# Equivalent to GlorotNormal initialization in TensorFlow
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class DownsamplingBlock(nn.Module):
    def __init__(self, in_filters, out_filters, height, tf_row_bn=True):
        super().__init__()
        self.bn1 = BN(in_filters)
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1)
        # `height` = feature-map height at this stage (before pooling)
        self.bn2 = bn_noscale(out_filters, height, tf_row_bn)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.bn1(x)
        conv = F.relu(self.conv1(x))
        conv = self.bn2(conv)
        conv = F.relu(self.conv2(conv))
        pool = self.pool(conv)
        return conv, pool


def _mmff_path(filters):
    return nn.Sequential(
        BN(filters),
        nn.Conv2d(filters, filters, kernel_size=1),
        nn.ReLU(),
        BN(filters),
        nn.Conv2d(filters, filters, kernel_size=3, padding=1),
        nn.ReLU(),
    )


class MMFF(nn.Module):
    """Multi-Modal Feature Fusion, for n modalities."""
    def __init__(self, filters, n_branches):
        super().__init__()
        self.paths = nn.ModuleList([_mmff_path(filters) for _ in range(n_branches)])

    def forward(self, feats):                       # list of n tensors
        return torch.cat([p(f) for p, f in zip(self.paths, feats)], dim=1)


class MSFU(nn.Module):
    """Multi-Scale Feature Upsampling. n_branches = how many modality
    feature maps are concatenated in `multimodal` (1 for early fusion)."""
    def __init__(self, high_filters, low_filters, height, n_branches, tf_row_bn=True):
        super().__init__()
        self.bn_low = BN(low_filters)
        self.conv_low = nn.Conv2d(low_filters, high_filters, kernel_size=1)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        conc_ch = high_filters * (n_branches + 1)   # was high*4 for 3 modalities
        self.bn_conc = BN(conc_ch)
        self.conv_conc1 = nn.Conv2d(conc_ch, high_filters, kernel_size=1)
        self.bn_conc2 = bn_noscale(high_filters, height, tf_row_bn)
        self.conv_conc2 = nn.Conv2d(high_filters, high_filters, kernel_size=3, padding=1)

    def forward(self, multimodal, lowres):
        lowres = self.bn_low(lowres)
        us_low = self.up(F.relu(self.conv_low(lowres)))
        conc = self.bn_conc(torch.cat([multimodal, us_low], dim=1))
        x = F.relu(self.conv_conc1(conc))
        x = self.bn_conc2(x)
        return F.relu(self.conv_conc2(x))


class BottomBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super().__init__()
        self.bn1 = BN(in_filters)
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1)
        self.bn2 = BN(out_filters)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.bn1(x)
        x = F.relu(self.conv1(x))
        x = self.bn2(x)
        return F.relu(self.conv2(x))


class Adapter(nn.Module):
    """Intermediate fusion: concat n modality features, mix with 1x1 conv
    back to `filters` channels (same idea as adapt_s1..s4 in TResUnet)."""
    def __init__(self, filters, n_branches):
        super().__init__()
        self.net = nn.Sequential(
            BN(filters * n_branches),
            nn.Conv2d(filters * n_branches, filters, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, feats):                       # list of n tensors
        return self.net(torch.cat(feats, dim=1))


class UNet(nn.Module):
    """
    fusion="early"        : channels stacked, single encoder.
    fusion="intermediate" : one encoder per modality, concat + 1x1 conv at
                            every level, single-width decoder.
    fusion="multibranch"  : one encoder per modality, MMFF at every level,
                            decoder sees n*f channels.

    forward accepts one tensor (B, n, H, W) or n tensors (B, 1, H, W).
    Returns LOGITS (B, 1, H, W).
    """
    FUSIONS = ("early", "intermediate", "multibranch")

    def __init__(self, filter_base, n_modalities=3, fusion="multibranch",
                 img_size=256, tf_row_bn=True):
        super().__init__()
        assert fusion in self.FUSIONS, f"fusion must be one of {self.FUSIONS}"
        assert img_size % 16 == 0, "img_size must be divisible by 16"
        assert n_modalities >= 1

        self.fusion = fusion
        self.n_modalities = n_modalities
        nb = 1 if fusion == "early" else n_modalities        # encoder branches
        in_ch = n_modalities if fusion == "early" else 1     # channels per branch
        dec = n_modalities if fusion == "multibranch" else 1 # width multiplier in decoder
        self.n_branches = nb

        f = [filter_base * 2**i for i in range(5)]
        h = [img_size, img_size // 2, img_size // 4, img_size // 8]
        enc_in = [in_ch, f[0], f[1], f[2]]

        self.encoders = nn.ModuleList([
            nn.ModuleList([DownsamplingBlock(enc_in[l], f[l], h[l], tf_row_bn) for l in range(4)])
            for _ in range(nb)
        ])
        self.bottoms = nn.ModuleList([BottomBlock(f[3], f[4]) for _ in range(nb)])

        if fusion == "multibranch":
            self.fusers = nn.ModuleList([MMFF(f[l], nb) for l in range(5)])
        elif fusion == "intermediate":
            self.fusers = nn.ModuleList([Adapter(f[l], nb) for l in range(5)])
        else:
            self.fusers = None

        self.up1 = MSFU(f[3], f[4] * dec, h[3], dec, tf_row_bn)
        self.up2 = MSFU(f[2], f[3], h[2], dec, tf_row_bn)
        self.up3 = MSFU(f[1], f[2], h[1], dec, tf_row_bn)
        self.up4 = MSFU(f[0], f[1], h[0], dec, tf_row_bn)

        self.final_conv = nn.Conv2d(f[0], 1, kernel_size=1)

        self.apply(weights_init)
        nn.init.xavier_uniform_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def _prepare(self, inputs):
        if len(inputs) == 1:
            x = inputs[0]
            if x.shape[1] != self.n_modalities:
                raise ValueError(f"Expected {self.n_modalities} channels, got {x.shape[1]}")
            return [x] if self.fusion == "early" else list(x.split(1, dim=1))
        if len(inputs) != self.n_modalities:
            raise ValueError(f"Expected {self.n_modalities} inputs, got {len(inputs)}")
        return [torch.cat(inputs, dim=1)] if self.fusion == "early" else list(inputs)

    def _fuse(self, level, feats):
        return feats[0] if self.fusion == "early" else self.fusers[level](feats)

    def forward(self, *inputs):
        feats = self._prepare(inputs)
        skips = []

        for lvl in range(4):
            outs = [self.encoders[b][lvl](feats[b]) for b in range(self.n_branches)]
            skips.append(self._fuse(lvl, [o[0] for o in outs]))
            feats = [o[1] for o in outs]

        bottom = [self.bottoms[b](feats[b]) for b in range(self.n_branches)]
        mm5 = self._fuse(4, bottom)

        u1 = self.up1(skips[3], mm5)
        u2 = self.up2(skips[2], u1)
        u3 = self.up3(skips[1], u2)
        u4 = self.up4(skips[0], u3)
        return self.final_conv(u4)


def callback_logic(model, val_loss, best_val_loss, patience_counter,
                   checkpoint_path1, checkpoint_path2, epoch, patience=200):
    """
    Equivalent to the Keras callback() function.
    Returns: (best_val_loss, patience_counter, stop_training)
    """
    stop_training = False

    # Periodic save every epoch (checkpoint_filepath2)
    torch.save(model.state_dict(), checkpoint_path2.format(epoch=epoch, val_loss=val_loss))

    # ModelCheckpoint(save_best_only=True)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), checkpoint_path1)
        print(f"--- Improved! Model saved to {checkpoint_path1} ---")
    else:
        patience_counter += 1

    # EarlyStopping
    if patience_counter >= patience:
        print(f"Early stopping triggered at epoch {epoch}")
        stop_training = True

    return best_val_loss, patience_counter, stop_training

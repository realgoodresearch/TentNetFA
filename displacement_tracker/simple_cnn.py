import torch
import torch.nn as nn



class SimpleCNN(nn.Module):
    ALLOWED_KWARGS = {"kernel_size", "dropout"}

    def __init__(self, n_channels, n_classes=1, **kwargs):
        super().__init__()

        unexpected_keys = set(kwargs) - self.ALLOWED_KWARGS
        if unexpected_keys:
            raise TypeError(
                f"Unexpected keyword argument(s) for {self.__class__.__name__}: "
                f"{', '.join(sorted(unexpected_keys))}"
            )

        kernel_size = kwargs.pop("kernel_size", 3)
        dropout = kwargs.pop("dropout", False)

        self.config = {
            "n_channels": n_channels,
            "n_classes": n_classes,
            "kernel_size": kernel_size,
            "dropout": dropout,
        }

        dropout_rates = [0.1, 0.1, 0.2, 0.2, 0.3, 0.3] if dropout else [0.0] * 6

        def _drop(p):
            return nn.Dropout2d(p) if dropout else nn.Identity()

        self.conv1 = nn.Conv2d(
            n_channels, 8, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = _drop(dropout_rates[0])

        self.conv2 = nn.Conv2d(
            8, 16, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = _drop(dropout_rates[1])

        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=2, dilation=2, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.relu3 = nn.ReLU(inplace=True)
        self.drop3 = _drop(dropout_rates[2])

        self.conv4 = nn.Conv2d(16, 8, kernel_size=7, padding=3, bias=False)
        self.bn4 = nn.BatchNorm2d(8)
        self.relu4 = nn.ReLU(inplace=True)
        self.drop4 = _drop(dropout_rates[3])

        self.conv5 = nn.Conv2d(
            8, 8, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )
        self.bn5 = nn.BatchNorm2d(8)
        self.relu5 = nn.ReLU(inplace=True)
        self.drop5 = _drop(dropout_rates[4])

        self.conv6 = nn.Conv2d(
            8, 8, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )
        self.bn6 = nn.BatchNorm2d(8)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = _drop(dropout_rates[5])

        self.conv_ctx = nn.Conv2d(8, 8, kernel_size=9, padding=4, bias=False)
        self.bn_ctx = nn.BatchNorm2d(8)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Conv2d(8, 8, kernel_size=1)
        self.global_relu = nn.ReLU(inplace=True)

        self.convend = nn.Conv2d(8, n_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.drop2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.drop3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.drop4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.drop5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        x = self.drop6(x)

        x = self.conv_ctx(x)
        x = self.bn_ctx(x)

        g = self.global_pool(x)  # (B, 8, 1, 1)
        g = self.global_fc(g)  # (B, 8, 1, 1)
        g = self.global_relu(g)
        x = x + g  # broadcast add

        x = self.convend(x)
        return x

    @classmethod
    def from_pth(cls, file_name: str, map_location=None, model_args=None):
        if map_location is None:
            map_location = torch.device("cpu")

        checkpoint = torch.load(file_name, map_location=map_location)

        model_args = model_args or {}
        if isinstance(checkpoint, dict) and "model_args" in checkpoint:
            allowed = cls.ALLOWED_KWARGS | {"n_channels", "n_classes"}
            ckpt_args = {
                k: v for k, v in checkpoint["model_args"].items() if k in allowed
            }
            model_args = model_args | ckpt_args

        model = cls(**model_args)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        return model

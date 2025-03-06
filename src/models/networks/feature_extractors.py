import torch
from torch import nn
from torchvision import models
from torch.nn.parameter import Parameter


class FrozenBatchNorm2d(nn.Module):
    """A BatchNorm2d wrapper for Pytorch's BatchNorm2d where the batch
    statictis are fixed.
    """
    def __init__(self, num_features):
        super(FrozenBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.register_parameter("weight", Parameter(torch.ones(num_features)))
        self.register_parameter("bias", Parameter(torch.zeros(num_features)))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def extra_repr(self):
        return '{num_features}'.format(**self.__dict__)

    @classmethod
    def from_batch_norm(cls, bn):
        fbn = cls(bn.num_features)
        # Update the weight and biases based on the corresponding weights and
        # biases of the pre-trained bn layer
        with torch.no_grad():
            fbn.weight[...] = bn.weight
            fbn.bias[...] = bn.bias
            fbn.running_mean[...] = bn.running_mean
            fbn.running_var[...] = bn.running_var + bn.eps
        return fbn

    @staticmethod
    def _getattr_nested(m, module_names):
        if len(module_names) == 1:
            return getattr(m, module_names[0])
        else:
            return FrozenBatchNorm2d._getattr_nested(
                getattr(m, module_names[0]), module_names[1:]
            )

    @staticmethod
    def freeze(m):
        for (name, layer) in m.named_modules():
            if isinstance(layer, nn.BatchNorm2d):
                nest = name.split(".")
                if len(nest) == 1:
                    setattr(m, name, FrozenBatchNorm2d.from_batch_norm(layer))
                else:
                    setattr(
                        FrozenBatchNorm2d._getattr_nested(m, nest[:-1]),
                        nest[-1],
                        FrozenBatchNorm2d.from_batch_norm(layer)
                    )

    def forward(self, x):
        # Cast all fixed parameters to half() if necessary
        if x.dtype == torch.float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()

        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        out = x * scale + bias
        # print(out.shape)
        return out


def freeze_network(network, freeze=False):
    if freeze:
        for p in network.parameters():
            p.requires_grad = False
    return network


class BaseFeatureExtractor(nn.Module):
    """Hold some common functions among all feature extractor networks.
    """
    @property
    def feature_size(self):
        return self._feature_size

    def forward(self, X):
        out = self._feature_extractor(X)
        return out


class ResNet18(BaseFeatureExtractor):
    """Build a feature extractor using the pretrained ResNet18 architecture for
    image based inputs.
    """
    def __init__(self, freeze_bn, input_channels, feature_size):
        super(ResNet18, self).__init__()
        self._feature_size = feature_size

        self._feature_extractor = models.resnet18(pretrained=False)
        if freeze_bn:
            FrozenBatchNorm2d.freeze(self._feature_extractor)

        self._feature_extractor.conv1 = torch.nn.Conv2d(
            input_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False
        )

        # 移除原有的fc层
        self._feature_extractor.fc = nn.Sequential(
            nn.Linear(512 * 10, 512 * 10), nn.ReLU(),
            nn.Linear(512 * 10, self.feature_size * 10)
        )
        # 修改avgpool为自适应池化，输出维度为 (batch_size, 512, 10, 1)
        self._feature_extractor.avgpool = nn.AdaptiveAvgPool2d((10, 1))

    def forward(self, X):
        # 通过特征提取器
        out = self._feature_extractor(X)
        # 调整维度为 [batch_size, 10, 128]
        out = out.view(out.size(0), 10, 128)
        return out


class AlexNet(BaseFeatureExtractor):
    def __init__(self, input_channels, feature_size):
        super(AlexNet, self).__init__()
        self._feature_size = feature_size

        self._feature_extractor = models.alexnet(pretrained=False)
        self._feature_extractor.features[0] = torch.nn.Conv2d(
            input_channels,
            64,
            kernel_size=(11, 11),
            stride=(4, 4),
            padding=(2, 2),
        )

        self._fc = nn.Linear(9216, self._feature_size)

    def forward(self, X):
        X = self._feature_extractor.features(X)
        X = self._feature_extractor.avgpool(X)
        X = self._fc(X.view(X.shape[0], -1))

        return X


def get_feature_extractor(
    name,
    freeze_bn=False,
    input_channels=1,
    feature_size=128
):
    """Based on the name return the appropriate feature extractor."""
    return {
        "resnet18": ResNet18(
            freeze_bn=freeze_bn,
            input_channels=input_channels,
            feature_size=feature_size
        ),
        # "alexnet": AlexNet(input_channels, feature_size=feature_size)
    }[name]

import torch.nn as nn
import torch
from torch.distributions.normal import Normal
from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
                               MetaBatchNorm2d, MetaLinear)


def conv3x3(in_channels, out_channels, **kwargs):
    return MetaSequential(
        MetaConv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        MetaBatchNorm2d(out_channels, momentum=1., track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class ConvolutionalNeuralNetwork(MetaModule):
    def __init__(self, in_channels, out_features, hidden_size=64):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size

        self.features = MetaSequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size)
        )

        self.classifier = MetaLinear(hidden_size, out_features)

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        features = features.view((features.size(0), -1))
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits


class PolicyNetwork(MetaModule):
    def __init__(self, in_dims, out_dims, hidden_size=100):
        super(PolicyNetwork, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims * 2 # diag guassian distribution
        self.hidden_size = hidden_size
        fc1 = MetaLinear(in_dims, hidden_size)
        fc2 = MetaLinear(hidden_size, hidden_size)
        fc3 = MetaLinear(hidden_size, self.out_dims)

        self.features = MetaSequential(
            fc1,
            nn.Tanh(),
            fc2,
            nn.Tanh(),
            fc3,
            nn.Tanh(),
        )

        self.activation = {}
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output.detach()
            return hook

        fc1.register_forward_hook(get_activation('fc1'))
        fc2.register_forward_hook(get_activation('fc2'))
        fc3.register_forward_hook(get_activation('fc3'))


    def forward(self, inputs, params=None):
        """Return logp
        """
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        outputs = features.view((features.size(0), -1))
        mean, log_std = torch.split(outputs, int(self.out_dims / 2), dim=-1)
        out_dist = Normal(mean, log_std.exp())
        return out_dist

    # def compute_action(self, inputs, params=None):
    #     """Return logp
    #     """
    #     features = self.features(inputs, params=self.get_subdict(params, 'features'))
    #     outputs = features.view((features.size(0), -1))
    #     mean, log_std = torch.split(outputs, int(self.out_dims / 2), dim=-1)
    #     return mean

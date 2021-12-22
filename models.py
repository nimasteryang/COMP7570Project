from torch import nn
from torch.nn import ModuleList, ReLU


class AE(nn.Module):
    def __init__(self, n_layers, first_layer_size):
        super(AE, self).__init__()
        self.n_layers = n_layers
        self.enc = ModuleList()
        self.relu = ReLU(inplace=True)
        init_layer = first_layer_size
        for i in range(n_layers):
            self.enc.append(nn.Linear(int(init_layer), int(init_layer * 0.8)))
            init_layer = init_layer * 0.8
        self.dec = ModuleList()
        for i in range(n_layers):
            self.dec.append(nn.Linear(int(init_layer), int(init_layer * 1.25)))
            init_layer = init_layer * 1.25

    def forward(self, x):
        for i in range(self.n_layers):
            x = self.enc[i](x)
            x = self.relu(x)
        for i in range(self.n_layers):
            x = self.dec[i](x)
            x = self.relu(x)
        return x


class MLP_classifier(nn.Module):
    def __init__(self):
        super(MLP_classifier, self).__init__()
        self.mlp = nn.Linear(250,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.mlp(x)
        return self.sigmoid(x)
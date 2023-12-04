import torch


class BaseModule(torch.nn.Module):

    def __init__(self, params):
        super().__init__()
        self.layers = torch.nn.Sequential()
        self.build_model(params)

    def build_model(self, params):
        for key, value in params.items():
            if value['type'] == 'linear':
                self.layers.append(torch.nn.Linear(**value['params']))
            if value['type'] == 'cnn':
                self.layers.append(torch.nn.Conv1d(**value['params']))
            if value['type'] == 'sigmoid':
                self.layers.append(torch.nn.Sigmoid())

    def forward(self, x):
        x = self.layers.forward(x)
        return x


if __name__ == '__main__':
    params = {"layer1": {"type": "linear", "params": {"in_features": 4, "out_features": 4, "bias": True}},
              "layer2": {"type": "sigmoid", },
              "layer3": {"type": "linear", "params": {"in_features": 4, "out_features": 8, "bias": True}},
              "layer4": {"type": "sigmoid", }}
    model = BaseModule(params)
    print(model)

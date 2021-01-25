from collections import OrderedDict

import torch

from torchhandle.workflow import BaseContext


class Net(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.layer = torch.nn.Sequential(OrderedDict([
            ('l1', torch.nn.Linear(10, 20)),
            ('a1', torch.nn.ReLU()),
            ('l2', torch.nn.Linear(20, 10)),
            ('a2', torch.nn.ReLU()),
            ('l3', torch.nn.Linear(10, 1))
        ]))

    def forward(self, x):
        x = self.layer(x)
        return x


if __name__ == "__main__":
    num_samples, num_features = int(1e4), int(1e1)
    print(num_samples, num_features)
    X, Y = torch.rand(num_samples, num_features), torch.rand(num_samples)
    dataset = torch.utils.data.TensorDataset(X, Y)
    trn_loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=0, shuffle=True)
    loaders = {"train": trn_loader, "valid": trn_loader}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = {"fn": Net}
    criterion = {"fn": torch.nn.MSELoss}
    optimizer = {"fn": torch.optim.Adam,
                 "args": {"lr": 0.1},
                 "params": {"layer.l1.weight": {"lr": 0.01},
                            "layer.l1.bias": {"lr": 0.02}}
                 }
    scheduler = {"fn": torch.optim.lr_scheduler.StepLR,
                 "args": {"step_size": 2, "gamma": 0.9}
                 }

    c = BaseContext(model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    context_tag="ex01")
    train = c.make_train_session(device, dataloader=loaders)
    train.train(epochs=10)

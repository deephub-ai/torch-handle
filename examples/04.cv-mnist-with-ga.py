import torch
import numpy as np
from torchvision import datasets, transforms,models
from torchhandle.workflow import BaseContext,Metric


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 3, kernel_size=1)
        self.resnet18=models.resnet18(pretrained=False,num_classes=10)


    def forward(self, x):
        x = self.conv1(x)
        x= self.resnet18(x)
        return x
class ACCU(Metric):
    @property
    def name(self):
        return ["accuracy","fake_metric"]

    @property
    def best(self):
        return ["max","min"]
    @property
    def agg_fn(self) -> list:
        return [np.mean,np.sum]

    def calculate(self,session):
        pred = session.state.output_batch.detach().cpu()
        targets = session.state.target_batch
        pred = torch.argmax(pred, 1)
        correct = (pred == targets).sum().float()
        total = len(targets)
        return [(correct/total).item(),session.state.current_epoch]

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    data_train = datasets.MNIST(root="./data/",
                                transform=transform,
                                train=True,
                                download=True)

    data_test = datasets.MNIST(root="./data/",
                               transform=transform,
                               train=False)
    trn_loader = torch.utils.data.DataLoader(data_train, batch_size=256, num_workers=0,shuffle=True)
    val_loader = torch.utils.data.DataLoader(data_test, batch_size=512, num_workers=0)
    loaders = {"train": trn_loader, "valid": val_loader}
    model = {"fn": Model
             }
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = {"fn": torch.nn.CrossEntropyLoss
                 }
    optimizer = {"fn": torch.optim.Adam
                 }
    scheduler = {"fn": torch.optim.lr_scheduler.StepLR,
                 "args": {"step_size": 2, "gamma": 0.9}
                 }
    c = BaseContext(model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metric_fn=[ACCU()],
                    output_dir="./outputs",
                    progress=20,
                    ga_step_size=4,
                    context_tag="ex04")
    session=c.make_train_session(device,loaders)
    session.train(10)


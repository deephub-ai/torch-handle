import torch
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms, models

from torchhandle.workflow import BaseContext, Metric


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 3, kernel_size=1)
        self.resnet18 = models.resnet18(pretrained=False, num_classes=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.resnet18(x)
        return x


class ACCU(Metric):
    def __init__(self):
        self.target_list = None
        self.pred_list = None

        self.correct = None

    @property
    def name(self):
        return ["accuracy", "accuracy_score"]

    @property
    def best(self):
        return ["max", "max"]

    def map(self, state):
        target = state.target_batch.cpu().detach()
        output = state.output_batch.cpu().detach()
        pred = torch.argmax(output, 1)
        # example 1 :suggest way cal metric

        correct = (pred == target)
        if self.correct is None:
            self.correct = correct
        else:
            self.correct = torch.cat([self.correct, correct], dim=0)

        # example 2 save output and cal by sklearn
        if self.target_list is None:
            self.target_list = target
        else:
            self.target_list = torch.cat([self.target_list, target], dim=0)
        if self.pred_list is None:
            self.pred_list = pred
        else:
            self.pred_list = torch.cat([self.pred_list, pred], dim=0)

    def reduce(self):
        # example 1
        out1 = self.correct.sum().float() / self.correct.shape[0]
        # example 2
        out2 = accuracy_score(self.target_list.numpy(), self.pred_list.numpy())
        return [out1, out2]


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
    trn_loader = torch.utils.data.DataLoader(data_train, batch_size=256, num_workers=0, shuffle=True)
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
                    metric_fn=[{"fn": ACCU}],
                    output_dir="./outputs",
                    progress=20,
                    ga_step_size=4,
                    context_tag="ex04")
    session = c.make_train_session(device, loaders)
    session.train(10)

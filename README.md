# torchhandle

TorchHandle makes your PyTorch development more efficient and make you use PyTorch more comfortable

torchhandle is an auxiliary framework for PyTorch. It abstracts the cumbersome and repetitive training code of PyTorch, allowing data scientists to focus on data processing, model creation and arameter optimization instead of writing repetitive training loop codes.

Torchhandle will make your code  more concise and readable, and make your development tasks more efficient.

## Introduction
Torchhandle abstractly organizes and extracts the training and inference process of Pytorch, build  deep learning pipeline of PyTorch only need a few lines of code.

Custom training metrics, cross validation, early stop, gradient accumulation, checkpoint recovery, full training reporting, integrated Tensorboard visualization, and more are all available with a few simple options.

[中文文档](docs/zh-cn/)

### Install

```bash
pip install -U torchhandle
```


TorchHandle requires very few dependent libraries to run. The recommended versions of the libraries are as follows

- Python 3.6 +
- PyTorch 1.5 + (1.1+ will ok, preferably 1.5 +)
- tqdm 4.33.0 +
- matplotlib
- OS (Centos7, Ubuntu),Windows 10, Colab,Kaggle tested, MacOS not tested

### Quick Start 
```python
#model
model = {"fn": "model class",
         "args":"Parameters that need to be passed to model"# optional
        } 

#loss function
criterion = {"fn": "loss function  class",
             "args":"Parameters that need to be passed"# optional
             }
#optimizer
optimizer = {"fn": "optimizer class",
             "args":"Parameters for create optimizer",# optional
             "params":"different parameters of each  model layers" # optional  see example 01
            }
# lr scheduler
scheduler = {"fn": "lr scheduler class",
             "args": "scheduler arameters",
             "type": "batch/epoch" # call scheduler per epoch/batch  ,default epoch
             }

#dataloader
loaders = {"train": "train dataloader",  
           "valid": "valid dataloader" # optional 
           }
```

### Workflow object definition

Context: The Context of the training environment, containing the model to be trained, optimizer, loss function, scheduler, and other parameter that not change in train loop

Session: Session object is created according to the context. Each Session holds a separate model object, optimizer, etc. and cross-validation can be achieved by creating different sessions in  one Context

Metric: Custom metrics

## Examples

<details>
<summary>01 ML - MLP with different learning rate for specific layer</summary>
<p>

```python

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
    
num_samples, num_features = int(1e4), int(1e1)
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
```
</p>
</details>

<details>
<summary>02 ML - linear regression with early stopping by custom metrics and save all metrics to tensorboard</summary>
<p>

```python

import torch

from torchhandle.workflow import BaseContext,Metric

import math


class C1(BaseContext):
    def init_state_fn(self):
        state=super().init_state_fn()
        state.es_current_step=0
        state.es_metric=1000
        return state

    def early_stopping_fn(self,session):
        """
        return true to stop
        """
        es_steps = 5
        valid_loss = session.epoch_metric["valid_loss"]
        session.state.es_current_step=session.state.es_current_step+1
        if valid_loss < session.state.es_metric:
            session.state.es_metric=valid_loss
            session.state.es_current_step=0
        elif session.state.es_current_step >= es_steps:
            return True

        return False

class RMSE(Metric):
    def __init__(self):
        self.diff = None

    def map(self, state):

        target = state.target_batch.cpu().detach().unsqueeze(dim=1)
        output = state.output_batch.cpu().detach()
        if self.diff is None:
            self.diff = torch.pow(target - output, 2)
        else:
            self.diff = torch.cat([self.diff, torch.pow(target - output, 2)], dim=0)

    def reduce(self):
        mse = torch.sum(self.diff) / self.diff.shape[0]
        rmse = torch.sqrt(mse)
        return [rmse]

    @property
    def name(self) -> list:
        return ["RMSE"]

    @property
    def best(self) -> list:
        return ["min"]

num_samples, num_features = int(1e4), int(1e1)
X, Y = torch.rand(num_samples, num_features), torch.rand(num_samples)
dataset = torch.utils.data.TensorDataset(X, Y)
trn_loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=0,shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=0)
loaders = {"train": trn_loader, "valid": val_loader}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = {"fn": torch.nn.Linear,
         "args": {"in_features": 10, "out_features": 1}
         }
criterion = {"fn": torch.nn.MSELoss
             }
optimizer = {"fn": torch.optim.Adam
             }
metric_fn = [{"fn": RMSE}]
c = C1(model=model,
                criterion=criterion,
                optimizer=optimizer,
                metric_fn=metric_fn,
                output_dir="./outputs",
                logging_file="output.log",
                context_tag="ex02")
train = c.make_train_session(device, dataloader=loaders)
train.train(epochs=100)
print("this line was not write to log file")
```
</p>
</details>

<details>
<summary>03 ML - Cross Validation </summary>
<p>

```python

import torch
from torchhandle.workflow import BaseContext


num_samples, num_features = int(1e4), int(1e1)

X, Y = torch.rand(num_samples, num_features), torch.rand(num_samples)
dataset = torch.utils.data.TensorDataset(X, Y)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = {"fn": torch.nn.Linear,
         "args": {"in_features": 10, "out_features": 1}
         }
criterion = {"fn": torch.nn.MSELoss
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
                output_dir="./outputs",
                logging_file="log.txt",
                context_tag="ex03")
for i in range(5):
    # use all data just for  for demo , not actual Kford
    trn_loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=0,shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=0)
    loaders = {"train": trn_loader, "valid": val_loader}

    session=c.make_train_session(device,dataloader=loaders,fold_tag=i)
    session.train(10)
```

</p>
</details>

<details>
<summary>04 CV mnist - Training built-in model and dataset using gradient accumulation</summary>
<p>

```python

import torch

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
                metric_fn=[{"fn": ACCU}],
                output_dir="./outputs",
                progress=20,
                ga_step_size=4,
                context_tag="ex04")
session=c.make_train_session(device,loaders)
session.train(10)



```

</p>
</details>

<details>
<summary>05 CV mnist - lr_scheduler per batch  </summary>
<p>

```python

import torch

from torchvision import datasets, transforms,models
from torchhandle.workflow import BaseContext
class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 3, kernel_size=1)
        self.resnet18=models.resnet18(pretrained=False,num_classes=10)


    def forward(self, x):
        x = self.conv1(x)
        x= self.resnet18(x)
        return x

class C1(BaseContext):
    # custom scheduler step for pass epoch
    def scheduler_step_fn(self,session):
        epoch = session.state.current_epoch
        session.scheduler.step(epoch)

EPOCHS=10
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5])])
data_train = datasets.MNIST(root="./data/",
                            transform=transform,
                            train=True,
                            download=True)

trn_loader = torch.utils.data.DataLoader(data_train, batch_size=256, num_workers=0,shuffle=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = {"fn": Model
             }
optimizer = {"fn": torch.optim.Adam
             }
scheduler = {"fn": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
             "args": {"T_0": EPOCHS // 3, "T_mult": 1,"eta_min":0,"last_epoch":-1},
             "type" : "batch"
             }
criterion = {"fn": torch.nn.CrossEntropyLoss}
c = C1(model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                context_tag="ex05",
                output_dir="./outputs",
                ga_steps=4)
session = c.make_train_session(device, {"train": trn_loader})
session.train(EPOCHS)

```

</p>
</details>

<p>more example please check examples fold and docs</p>

## TODO

inference function

Checkpoint 

XLA（TPU） Support

Distributed Training

More examples

## Contact us
If you don't like GitHub issues, contact us at deephub.ai[at]gmail.com.

If you planning to contribute  bug fixes, please do PR.

If you planning to contribute new  features , please first open an issue and discuss the feature with us.

If you would like to start a collaboration between your team and deephub, or join our team for better deep learning development, you are always welcome.

If you have any questions, please feel free to send us an email, and we welcome and appreciate any kind of contribution and feedback.

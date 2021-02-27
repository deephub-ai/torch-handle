# torchhandle 中文文档

torchhandle 让你的PyTorch研发更高效，让你使用PyTorch训练更加顺手

torchhandle是一个PyTorch的辅助框架。 它将PyTorch繁琐和重复的训练代码抽象出来，使得数据科学家们能够将精力放在数据处理、创建模型和参数优化，而不是编写重复的训练循环代码。
使用torchhandle，可以让你的代码更加简洁易读，让你的开发任务更加高效。


## 主要功能介绍
torchhandle将Pytorch的训练和推理过程进行了抽象整理和提取，只要使用几行代码就可以实现PyTorch的深度学习管道。

自定义训练指标、交叉验证、早停机制、梯度累加、检查点恢复、完整训练报告、集成tensorboard可视化等等功能只需要几个简单的选项既可以实现。

### 安装

```bash
pip install -U torchhandle
```
torchhandle的需要的依赖库非常的少，只需要几个常用的python库即可运行，我们推荐的库版本如下

 - Python 3.6+
 - PyTorch 1.5 + （1.1+即可，建议 1.5以上版本）
 - tqdm 4.33.0 +
 - matplotlib 
 - OS （Centos7、Ubuntu）,Windows 10, Colab,Kaggle 已测试 ,macOS 未测试

### 快速开始
```python

#网络模型
model = {"fn": "模型的class或者返回模型实例的函数",
         "args":"模型或函数需要传递的参数"# 可选
        } 

#损失函数
criterion = {"fn": "损失函数类",
             "args":"损失函数参数"# 可选
             }
#优化器
optimizer = {"fn": "优化器类",
             "args":"优化器参数",# 可选
             "params":"针对网络的不同层设定不同的参数" # example 01
            }
#学习率scheduler
scheduler = {"fn": "scheduler类",
             "args": "scheduler参数",
             "type": "batch/epoch" # 调用scheduler的类型，默认epoch
                 }

#dataloader
loaders = {"train": "训练集的dataloader",  
           "valid": "验证集的dataloader" # 可选的 
           }
```


### workflow对象定义

Context: 训练环境的上下文，包含需要训练的模型，优化器，损失函数，scheduler和其他一些固定的配置信息

Session: 根据上下文创建的训练Session，每个session保存独立的模型，优化器等对象，在一个上下文中创建不同的session可以实现交叉验证

Metric: 训练的指标，除损失函数外的训练指标

## 使用例子

<details>
<summary>01 ML - 训练多层感知机，每层优化器设置不同的参数</summary>
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
<summary>02 ML - 使用自定义指标和早停机制训练线性回归，训练数据保存到tensorboard中</summary>
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
<summary>03 ML - 交叉验证 </summary>
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
<summary>04 CV mnist - 使用梯度积累训练pytorch内置模型和数据集</summary>
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
<summary>05 CV mnist - 每批次调用 lr_scheduler   </summary>
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

## TODO

微调支持

torch混合精度

推理函数实现

session的检查点功能

TPU支持

分布式训练

更多的样例

## 联系我们

如果你不喜欢Github的issues，可以使用以下邮箱和我们取得联系  deephub.ai[at]gmail.com。

如果你想提供bug修复，请直接PR。

如果你计划贡献功能或扩展，请首先创建issue并与我们确认。

如果您希望您的团队和我们进行合作，或者加入我们更好地进行深度学习研发，欢迎您的参与。

最后，如果你有任何的问题，都可以给我们发送邮件，我们欢迎并感谢任何形式的贡献和反馈。

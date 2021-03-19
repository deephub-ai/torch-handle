import torch
from torchvision import datasets, transforms, models

from torchhandle.workflow import BaseContext


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 3, kernel_size=1)
        self.resnet18 = models.resnet18(pretrained=False, num_classes=5) # initial model classnum is 5,will raise t < n_classes error

    def forward(self, x):
        x = self.conv1(x)
        x = self.resnet18(x)
        return x
def ft_fn(self,class_nums):
    '''
    finetune function
    :param self:frist parameters is session object
    :return:
    '''
    print("before finetune:", self.model.resnet18.fc)
    # we reset output to 10
    self.model.resnet18.fc=torch.nn.Linear(512,10)
    print("after finetune:",self.model.resnet18.fc)




if __name__ == "__main__":
    EPOCHS = 10
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    data_train = datasets.MNIST(root="./data/",
                                transform=transform,
                                train=True,
                                download=True)

    trn_loader = torch.utils.data.DataLoader(data_train, batch_size=256, num_workers=0, shuffle=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = {"fn": Model
             }
    optimizer = {"fn": torch.optim.Adam
                 }
    criterion = {"fn": torch.nn.CrossEntropyLoss}
    c = BaseContext(model=model,
           criterion=criterion,
           optimizer=optimizer,
           context_tag="ex06",
           ft_fn=ft_fn,
            )
    session = c.make_train_session(device, {"train": trn_loader},ft_args=dict(class_nums=10))
    session.train(EPOCHS)

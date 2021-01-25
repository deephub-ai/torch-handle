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
if __name__ == "__main__":
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
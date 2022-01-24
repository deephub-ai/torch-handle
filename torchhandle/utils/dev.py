import matplotlib.pyplot as plt
import torchhandle
from pprint import pprint
def show_scheduler_lr_plt(ctx, epoch):
    if ctx.scheduler is None:
        print("No Scheduler found")
        exit()
    lrs = []
    eps=[]
    ctx.init_state_fn()
    bn=1
    for ep in range(1,epoch+1):
        ctx.set_state(current_epoch=ep)
        if ctx.scheduler_type == "epoch":
            ctx.scheduler_step_fn()
            lrs.append(ctx.scheduler.get_last_lr())
            eps.append(ep)
        else:
            bn=len(ctx.dataloader["train"])

            ctx.set_state(total_epoch=bn)
            for bt in  range(bn):
                ctx.set_state(current_batch=bt)
                ctx.scheduler_step_fn()
                lrs.append(ctx.scheduler.get_last_lr())
                eps.append(ep)
    fig, ax1 = plt.subplots()
    ax1.plot(lrs, 'r-')
    ax1.set_ylabel('LR', color='r')
    if ctx.scheduler_type == "epoch":
        ax1.set_xlabel(f"epoch total:{epoch}", color='b')
    else:
        ax1.set_xlabel(f"{bn} batches pre epoch,total :{epoch}", color='b')

    ax1.xaxis.set_major_locator(plt.MultipleLocator(bn))  # epoch
    plt.show()

def print_torch_info():
    ver={}
    import torch
    ver["torch"]=torch.__version__
    try:
        import torchvision
        ver["torchvision"] = torchvision.__version__
    except :
        ver["torchtext"]="not found"
    try:
        import torchtext
        ver["torchtext"] = torchtext.__version__
    except :
        pass
    try:
        import torch_xla
        ver["torch_xla"] = torch_xla.__version__
    except :
        pass
    ver["torchhandle"] = torchhandle.__version__
    pprint(ver)



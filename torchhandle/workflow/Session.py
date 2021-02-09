import json
import statistics
from pprint import pprint

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from torchhandle.utils import ObjectDict, RedirectStart, RedirectStop


class Session(ObjectDict):
    def __init__(self, context, **kwargs):
        self.ctx = context
        self.fold_tag = ""
        self.tensorboard = True
        self.bar_format = "{desc}:{n_fmt}/{total_fmt} [{elapsed}<{remaining},{rate_fmt}]{postfix}"
        self.update(kwargs)
        if not isinstance(self.fold_tag, str):
            self.fold_tag = str(self.fold_tag)
        # init dir
        if self.ctx.output_dir is None:
            self.session_dir = None
            print("output_dir not set,output will not be saved")
        else:
            self.session_dir = self.ctx.output_dir / self.ctx.context_tag / kwargs["uuid"]
            # tensorboard logs
            (self.session_dir / "logs").mkdir(parents=True, exist_ok=True)
            # model
            (self.session_dir / "model").mkdir(parents=True, exist_ok=True)
        if self.tensorboard and self.session_dir:
            self.writer = SummaryWriter(str(self.session_dir / 'logs'))
        else:
            self.writer = None

        #  init  train parameters
        model_args = {}
        if "args" in self.ctx.model:
            model_args = self.ctx.model["args"]
        self.model = self.ctx.model["fn"](**model_args)
        if self.model_file is not None:
            # session model frist
            self.model.load_state_dict(torch.load(str(self.model_file), map_location=torch.device('cpu')))
        else:
            if self.ctx.model_file is not None:
                self.model.load_state_dict(torch.load(str(self.ctx.model_file), map_location=torch.device('cpu')))

        self.model = self.model.to(self.device)
        criterion_args = {}
        if "args" in self.ctx.criterion:
            criterion_args = self.ctx.criterion["args"]
        self.criterion = self.ctx.criterion["fn"](**criterion_args)
        optimizer_args = {}
        if "args" in self.ctx.optimizer:
            optimizer_args = self.ctx.optimizer["args"]
        optimizer_parameters = self.model.parameters()
        if "params" in self.ctx.optimizer:
            optimizer_parameters = []
            params_list = self.ctx.optimizer["params"].keys()

            base_params = list(filter(lambda kv: kv[0] not in params_list, self.model.named_parameters()))
            spec_params = list(filter(lambda kv: kv[0] in params_list, self.model.named_parameters()))
            params_not_spec = []
            for i in range(len(base_params)):
                params_not_spec.append(base_params[i][1])
            optimizer_parameters.append({"params": params_not_spec})

            for i in range(len(spec_params)):
                k = spec_params[i][0]
                v = self.ctx.optimizer["params"][k].copy()
                v["params"] = spec_params[i][1]
                optimizer_parameters.append(v)
            # optimizer_parameters=self.ctx.optimizer["params_fn"](self.model)
        self.optimizer = self.ctx.optimizer["fn"](optimizer_parameters, **optimizer_args)
        self.scheduler_type = "epoch"
        if self.ctx.scheduler is not None:
            scheduler_args = {}
            if "args" in self.ctx.scheduler:
                scheduler_args = self.ctx.scheduler["args"]
            self.scheduler = self.ctx.scheduler["fn"](self.optimizer, **scheduler_args)
            if "type" in self.ctx.scheduler:
                self.scheduler_type = self.ctx.scheduler["type"]
        else:
            self.scheduler = None
        # calculate dataloader len
        self.train_dl = kwargs["train_dl"]
        self.train_dl_len = len(self.train_dl)
        if kwargs["valid_dl"] is not None:
            self.valid_dl = kwargs["valid_dl"]
            self.valid_dl_len = len(self.valid_dl)
        else:
            self.valid_dl = None
            self.valid_dl_len = 0
        self.epoch_metric = {}
        self.metric_epoch_list = []

    def show_session_summary(self, epochs):
        print("=" * 10, "SESSION SUMMARY BEGIN", "=" * 10)
        pprint(self.ctx)
        print("=" * 20)
        for k in ['device', 'scheduler_type', 'session_dir', 'fold_tag']:
            print(f"{k}: ", self[k])
        print(f"train epochs: ", epochs)
        print("=" * 10, "SESSION SUMMARY END", "=" * 10)

    def train(self, epochs):
        if self.session_dir is not None and self.logging_file is not None:
            lfn = self.session_dir / self.logging_file
            if len(self.fold_tag) > 0:
                lfn = self.session_dir / f"fold_{self.fold_tag}_{self.logging_file}"
            RedirectStart(str(lfn))

        self.show_session_summary(epochs)
        self.metric_epoch_list = []
        # call context init_state_fn(), initialize state
        self.state = self.ctx.init_state_fn()
        for epoch in range(1, epochs + 1):
            # train epoch start
            self.model.train()
            self.state.current_epoch = epoch
            self.epoch_metric = {"epoch": epoch}
            train_dataloader = self.train_dl
            desc = f"EPOCH:{epoch}"
            if len(self.fold_tag) > 0:
                desc = f"FOLD: {self.fold_tag},{desc}"
            if self.progress == "bar":
                train_dataloader = tqdm(train_dataloader, desc=f"TRAIN {desc}", bar_format=self.bar_format, ncols=100)
            self.ctx.epoch_start_fn(self)
            train_epoch_metric_list = []
            # batch start
            for batch_num, data in enumerate(train_dataloader):
                self.state.current_batch = batch_num
                self.ctx.init_batch_data_fn(data, self)
                self.ctx.forward_fn(self)
                self.ctx.loss_fn(self)
                self.ctx.backward_fn(self)
                self.ctx.cal_metric_fn(self)
                if self.progress == "bar":
                    train_dataloader.set_postfix(**self.state.metric)
                elif isinstance(self.progress, int) and self.progress > 0:
                    if (batch_num + 1) % self.progress == 0 or (batch_num + 1) == self.train_dl_len:
                        print(f'TRAIN: {batch_num + 1} / {self.train_dl_len} | {self.state.metric}')
                train_epoch_metric_list.append(self.state.metric)
                # batch scheduler
                if self.scheduler is not None and self.scheduler_type == "batch":
                    if self.ctx.ga_steps(self):
                        self.ctx.scheduler_step_fn(self)

            # cal and print train mean metric
            self.epoch_metric.update(self.agg_metric(train_epoch_metric_list, stage="train"))
            self.ctx.epoch_train_end_fn(self)
            # valid
            self.model.eval()
            if self.valid_dl is not None:
                vaild_epoch_metric_list = []
                valid_dataloader = self.valid_dl
                if self.progress == "bar":
                    valid_dataloader = tqdm(valid_dataloader, desc=f"VALID {desc}", bar_format=self.bar_format,
                                            ncols=100)
                for batch_num, data in enumerate(valid_dataloader):
                    self.ctx.init_batch_data_fn(data, self)
                    self.ctx.eval_fn(self)
                    self.ctx.cal_metric_fn(self)
                    if self.progress == "bar":
                        valid_dataloader.set_postfix(**self.state.metric)
                    elif isinstance(self.progress, int) and self.progress > 0:
                        if (batch_num + 1) % self.progress == 0 or (batch_num + 1) == self.train_dl_len:
                            print(f'VALID : {batch_num + 1} / {self.valid_dl_len} | {self.state.metric}')
                    vaild_epoch_metric_list.append(self.state.metric)
                    # cal valid mean metric
                    self.epoch_metric.update(self.agg_metric(vaild_epoch_metric_list, stage="valid"))

            # epoch scheduler
            if self.scheduler is not None and self.scheduler_type == "epoch":
                self.ctx.scheduler_step_fn(self)
            # log lr every epoch
            self.ctx.get_lr(self)
            self.epoch_metric["lr"] = self.state.lr

            # log eporch metric
            self.metric_epoch_list.append(self.epoch_metric)

            # print metric
            self.print_epoch_metric()

            # tensorboard
            self.save_mertic_tensorboard()

            # save model
            self.save_model()
            # gen report and save best model
            self.save_report()

            # early stopping
            if self.ctx.early_stopping_fn(self):
                print(f"early stopping:{epoch}")
                # valid end
                self.ctx.valid_end_fn(self.state)
                break
            else:
                self.ctx.valid_end_fn(self.state)

        # end train
        if self.writer:
            self.writer.close()
        print("end train")
        RedirectStop()

    def agg_metric(self, epoch_metric_list, stage="train"):
        metric_list = {}
        for key in self.ctx.metric_keys.keys():
            metric_list[f"{stage}_{key}"] = self.ctx.metric_agg_fn[key]([item[key] for item in epoch_metric_list]).tolist()
        return metric_list

    def print_epoch_metric(self):
        print("\r", "*" * 10, " EPOCH METRICS ", "*" * 10)
        if len(self.fold_tag) > 0:
            print(f"Fold : {self.fold_tag}")
        for key in self.epoch_metric.keys():
            print(f"{key} : {self.epoch_metric[key]}")
        print("*" * 30, "\r")

    def save_mertic_tensorboard(self):
        if self.writer is None:
            return
        main_tag = self.ctx.context_tag
        if len(self.fold_tag) > 0:
            main_tag = f"{self.ctx.context_tag}_fold_{self.fold_tag}"
        # metrics
        for mk in self.ctx.metric_keys.keys():
            scalars = {}
            for t in ["train", "valid"]:
                k = f"{t}_{mk}"
                if k in self.epoch_metric:
                    scalars[k] = np.array(self.epoch_metric[k])
            self.writer.add_scalars(f"{main_tag}/{mk}", scalars, global_step=self.epoch_metric["epoch"])
            self.writer.flush()

        # lr
        lrs = {}
        for i in range(0, len(self.epoch_metric["lr"])):
            key = f"lr_{i}"
            lrs[key] = np.array(self.epoch_metric["lr"][i])
        self.writer.add_scalars(f"{main_tag}/lr", lrs, global_step=self.epoch_metric["epoch"])
        self.writer.flush()

    def save_model(self):
        if self.session_dir is None:
            return
        path = self.session_dir / "model"
        if len(self.fold_tag) > 0:
            path = path / f"fold_{self.fold_tag}"
            path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), str(path / f"epoch_{self.epoch_metric['epoch']}.pth"))

    def save_report(self):
        if self.session_dir is None:
            return
        filename = "report.json"
        if len(self.fold_tag) > 0:
            filename = f"fold_{self.fold_tag}_{filename}"
        rep = {"best": {}}
        # epoch
        rep["epoch_metric_list"] = self.metric_epoch_list
        # best
        ml = self.metric_epoch_list.copy()
        tag = ["train"]
        if self.valid_dl_len > 0:
            tag.append("valid")
        for mk, mv in self.ctx.metric_keys.items():
            for t in tag:
                ml.sort(key=lambda x: x[f"{t}_{mk}"])
                if mv == "min":
                    best_value = ml[0]
                elif mv == "max":
                    best_value = ml[-1]
                else:
                    best_value = ""
                rep["best"][f"{t}_{mk}"] = best_value
                # save best model
                if isinstance(best_value, dict) and best_value["epoch"] == self.epoch_metric["epoch"]:
                    print(f"save best model : {t}_{mk} , {self.epoch_metric['epoch']}")
                    best_filename = f"bestmodel_{t}_{mk}.pth"
                    if len(self.fold_tag) > 0:
                        best_filename = f"fold_{self.fold_tag}_bestmodel_{t}_{mk}.pth"
                    torch.save(self.model.state_dict(), str(self.session_dir / best_filename))

        # save file
        with open(str(self.session_dir / filename), "w") as f:
            json.dump(rep, f, indent=3, separators=(',', ':'))


# https://discuss.pytorch.org/t/different-learning-rate-for-a-specific-layer/33670/10
def group_wise_lr(model, group_lr_conf: dict, path=""):
    """
    Refer https://pytorch.org/docs/master/optim.html#per-parameter-options


    torch.optim.SGD([
        {'params': model.base.parameters()},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ], lr=1e-2, momentum=0.9)


    to


    cfg = {"classifier": {"lr": 1e-3},
           "lr":1e-2, "momentum"=0.9}
    confs, names = group_wise_lr(model, cfg)
    torch.optim.SGD([confs], lr=1e-2, momentum=0.9)



    :param model:
    :param group_lr_conf:
    :return:
    """
    assert type(group_lr_conf) == dict
    confs = []
    nms = []
    for kl, vl in group_lr_conf.items():
        assert type(kl) == str
        assert type(vl) == dict or type(vl) == float or type(vl) == int

        if type(vl) == dict:
            assert hasattr(model, kl)
            cfs, names = group_wise_lr(getattr(model, kl), vl, path=path + kl + ".")
            confs.extend(cfs)
            names = list(map(lambda n: kl + "." + n, names))
            nms.extend(names)

    primitives = {kk: vk for kk, vk in group_lr_conf.items() if type(vk) == float or type(vk) == int}
    remaining_params = [(k, p) for k, p in model.named_parameters() if k not in nms]
    if len(remaining_params) > 0:
        names, params = zip(*remaining_params)
        conf = dict(params=params, **primitives)
        confs.append(conf)
        nms.extend(names)

    plen = sum([len(list(c["params"])) for c in confs])
    assert len(list(model.parameters())) == plen
    assert set(list(zip(*model.named_parameters()))[0]) == set(nms)
    assert plen == len(nms)
    if path == "":
        for c in confs:
            c["params"] = (n for n in c["params"])
    return confs, nms

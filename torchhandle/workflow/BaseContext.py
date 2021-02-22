import time
from pathlib import Path

import torch
import numpy as np
from torchhandle.utils import ObjectDict
from torchhandle.workflow.Metric import Metric
from torchhandle.workflow.Session import Session


class BaseContext(ObjectDict):
    def __init__(self, **kwargs):
        """
        :param model :
        :param criterion :
        :param optimizer :
        :param scheduler :
        :param scheduler_type :
        :param ga_step_size :
        :param metric_fn :
        :param context_tag :
        :param output_dir :
        :param logging_file :
        :param progress :
        :param model_file :
        """

        self.output_dir = None
        self.metric_fn = []
        self.ga_step_size = 0
        self.scheduler = None
        self.logging_file = None
        self.progress = "bar"
        self.uuid = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
        self.model_file = None
        self.update(kwargs)
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)
        self.metric_keys = {"loss": "min"}
        for i in range(len(self.metric_fn)):
            if isinstance(self.metric_fn[i], Metric):
                m = self.metric_fn[i]
                for j in range(len(m.name)):
                    self.metric_keys[m.name[j]] = m.best[j]
                    #self.metric_agg_fn[m.name[j]] = m.agg_fn[j]

    #############################
    # Context
    #############################

    def make_train_session(self, device, dataloader, fold_tag="", model_file=None):
        """Create train session object
        :return:Session
        """

        train_dl = dataloader["train"]
        valid_dl = None
        if "valid" in dataloader:
            valid_dl = dataloader["valid"]
        return Session(self, device=device,
                       train_dl=train_dl,
                       valid_dl=valid_dl,
                       fold_tag=fold_tag,
                       uuid=self.uuid,
                       logging_file=self.logging_file,
                       progress=self.progress,
                       model_file=model_file,
                       session_type="train")

    #############################
    # flow
    #############################

    def init_state_fn(self):
        """init status
        :return: ObjectDict

        """
        return ObjectDict()

    def epoch_start_fn(self, session: Session):
        """epoch train start
        :param session:
        :return:
        """
        pass

    def epoch_train_end_fn(self, session: Session):
        """ epoch train end ,before valid start
        :param session:
        :return:
        """
        pass

    def valid_end_fn(self, session: Session):
        """
        valid end and then one epoch end
        :param session:
        :return:
        """
        pass

        #############################
        # wapper pytorch functionality
        #############################

    def init_batch_data_fn(self, data, session: Session):
        '''
        Process datafrom DataLoader
        :param data:
        :param session:
        :return:
        '''
        session.state.input_batch, session.state.target_batch = data

    def forward_fn(self, session: Session):
        """
         move data  to the device and forward propagation
        :param session:
        :return:
        """
        x = session.state.input_batch.to(session.device)
        session.state.output_batch = session.model(x)



    def loss_fn(self, session: Session):
        """
        Calculate the loss function
        :param session:
        :return:
        """
        y = session.state.target_batch.to(session.device)
        session.state.loss = session.criterion(session.state.output_batch, y)


    def backward_fn(self, session: Session):
        """
        backward propagation and Optimizer step
        :param session:
        :return:
        """
        gradients = torch.ones_like(session.state.loss)
        session.state.loss.backward(gradients)
        # Gradient Accumulation
        if self.ga_steps(session):
            session.optimizer.step()
            session.optimizer.zero_grad()

    @torch.no_grad()
    def eval_fn(self, session: Session):
        """eval for valid data only forward
        :param session:
        :return:
        """
        self.forward_fn(session)
        self.loss_fn(session)

    #############################
    # additional functionality
    #############################
    def cal_metric_fn(self, session: Session):
        """Calculate the metric using metric_fn and save it to session
        :param session:
        :return:
        """
        loss = session.state.loss.cpu().detach()
        target=session.state.target_batch.cpu().detach()
        output=session.state.output_batch.cpu().detach()
        metric = {}

        if len(loss.size()) == 0:
            metric["loss"] = loss.item()
        else:
            metric["loss"] = torch.mean(loss).item()
        if session.stage in ["train","valid"]:
            if session.batch_data[session.stage]["output"] is None:
                session.batch_data[session.stage]["output"]=output
            else:
                session.batch_data[session.stage]["output"]=torch.cat([session.batch_data[session.stage]["output"],output],dim=0)

            if session.batch_data[session.stage]["target"] is None:
                session.batch_data[session.stage]["target"]=target
            else:
                session.batch_data[session.stage]["target"]=torch.cat([session.batch_data[session.stage]["target"],target],dim=0)

            session.batch_data[session.stage]["loss"].append(metric["loss"])


        session.state.metric = metric

    def ga_steps(self, session: Session):
        if self.ga_step_size > 1:
            bn = session.state.current_batch + 1
            if bn % self.ga_step_size == 0 or bn >= session.train_dl_len:
                return True
            return False
        else:
            return True

    def scheduler_step_fn(self, session: Session):
        """By default, scheduler.step() is called without passing arguments.
        If you need to pass arguments, override this method
        :param session:
        :return:
        """
        session.scheduler.step()

    def early_stopping_fn(self, session: Session):
        """Calculates metric and returns Ture for early stop
        :param session:
        :return: bool
        """
        return False

    def get_lr(self, session: Session):
        """
        logging learning rate
        :return:
        """
        last_lr = []
        if session.scheduler and hasattr(session.scheduler, "get_last_lr"):
            last_lr = session.scheduler.get_last_lr()
        else:
            for param_group in session.optimizer.param_groups:
                last_lr.append(param_group['lr'])
        session.state.lr = last_lr

import numpy as np
import torch

from AlignVelo.basetrainer import BaseTrainer
from AlignVelo._utils import inf_loop, MetricTracker

class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        optimizer,
        config,
        data_loader,
        valid_data_loader=None,
        lr_scheduler=None,
        lr_scheduler2=None,
        len_epoch=None,
    ):
        super().__init__(model, optimizer, config)
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler2 = lr_scheduler2
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker("loss", writer=self.writer)
        self.valid_metrics = MetricTracker("loss", writer=self.writer)
        self.saved_candidate_ids = {}

    def _compute_core(self, batch_data):
        data_dict = batch_data
        x_u, x_s, batch, mask_u = data_dict["Ux_sz"], data_dict["Sx_sz"], data_dict["batch"], data_dict["mask_u"]
        x_u, x_s, batch, mask_u = (
            x_u.to(self.device),
            x_s.to(self.device),
            batch.to(self.device),
            mask_u.to(self.device)
        )
        inputdata=torch.cat([x_s,x_u],dim=1)
        loss, pseudotime, pred = self.model(inputdata, batch, mask_u)
        return loss, pseudotime, pred

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        if (not self.data_loader.shuffle) and self.data_loader.is_large_batch:
            loader = self.data_loader.dataset.large_batch(self.device)
        else:
            loader = self.data_loader
        
        for batch_idx, batch_data in enumerate(loader):
            loss, pseudotime, mean_pred = self._compute_core(batch_data)
           
            self.optimizer.zero_grad()
            
            loss.backward()
            if self.config["trainer"].get("grad_clip", True):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update("loss", loss.item())

            
            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), loss.item()
                    )
                )
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})

        if self.model.loss_select in ["loss1", "loss2"]:
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        else:
            if self.lr_scheduler2 is not None:
                self.lr_scheduler2.step()
        return log
    
    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.valid_data_loader):
                loss, pseudotime, mean_pred = self._compute_core(batch_data)

                self.writer.set_step(
                    (epoch - 1) * len(self.valid_data_loader) + batch_idx, "valid"
                )
                self.valid_metrics.update("loss", loss.item())

        # add histogram of model parameters to the tensorboard
        #for name, p in self.model.named_parameters():
        #    self.writer.add_histogram(name, p, bins="auto")
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _get_latent(self,
                    batch_data):
        data_dict = batch_data
        x_u, x_s, batch, mask_u = data_dict["Ux_sz"], data_dict["Sx_sz"], data_dict["batch"], data_dict["mask_u"]
        x_u, x_s, batch, mask_u = (
            x_u.to(self.device),
            x_s.to(self.device),
            batch.to(self.device),
            mask_u.to(self.device)
        )
        inputdata=torch.cat([x_s,x_u],dim=1)
        
        inference_outputs = self.model.inference(inputdata,batch)
        z=inference_outputs["z"]
        alpha=inference_outputs["alpha"]
        x_pre=inference_outputs["x_pre"]

        return z, alpha, x_pre

    def eval(self, eval_loader, return_kinetic_rates=False):
        """
        Evaluate the model on a given dataset, provided by eval_loader

        :param model: model to evaluate
        :param eval_loader: dataset loader containing dataset to evaluate on
        """
        self.model.eval()
        n_gene = self.config["arch"]["args"]["n_gene"]
        velo_mat = []
        velo_mat_u = []
        pseudotime = []
        z = []
        x_pre = []
        kinetic_rates = {}
        alpha_rates = []
        if (not eval_loader.shuffle) and eval_loader.is_large_batch:
            loader = eval_loader.dataset.large_batch(self.device)
        else:
            loader = eval_loader
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(loader):
                loss, pseudotime_, mean_pred= self._compute_core(batch_data)
                z_, alpha_, x_pre_ = self._get_latent(batch_data)
                z.append(z_.cpu().data)
                x_pre.append(x_pre_.cpu().data)
                if return_kinetic_rates:
                    alpha_rates.append(alpha_.cpu().data)

                pred_s = mean_pred[:, 0:n_gene]
                velo_mat.append(pred_s.cpu().data)
                pseudotime.append(pseudotime_.cpu().data)
                if self.config["arch"]["args"]["pred_unspliced"]:
                    pred_u = mean_pred[:, n_gene:n_gene*2]
                    velo_mat_u.append(pred_u.cpu().data)

            velo_mat = np.concatenate(velo_mat, axis=0)
            pseudotime = np.concatenate(pseudotime, axis=0)
            alpha_rates = np.concatenate(alpha_rates, axis=0)
            z = np.concatenate(z, axis=0)
            x_pre = np.concatenate(x_pre, axis=0)
            if self.config["arch"]["args"]["pred_unspliced"]:
                velo_mat_u = np.concatenate(velo_mat_u, axis=0)
        s=x_pre[:, :n_gene]
        u=x_pre[:, n_gene:n_gene*2]
        from AlignVelo.loss import direction_loss
        loss_pearson, reverse = direction_loss(
                        velocity=torch.tensor(velo_mat).to(self.device),
                        spliced_counts=torch.tensor(s).to(self.device),
                        unspliced_counts=torch.tensor(u).to(self.device),
                        reduce=False,
                        )
        print(f"Loss_Pearson:{loss_pearson}")
        if reverse:
            velo_mat = - velo_mat
            pseudotime = 1 - pseudotime
            print("Reverse pseudotime and velocity for the negative direction loss.")
            if self.config["arch"]["args"]["pred_unspliced"]:
                velo_mat_u = -velo_mat_u
        # record kinetic rates
        if return_kinetic_rates:
            cur_kinetic_rates = self.model.get_kinetic_rates()
            for k, v in cur_kinetic_rates.items():
                if k not in kinetic_rates:
                    kinetic_rates[k] = []
                kinetic_rates[k].append(v.cpu().data)
        if return_kinetic_rates:
            for k, v in kinetic_rates.items():
                kinetic_rates[k] = np.concatenate(v, axis=0)

        return velo_mat, velo_mat_u, alpha_rates, kinetic_rates, pseudotime, z , u, s
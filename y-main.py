import argparse
import logging
import os
import pprint
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

import datasets
import lib.utils as utils
import losses
import models
from evaluation.evaler import Evaler
from lib.config import cfg, cfg_from_file
from optimizer.optimizer import Optimizer
from scorer.scorer import Scorer


class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        if cfg.SEED > 0:
            random.seed(cfg.SEED)
            torch.manual_seed(cfg.SEED)
            torch.cuda.manual_seed_all(cfg.SEED)

        self.num_gpus = torch.cuda.device_count()
        self.distributed = self.num_gpus > 1

        #self.tb_summary_writer = None
        if self.distributed:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(backend="nccl",
                                                 init_method="env://")
            if args.local_rank == 0:
                self.tb_summary_writer = SummaryWriter(args.folder)
        else:
            self.tb_summary_writer = SummaryWriter(args.folder)
        self.device = torch.device("cuda")

        self.setup_logging()
        self.setup_dataset()
        self.setup_network()
        self.val_evaler = Evaler(eval_ids=cfg.DATA_LOADER.VAL_ID,
                                 gv_feat=cfg.DATA_LOADER.VAL_GV_FEAT,
                                 att_feats=cfg.DATA_LOADER.VAL_ATT_FEATS,
                                 eval_annfile=cfg.INFERENCE.VAL_ANNFILE)
        self.test_evaler = Evaler(eval_ids=cfg.DATA_LOADER.TEST_ID,
                                  gv_feat=cfg.DATA_LOADER.TEST_GV_FEAT,
                                  att_feats=cfg.DATA_LOADER.TEST_ATT_FEATS,
                                  eval_annfile=cfg.INFERENCE.TEST_ANNFILE)
        self.scorer = Scorer()
        self._init_ppo()

    def _init_ppo(self):
        self.nenvs = 128
        self.noptepochs = 2
        self.envsperbatch = 32
        self.clip_range = 0.1
        cfg.SOLVER.TEST_INTERVAL = 100
        assert self.nenvs % cfg.TRAIN.BATCH_SIZE == 0
        assert cfg.SOLVER.TEST_INTERVAL % (self.nenvs //
                                           cfg.TRAIN.BATCH_SIZE) == 0
        self.batch_next = None
        self.num_samples = cfg.DATA_LOADER.SEQ_PER_IMG

    def setup_logging(self):
        self.logger = logging.getLogger(cfg.LOGGER_NAME)
        self.logger.setLevel(logging.INFO)
        if self.distributed and dist.get_rank() > 0:
            return

        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(levelname)s: %(asctime)s] %(message)s")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if not os.path.exists(cfg.ROOT_DIR):
            os.makedirs(cfg.ROOT_DIR)

        fh = logging.FileHandler(
            os.path.join(cfg.ROOT_DIR, cfg.LOGGER_NAME + '.txt'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.info('Training with config:')
        self.logger.info(pprint.pformat(cfg))

    def setup_network(self):
        model = models.create(cfg.MODEL.TYPE)

        if self.distributed:
            # this should be removed if we update BatchNorm stats
            self.model = torch.nn.parallel.DistributedDataParallel(
                model.to(self.device),
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                broadcast_buffers=False)
        else:
            self.model = torch.nn.DataParallel(model).cuda()

        if self.args.resume > 0:
            self.model.load_state_dict(
                torch.load(self.snapshot_path("caption_model",
                                              self.args.resume),
                           map_location=lambda storage, loc: storage))

        self.optim = Optimizer(self.model)
        self.rl_criterion = losses.create(cfg.LOSSES.RL_TYPE).cuda()

    def setup_dataset(self):
        self.coco_set = datasets.coco_dataset.CocoDataset(
            image_ids_path=cfg.DATA_LOADER.TRAIN_ID,
            input_seq=cfg.DATA_LOADER.INPUT_SEQ_PATH,
            target_seq=cfg.DATA_LOADER.TARGET_SEQ_PATH,
            gv_feat_path=cfg.DATA_LOADER.TRAIN_GV_FEAT,
            att_feats_folder=cfg.DATA_LOADER.TRAIN_ATT_FEATS,
            seq_per_img=cfg.DATA_LOADER.SEQ_PER_IMG,
            max_feat_num=cfg.DATA_LOADER.MAX_FEAT)

    def setup_loader(self, epoch):
        self.training_loader = datasets.data_loader.load_train(
            self.distributed, epoch, self.coco_set)

    def snapshot_path(self, name, epoch):
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        return os.path.join(snapshot_folder, name + "_" + str(epoch) + ".pth")

    def save_model(self, epoch):
        if epoch % cfg.SOLVER.SNAPSHOT_ITERS != 0:
            return
        if self.distributed and dist.get_rank() > 0:
            return
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        if not os.path.exists(snapshot_folder):
            os.mkdir(snapshot_folder)
        torch.save(self.model.state_dict(),
                   self.snapshot_path("caption_model", epoch))

    def get_batch(self):
        epoch = 0
        while True:
            self.setup_loader(epoch)
            for x in self.training_loader:
                yield epoch, x
            epoch += 1

    def _expand_kwargs(self, kwargs):
        indices, gv_feat, att_feats, att_mask = [
            kwargs[_] for _ in [
                cfg.PARAM.INDICES, cfg.PARAM.GLOBAL_FEAT, cfg.PARAM.ATT_FEATS,
                cfg.PARAM.ATT_FEATS_MASK
            ]
        ]
        indices = utils.expand_numpy(indices, self.num_samples)
        gv_feat, att_feats, att_mask = [
            utils.expand_tensor(_, self.num_samples)
            for _ in [gv_feat, att_feats, att_mask]
        ]
        # sample
        kwargs['BEAM_SIZE'] = 1
        kwargs['GREEDY_DECODE'] = False
        for x, y in zip([
                cfg.PARAM.INDICES, cfg.PARAM.GLOBAL_FEAT, cfg.PARAM.ATT_FEATS,
                cfg.PARAM.ATT_FEATS_MASK
        ], [indices, gv_feat, att_feats, att_mask]):
            kwargs[x] = y

        return kwargs

    def _sample_trajectory(self, kwargs):
        kwargs = self._expand_kwargs(kwargs)

        with torch.no_grad():
            seq_sample, log_prob_sample = self.model.module.decode(**kwargs)
            seq_sample_list = seq_sample.detach().cpu().numpy().tolist()
            log_prob_sample, seq_sample = [
                _.detach().cpu().numpy().reshape(
                    (cfg.TRAIN.BATCH_SIZE, self.num_samples,
                     cfg.MODEL.SEQ_LEN))
                for _ in [log_prob_sample, seq_sample]
            ]

        rewards_sample, _ = self.scorer(kwargs[cfg.PARAM.INDICES],
                                        seq_sample_list)
        rewards_sample = rewards_sample.reshape(
            (-1, self.num_samples))
        rewards_avg = np.expand_dims(rewards_sample.mean(-1), axis=1)
        rewards = rewards_sample - rewards_avg
        #print(rewards.shape)
        #exit(0)
        #rewards = (rewards - rewards.mean(-1)[..., None]) / (
        #    rewards.std(-1)[..., None] + 1e-8)

        return seq_sample, log_prob_sample, rewards

    def runner_run(self, iteration):
        mb_indices = []
        mb_gv_feat = []
        mb_att_feats = []
        mb_att_mask = []

        mb_sample_logprobs = []
        mb_gen_result = []
        mb_advs = []

        for _ in range(self.nenvs // cfg.TRAIN.BATCH_SIZE):
            # data - indices, input_seq, target_seq, gv_feat, att_feats, att_mask
            epoch, data = next(self.batch_next)
            iteration += 1

            indices = data[0]
            mb_indices.append(indices.reshape(-1, 1))
            for x, y in zip(data[-3:],
                            [mb_gv_feat, mb_att_feats, mb_att_mask]):
                y.append(x.numpy())

            gv_feat, att_feats, att_mask = [_.cuda() for _ in data[-3:]]

            kwargs = {
                cfg.PARAM.INDICES: indices,
                cfg.PARAM.GLOBAL_FEAT: gv_feat,
                cfg.PARAM.ATT_FEATS: att_feats,
                cfg.PARAM.ATT_FEATS_MASK: att_mask
            }
            seq_sample, log_prob_sample, rewards = self._sample_trajectory(
                kwargs)
            trajectory = [log_prob_sample, seq_sample, rewards]
            for x, y in zip(trajectory,
                            [mb_sample_logprobs, mb_gen_result, mb_advs]):
                y.append(x)

        max_att_num = np.max([_.shape[1] for _ in mb_att_feats])
        for k, x in enumerate(mb_att_feats):
            after = max_att_num - x.shape[1]
            mb_att_feats[k] = np.pad(x, ((0, 0), (0, after), (0, 0)),
                                     mode="constant")
            mb_att_mask[k] = np.pad(mb_att_mask[k], ((0, 0), (0, after)),
                                    mode="constant")

        mb_indices, mb_gv_feat, mb_att_feats, mb_att_mask, \
        mb_sample_logprobs, mb_gen_result, mb_advs = [np.vstack(_) for _ in [
            mb_indices, mb_gv_feat, mb_att_feats, mb_att_mask,
            mb_sample_logprobs, mb_gen_result, mb_advs
        ]]
        return iteration, epoch, (mb_indices, mb_gv_feat, mb_att_feats,
                                  mb_att_mask, mb_sample_logprobs,
                                  mb_gen_result, mb_advs)

    def mb_train(self, kwargs):
        ent_coef = 0.01
        kwargs = self._expand_kwargs(kwargs)
        _, neglogpac = self.model.module.decode(**kwargs)

        sample_logprobs, gen_result, advs = [
            kwargs[_] for _ in
            [cfg.PARAM.SAMPLE_LOGPROBS, cfg.PARAM.GEN_RESULT, cfg.PARAM.ADVS]
        ]

        trajectory = [sample_logprobs, gen_result]
        for k, _ in enumerate(trajectory):
            trajectory[k] = _.view(-1, cfg.MODEL.SEQ_LEN)
        sample_logprobs, gen_result = trajectory
        advs = advs.view(-1, 1).expand_as(gen_result)

        mask = gen_result > 0
        mask = torch.cat(
            [mask.new_full((mask.shape[0], 1), True), mask[:, :-1]], 1)
        entropy = torch.sum(torch.exp(neglogpac) * (-neglogpac), dim=-1)
        entropy = entropy[mask].mean()
        neglogpac = torch.gather(neglogpac, 2,
                                 gen_result.unsqueeze(-1)).squeeze(-1)

        neglogpac = -torch.masked_select(neglogpac, mask)
        oldneglogpac = -torch.masked_select(sample_logprobs, mask)
        advs = torch.masked_select(advs, mask)

        ratio = torch.exp(oldneglogpac - neglogpac)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        pg_losses = -advs * ratio
        pg_losses2 = -advs * torch.clamp(ratio, 1.0 - self.clip_range,
                                         1.0 + self.clip_range)
        pg_loss = torch.max(pg_losses, pg_losses2).mean()
        #loss = pg_loss - entropy * ent_coef
        loss = pg_loss

        return loss

    def train(self):
        self.batch_next = self.get_batch()
        # eval - crucial to disable dropout
        self.model.eval()
        # DDP - want zero_grad() before backward?
        self.optim.zero_grad()

        epoch, iteration = 0, 0
        while True:
            if epoch >= cfg.SOLVER.MAX_EPOCH:
                break
            iteration, epoch_this, data = self.runner_run(iteration)

            envinds = np.arange(self.nenvs)
            for _ in range(self.noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, self.nenvs, self.envsperbatch):
                    end = start + self.envsperbatch
                    mbenvinds = envinds[start:end]
                    indices = data[0][mbenvinds].reshape(-1)
                    gv_feat, att_feats, att_mask, sample_logprobs, gen_result, advs = \
                        [torch.from_numpy(x).cuda()
                         for x in [_[mbenvinds] for _ in data[1:]]]

                    kwargs = {
                        cfg.PARAM.INDICES: indices,
                        cfg.PARAM.GLOBAL_FEAT: gv_feat,
                        cfg.PARAM.ATT_FEATS: att_feats,
                        cfg.PARAM.ATT_FEATS_MASK: att_mask,
                        cfg.PARAM.SAMPLE_LOGPROBS: sample_logprobs,
                        cfg.PARAM.GEN_RESULT: gen_result,
                        cfg.PARAM.ADVS: advs
                    }
                    loss = self.mb_train(kwargs)

                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   0.5, 2)
                    #utils.clip_gradient(self.optim.optimizer, self.model,
                    #                    cfg.SOLVER.GRAD_CLIP_TYPE,
                    #                    cfg.SOLVER.GRAD_CLIP)
                    self.optim.step()
                    self.optim.zero_grad()
                    # to cool down GPU
                    #time.sleep(2.5)

            if iteration % cfg.SOLVER.TEST_INTERVAL == 0:
                self.period_checkpoint(iteration, epoch)
            if epoch_this > epoch:
                epoch = epoch_this
                self.save_model(epoch)
                self._optim_step(iteration, epoch)
            if self.distributed:
                dist.barrier()

    def _optim_step(self, iteration, epoch):
        if self.distributed and dist.get_rank() > 0:
            return
        val_res = self.val_evaler(self.model, 'val_' + str(iteration))

        # self.test_evaler or self.val_evaler will call `model.train()'
        # critical to reset `model.eval()'
        # eval - crucial to disable dropout
        self.model.eval()
        self.logger.info('######## Iteration (VAL) ' + str(iteration) +
                         ' ########')
        self.logger.info(str(val_res))
        val = 0
        for score_type, weight in zip(cfg.SCORER.TYPES, cfg.SCORER.WEIGHTS):
            val -= val_res[score_type] * weight
            self.tb_summary_writer.add_scalar("val_" + score_type,
                                              val_res[score_type], iteration)
        self.optim.scheduler_step('Epoch', val)
        self.tb_summary_writer.add_scalar("learning_rate",
                                          self.optim.get_lr()[0], iteration)
        self.tb_summary_writer.add_scalar("epoch", epoch, iteration)

    def period_checkpoint(self, iteration, epoch):
        if self.distributed and dist.get_rank() > 0:
            return
        self.logger.info("iteration {}".format(iteration))
        # val_res = self.val_evaler(self.model, 'val_' + str(iteration))
        # self.logger.info('######## Epoch (VAL) ' + str(iteration) +
        #                 ' ########')
        # self.logger.info(str(val_res))

        test_res = self.test_evaler(self.model, 'test_' + str(iteration))
        # self.test_evaler or self.val_evaler will call `model.train()'
        # critical to reset `model.eval()'
        # eval - crucial to disable dropout
        self.model.eval()
        self.logger.info('######## Iteration (TEST) ' + str(iteration) +
                         ' ########')
        self.logger.info(str(test_res))

        # val = 0
        for score_type, weight in zip(cfg.SCORER.TYPES, cfg.SCORER.WEIGHTS):
            # val -= val_res[score_type] * weight
            # tb_summary_writer.add_scalar("Val " + score_type, val_res[score_type], iteration)
            self.tb_summary_writer.add_scalar("test_" + score_type,
                                              test_res[score_type], iteration)
        self.tb_summary_writer.add_scalar("learning_rate",
                                          self.optim.get_lr()[0], iteration)
        self.tb_summary_writer.add_scalar("epoch", epoch, iteration)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--folder', dest='folder', type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume", type=int, default=-1)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    if args.folder is not None:
        cfg_from_file(os.path.join(args.folder, 'config.yml'))
    cfg.ROOT_DIR = args.folder

    trainer = Trainer(args)
    trainer.train()

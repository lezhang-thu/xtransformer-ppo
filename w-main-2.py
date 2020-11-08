import argparse
import logging
import os
import pprint
import sys
import time

import datasets
import lib.utils as utils
import losses
import models
import numpy as np
import torch
import torch.distributed as dist
from evaluation.evaler import Evaler
from lib.config import cfg, cfg_from_file
from optimizer.optimizer import Optimizer
from scorer.scorer import Scorer


class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args

        self.num_gpus = torch.cuda.device_count()
        self.distributed = self.num_gpus > 1

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
        self.nenvs = 4 * cfg.TRAIN.BATCH_SIZE
        self.noptepochs = 1
        self.envsperbatch = cfg.TRAIN.BATCH_SIZE
        self.clip_range = 0.1
        assert self.nenvs % cfg.TRAIN.BATCH_SIZE == 0
        self.batch_next = None
        self.beta = 1.0
        self.dtarg = 0.01

        self.mv_approxkl = 0
        self.mv_violate = 0
        self.mv_entropy = 0
        self.mv_total = 0
        #self.mv_pg_loss = 0

        #self.mv_loss = 0

        self.x_dataset = datasets.coco_dataset.CocoDataset(
            image_ids_path=cfg.DATA_LOADER.TRAIN_ID,
            input_seq=cfg.DATA_LOADER.INPUT_SEQ_PATH,
            target_seq=cfg.DATA_LOADER.TARGET_SEQ_PATH,
            gv_feat_path=cfg.DATA_LOADER.TRAIN_GV_FEAT,
            att_feats_folder=cfg.DATA_LOADER.TRAIN_ATT_FEATS,
            seq_per_img=1,
            max_feat_num=cfg.DATA_LOADER.MAX_FEAT)

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
        self.predictor = torch.nn.DataParallel(models.create(
            cfg.MODEL.TYPE)).cuda()
        self.predictor.module.load_state_dict(self.model.module.state_dict())

        # freeze encoder
        #self.freeze_encoder()
        self.bn = torch.nn.BatchNorm1d(num_features=1,
                                       track_running_stats=False).cuda()

        self.optim = Optimizer(self.model, self.bn)
        self.rl_criterion = losses.create(cfg.LOSSES.RL_TYPE).cuda()

    def setup_dataset(self):
        self.coco_set = datasets.coco_dataset.CocoDataset(
            image_ids_path=cfg.DATA_LOADER.TRAIN_ID,
            input_seq=cfg.DATA_LOADER.INPUT_SEQ_PATH,
            target_seq=cfg.DATA_LOADER.TARGET_SEQ_PATH,
            gv_feat_path=cfg.DATA_LOADER.TRAIN_GV_FEAT,
            att_feats_folder=cfg.DATA_LOADER.TRAIN_ATT_FEATS,
            seq_per_img=1,
            max_feat_num=cfg.DATA_LOADER.MAX_FEAT)

    def setup_loader(self, epoch):
        self.training_loader = datasets.data_loader.load_train(
            self.distributed, epoch, self.coco_set)

    def snapshot_path(self, name, epoch):
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        return os.path.join(snapshot_folder, name + "_" + str(epoch) + ".pth")

    def save_model(self, iteration):
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        if not os.path.exists(snapshot_folder):
            os.mkdir(snapshot_folder)
        torch.save(self.model.state_dict(),
                   self.snapshot_path("caption_model", iteration))

    def get_batch(self):
        epoch = 0
        while True:
            self.setup_loader(epoch)
            for x in self.training_loader:
                yield epoch, x
            epoch += 1

    def _set_kwargs(self, kwargs, repeat_factor):
        # sample
        kwargs['GREEDY_DECODE'] = False
        kwargs["NEED_PD"] = True
        kwargs["REPEAT_FACTOR"] = repeat_factor

        return kwargs

    def _prefix_rewards(self, kwargs, seq_prefix):
        kwargs[cfg.PARAM.MAX_GEN_LEN] = seq_prefix.shape[-1]
        kwargs[cfg.PARAM.GEN_RESULT] = utils.expand_tensor(seq_prefix, 5)
        with torch.no_grad():
            seq_sample = self.predictor.module.extend_trajectory(
                **kwargs).detach().cpu().numpy().tolist()
        return seq_sample

    def _sample_trajectory(self, kwargs):
        self._set_kwargs(kwargs, 1)
        with torch.no_grad():
            seq_sample, log_prob_sample = self.predictor.module.decode(
                **kwargs)
            seq_sample_list = seq_sample.detach().cpu().numpy().tolist()
            log_prob_sample, seq_sample = [
                _.detach().cpu().numpy()
                for _ in [log_prob_sample, seq_sample]
            ]
        indices = kwargs[cfg.PARAM.INDICES]
        #print("seq_sample.shape: {}, log_prob_sample.shape: {}".format(
        #    seq_sample.shape, log_prob_sample.shape))

        rewards_sample, _ = self.scorer(indices, seq_sample_list)
        #print("rewards_sample.shape: {}".format(rewards_sample.shape))
        #print("rewards_sample: {}".format(rewards_sample))
        #exit(0)

        repeat_factor = 5
        kwargs = self._set_kwargs(kwargs, repeat_factor)
        kwargs['gx'], kwargs['encoder_out'], kwargs['p_att_feats'] , kwargs['att_mask']= \
                self.predictor.module.init_gx_encoder_out_p_att_feats_att_mask(**kwargs)
        indices = utils.expand_numpy(kwargs[cfg.PARAM.INDICES], repeat_factor)
        advs = np.zeros_like(seq_sample, dtype=np.float32)
        for k in range(cfg.MODEL.SEQ_LEN):
            baseline, _ = self.scorer(
                indices,
                self._prefix_rewards(
                    kwargs,
                    torch.from_numpy(seq_sample[:, :k]).cuda()))
            baseline = baseline.reshape(-1, repeat_factor)
            #print("k: {}\tbaseline[0]: {}".format(k, baseline[0]))
            advs[:, k] = rewards_sample - baseline.mean(-1)
            if seq_sample[:, k].sum() == 0:
                break

        #exit(0)
        seq_sample = seq_sample[:, None, :]
        #advs = np.clip(advs[:, None, :], -1, 1)
        advs = advs[:, None, :]
        log_prob_sample = log_prob_sample[:, None, ...]
        return seq_sample, log_prob_sample, advs

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
        # special code - start
        #mb_advs /= sqrt(self.nenvs)
        mb_advs = np.clip(mb_advs, -1, 1)
        # special code - end

        return iteration, epoch, [
            mb_indices, mb_gv_feat, mb_att_feats, mb_att_mask,
            mb_sample_logprobs, mb_gen_result, mb_advs
        ]

    def freeze_encoder(self):
        x = self.model.module.encoder
        for param in x.parameters():
            param.requires_grad = False

    def mb_train(self, kwargs):
        ent_coef = 1e-4
        kwargs = self._set_kwargs(kwargs, 1)
        _, neglogpac = self.model.module.decode(**kwargs)

        sample_logprobs, gen_result, advs = [
            kwargs[_] for _ in
            [cfg.PARAM.SAMPLE_LOGPROBS, cfg.PARAM.GEN_RESULT, cfg.PARAM.ADVS]
        ]

        trajectory = [sample_logprobs, gen_result, advs]
        for k, _ in enumerate(trajectory):
            trajectory[k] = _.view(-1, *_.shape[2:])
        sample_logprobs, gen_result, advs = trajectory

        mask = gen_result > 0
        mask = torch.cat(
            [mask.new_full((mask.shape[0], 1), True), mask[:, :-1]], 1)

        kl_div = torch.exp(sample_logprobs) * (sample_logprobs - neglogpac)
        kl_div = kl_div.sum(-1)
        kl_div = torch.masked_select(kl_div, mask)
        entropy = torch.sum(torch.exp(neglogpac) * (-neglogpac), dim=-1)
        entropy = entropy[mask].mean()
        neglogpac = torch.gather(neglogpac, 2,
                                 gen_result.unsqueeze(-1)).squeeze(-1)

        sample_logprobs = torch.gather(sample_logprobs, 2,
                                       gen_result.unsqueeze(-1)).squeeze(-1)
        advs_close_zero = (-1e-5 < advs) & (advs < 1e-5)
        mask &= ~advs_close_zero
        neglogpac = -torch.masked_select(neglogpac, mask)
        oldneglogpac = -torch.masked_select(sample_logprobs, mask)
        advs = torch.masked_select(advs, mask)

        #print("advs.shape: {}".format(advs.shape))
        #print("advs[:10]: {}".format(advs[:10]))
        #time.sleep(1)

        ratio = torch.exp(oldneglogpac - neglogpac)
        #print("advs.mean(): {}".format(advs.mean()))
        #print("type(advs): {}".format(type(advs)))
        #exit(0)
        #advs = self.bn(advs.view(-1, 1).float()).view(-1)
        #advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        #advs = (advs - advs.mean()) / (advs.max() - advs.min() + 1e-8)
        #advs = torch.clamp(advs, -1, 1)
        #print("advs: {}".format(advs))
        #time.sleep(1)
        #print("advs.std(unbiased=False): {}".format(advs.std(unbiased=False)))
        #advs = advs / (advs.std() + 1e-8)
        pg_losses = -advs * ratio
        pg_losses2 = -advs * torch.clamp(ratio, 1.0 - self.clip_range,
                                         1.0 + self.clip_range)
        self.mv_total = 0.9 * self.mv_total + 0.1 * pg_losses.shape[0]
        #pg_loss = torch.max(pg_losses, pg_losses2).mean()
        pg_loss = torch.max(pg_losses,
                            pg_losses2).sum() / (cfg.TRAIN.BATCH_SIZE * 16)
        mask_positive = (advs > 0) & (ratio > 1 + self.clip_range)
        mask_negative = (advs < 0) & (ratio < 1 - self.clip_range)
        mask_total = mask_positive | mask_negative
        ##print("pg_loss.shape: {}".format(pg_loss.shape))
        #exit(0)
        #pg_loss = pg_loss.sum() / max(
        #    pg_loss.shape[0] - mask_total.detach().sum(), 1)
        # pg_loss = pg_losses.mean()
        kl_div = kl_div.mean()
        # mask_positive = (advs > 0) & (ratio > 1 + 0.2)
        # mask_negative = (advs < 0) & (ratio < 1 - 0.2)
        # mask_total = mask_positive | mask_negative
        #self.mv_violate = 0.9 * self.mv_violate + 0.1 * mask_total.sum().item()
        # if self.mv_violate < 1:
        #    self.mv_violate = 1
        # kl_div = kl_div[mask_total].sum() / self.mv_violate

        # if mask_total.sum().item() > 0:
        #    approxkl = .5 * torch.mean(
        #        torch.square((neglogpac - oldneglogpac)[mask_total]))
        # else:
        #    approxkl = torch.tensor(0)
        # approxkl = .5 * torch.square((neglogpac - oldneglogpac)[mask_total]).sum() / advs.shape[0]
        # loss = pg_loss - entropy * ent_coef + self.beta * kl_div
        #loss = pg_loss - ent_coef * entropy
        loss = pg_loss
        # if kl_div.item() < self.dtarg / 1.5:
        #    self.beta /= 2
        # if kl_div.item() > self.dtarg * 1.5:
        #    self.beta *= 2
        # self.beta = max(min(self.beta, 100), 0.5)
        self.mv_approxkl = 0.9 * self.mv_approxkl + 0.1 * kl_div.item()
        self.mv_entropy = 0.9 * self.mv_entropy + 0.1 * entropy.item()
        self.mv_violate = 0.9 * self.mv_violate + 0.1 * mask_total.sum().item()

        #self.mv_pg_loss = 0.9 * self.mv_pg_loss + 0.1 * pg_loss.item()
        #self.mv_loss = 0.9 * self.mv_loss + 0.1 * loss.item()
        return loss

    def process_advs(self, gen_result, advs):
        advs = advs.reshape(-1)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        return advs.reshape(self.nenvs, 5)
        gen_result = gen_result.reshape(-1, gen_result.shape[-1])
        advs = advs.reshape(-1, 1).repeat(gen_result.shape[-1], axis=-1)
        mask = gen_result > 0
        mask = np.concatenate((np.full(
            (mask.shape[0], 1), True), mask[:, :-1]), 1)
        x = advs[mask]
        print("x.mean(): {}".format(x.mean()))
        time.sleep(1)
        advs[mask] = np.clip((x - x.mean()) / (x.std() + 1e-8), -1, 1)
        advs[mask] = (x - x.mean()) / (x.std() + 1e-8)
        return advs.reshape(self.nenvs, 5, gen_result.shape[-1])

    def train(self):
        self.batch_next = self.get_batch()
        # eval - crucial to disable dropout
        self.model.eval()
        # self.model.train()
        self.predictor.eval()
        self.optim.zero_grad()

        epoch, iteration = 0, 0
        val_current = None
        val_best = None
        print("val @iteration 0: {}".format(
            self._compute_val(iteration, val_best)))
        while True:
            iteration, epoch, data = self.runner_run(iteration)

            #data[-1] = self.process_advs(data[-2], data[-1])
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
                    self.optim.step()
                    self.optim.zero_grad()
                    time.sleep(2)

            #print("iteration {}".format(iteration))
            if iteration % 32 == 0:
                self.predictor.module.load_state_dict(
                    self.model.module.state_dict())
                if iteration < 896 * 2 and iteration % 128 == 0:
                    self.logger.info("val_current @iteration {}: {}".format(
                        iteration, self._compute_val(iteration, val_best)))

            if iteration >= 896 * 2 and iteration % 32 == 0:
                val_current = self._compute_val(iteration, val_best)
                if val_best is None:
                    val_best = val_current

                self.logger.info(
                    "val_current @iteration {}: {}, val_predictor: {}".format(
                        iteration, val_current, val_best))
                self.logger.info("mv_approxkl: {}".format(self.mv_approxkl))
                self.logger.info("mv_entropy: {}".format(self.mv_entropy))
                self.logger.info("mv_violate: {}".format(self.mv_violate))
                self.logger.info("mv_total: {}".format(self.mv_total))

                if val_best <= val_current:
                    self.save_model(23)
                    val_best = val_current

    def _compute_val(self, iteration, val_best):
        val_res = self.val_evaler(self.model, 'val_' + str(iteration), 1)
        val = 0
        for score_type, weight in zip(cfg.SCORER.TYPES, cfg.SCORER.WEIGHTS):
            # plus!!!
            val += val_res[score_type] * weight

        #if val > val_best:
        if True:
            test_res = self.test_evaler(self.model, 'test_' + str(iteration),
                                        2)
            self.logger.info('######## Iter (TEST) ' + str(iteration) +
                             ' ########')
            self.logger.info(str(test_res))
        # crucial!
        self.model.eval()
        return val


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

import os
import sys
import tensorflow as tf
import numpy as np
from time import time
import argparse

from tokmangan.TokManGAN import TokManGAN
from maskgan.MaskGAN import MaskGAN

from common.DataLoader import DataLoader
from common.config import ACT, START_TOKEN_IDX, END_TOKEN_IDX, PAD_TOKEN_IDX
from common.text_process import text_precess, text_to_code
from common.text_process import get_tokenlized, get_word_list, get_dict
import joblib


def get_batch_seq_len(xs):
    return list(map(lambda x: len([y for y in x if y != END_TOKEN_IDX and y != PAD_TOKEN_IDX]), xs))


def generate_cond_samples(helper, data_loader, output_file=None, get_code=True, pass_rate=None, temp=1):
    # Generate Samples
    data_loader.reset_pointer()
    seeds = []
    seeds_len = []
    pass_rate = pass_rate or 0

    generated_samples = []
    generated_acts = []

    real_samples = []
    real_acts = []

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        if np.random.random() > pass_rate:
            _generated_samples, _act, _seed, _seed_len, _real_acts = helper.generate(batch, temp=temp)
            generated_samples.extend(_generated_samples)
            generated_acts.append(_act)
            seeds.append(_seed)
            seeds_len.append(_seed_len)
            real_samples.append(batch)
            real_acts.append(_real_acts)

    generated_acts = np.concatenate(generated_acts)
    seeds = np.concatenate(seeds)
    real_samples = np.concatenate(real_samples)
    real_acts = np.concatenate(real_acts)

    codes = list()
    if output_file is not None:
        with open(output_file, 'w') as fout:
            for poem in generated_samples:
                buffer = ' '.join([str(x) for x in poem]) + '\n'
                fout.write(buffer)
                if get_code:
                    codes.append(poem)
        return np.array(codes), generated_acts, seeds, seeds_len, real_samples, real_acts

    codes = ""
    for poem in generated_samples:
        buffer = ' '.join([str(x) for x in poem]) + '\n'
        codes += buffer

    return codes, generated_acts, seeds, seeds_len, real_samples, real_acts


def get_real_test_file(codes, samples_fpath, iw_dict, seed=None, acts=None, targets=None):
    if type(codes) == str:
        codes = get_tokenlized(codes, is_text_file=True, add_end_token=False)

    def code_to_text(a_codes, use_token=False):
        result = ''
        numbers = map(int, a_codes)
        for number in numbers:
            if use_token == False:
                if number == END_TOKEN_IDX:
                    break
                if number == START_TOKEN_IDX or number == PAD_TOKEN_IDX:
                    continue
            result += (iw_dict[str(number)] + ' ')
        return result

    paras = ""
    for i, sent in enumerate(codes):
        paras += code_to_text(sent) + '\n'
        if seed is not None:
            paras += '\tSEED:' + code_to_text(seed[i]) + '\n'
        if acts is not None:
            paras += '\tACT:' + ','.join(list(map(str, acts[i]))) + '\n'
        if targets is not None:
            paras += '\tTARGET:' + code_to_text(targets[i]) + '\n'
    with open(samples_fpath, 'w') as outfile:
        outfile.write(paras)


def sample_dummy_token(a_token, vocab_size):
    dummy_token = np.random.choice(vocab_size, 1)[0]
    no_candidates = [START_TOKEN_IDX, END_TOKEN_IDX, PAD_TOKEN_IDX, a_token]
    while dummy_token in no_candidates:
        dummy_token = np.random.choice(vocab_size, 1)[0]
    return dummy_token


def find_origin_id(p, seed_id):
    cnt_present = 0
    for origin_id, is_missing in enumerate(p):
        is_presented = 1-is_missing
        if is_presented:
            if  cnt_present == seed_id:
                return origin_id
            cnt_present += 1
    return None


def init_sess():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    return sess


def init_training(train_file, test_file, data_loc, test_data_loc, wi_dict=None, iw_dict=None, max_seq_len=None):
    # for train data
    tokens = get_tokenlized(data_loc, add_end_token=True, max_seq_len=max_seq_len)
    sequence_length = len(max(tokens, key=len))

    if wi_dict is None:
        word_set = get_word_list(tokens)
        [wi_dict, iw_dict] = get_dict(word_set)

    vocab_size = len(wi_dict)

    with open(train_file, 'w') as outfile:
        outfile.write(text_to_code(tokens, wi_dict, sequence_length))

    # for test data
    # the data including words that not included in the training data were excluded
    # see ./save/testdata/processed_image_coco.txt
    test_tokens = get_tokenlized(test_data_loc, add_end_token=True, max_seq_len=max_seq_len)
    filtered_tokens = []
    for token in test_tokens:
        check = True
        for _tok in token:
            if not _tok in wi_dict:
                check = False
                break
        if check:
            filtered_tokens.append(token)

    with open(test_file, 'w') as outfile:
        outfile.write(text_to_code(filtered_tokens, wi_dict, sequence_length))

    return sequence_length, vocab_size, wi_dict, iw_dict


class Helper:
    def __init__(self, db='coco', save_dir=None, name='tokmangan'):
        self.db = db
        self.name = name

        self.vocab_size = 20
        self.emb_dim = 32
        self.hidden_dim = 32
        self.sequence_length = 20

        self.batch_size = 64
        self.generate_num = 128

        self.pre_epoch_num = 80
        self.pre_dis_epoch_num = 40
        self.adversarial_epoch_num = 100

        self.is_training = False
        self.gen_vd_keep_prob = 1.0
        self.dis_vd_keep_prob = 1.0

        self.min_present_rate = 0.0
        self.max_present_rate = 0.5

        save_dir = os.path.join(save_dir, "{}_{}".format(self.db, self.name))
        log_dir_MLE = os.path.join(save_dir, 'MLE')
        log_dir_GAN = os.path.join(save_dir, 'GAN')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            os.makedirs(log_dir_MLE)
            os.makedirs(log_dir_GAN)

        self.oracle_file = os.path.join(save_dir, 'oracle.txt')
        self.test_oracle_file = os.path.join(save_dir, 'test_oracle.txt')
        self.generator_file = os.path.join(save_dir, 'generator.txt')
        self.test_file = os.path.join(save_dir, 'test_file_{}_{}.txt')

        self.save_dir = save_dir
        self.log_dir_MLE = log_dir_MLE
        self.log_dir_GAN = log_dir_GAN

    def init(self):
        self.sess = init_sess()
        self.max_seed_len = int(self.sequence_length * self.max_present_rate) + 1

    def build(self):
        self.generator = TokManGAN(self.vocab_size, self.batch_size, self.emb_dim, self.hidden_dim, self.sequence_length, self.max_seed_len,
                 is_training=self.is_training, gen_vd_keep_prob=self.gen_vd_keep_prob, dis_vd_keep_prob=self.dis_vd_keep_prob)

        self.t_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=self.t_vars, max_to_keep=50)
        self.sess.run(tf.global_variables_initializer())

        self._build_data_loader()

    def _build_data_loader(self):
        self.gen_data_loader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        self.oracle_data_loader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        self.dis_data_loader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        self.gen_data_loader.create_batches(self.oracle_file)
        self.dis_data_loader.create_batches(self.oracle_file)
        self.oracle_data_loader.create_batches(self.test_oracle_file)

    def save(self, logdir, step):
        model_name = 'model.ckpt'
        checkpoint_path = os.path.join(logdir, model_name)
        sys.stdout.flush()
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        self.saver.save(self.sess, checkpoint_path, global_step=step)
        print("{} model has stored.".format(step))

    def load(self, logdir):
        if logdir == self.log_dir_MLE or self.is_training == False:
            self.saver = tf.train.Saver(var_list=self.generator.g_params, max_to_keep=18)

        ckpt = tf.train.get_checkpoint_state(logdir)
        if ckpt:
            print("\tCheckpoint found: {}".format(ckpt.model_checkpoint_path))
            global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            sys.stdout.write('\t')
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            self.saver = tf.train.Saver(var_list=self.t_vars, max_to_keep=18)
            return global_step
        else:
            print('No checkpoint found')
            return None

    def _get_samples_fpath(self, train_method, epoch):
        return self.test_file.format(train_method, epoch)

    def get_samples_fpath(self, mode, epoch, temp=1):
        min_pr = self.min_present_rate
        max_pr = self.max_present_rate
        fpath = '{}_{}_epoch[{}]_pr[{}_{}]_temp[{}].txt'.format(self.db, mode, str(epoch), min_pr, max_pr, temp)
        return os.path.join(self.save_dir, fpath)

    def picking_seed(self, batch, batch_len, selection_rate=0.5, mask_strategy='mix'):
        seeds = []
        seeds_len = []
        present_rates = np.random.uniform(low=self.min_present_rate, high=self.max_present_rate, size=len(batch))
        target_acts = np.zeros(batch.shape, dtype=np.int32)
        for i, (x, n_x) in enumerate(list(zip(batch, batch_len))):
            _rate = present_rates[i]
            if mask_strategy == 'cont':
                mask_cond = False
            elif mask_strategy == 'random':
                mask_cond = True
            elif mask_strategy == 'mix':
                mask_cond = np.random.random() > selection_rate

            if mask_cond:
                p = np.random.choice([True, False], size=[n_x], p=[_rate, 1 - _rate])
                while np.sum(p) >= self.max_seed_len:
                    p = np.random.choice([True, False], size=[n_x], p=[_rate, 1 - _rate])

                p = np.concatenate([p, [False] * (len(x) - n_x)])
            else:
                masked_length = int(n_x * _rate) - 1
                if masked_length < 0:
                    masked_length = 0
                start_mask = np.random.randint(1, n_x - masked_length + 1, size=1)[0]

                p = np.zeros_like(x, dtype=np.bool)
                p[start_mask:start_mask + masked_length] = True

            seed = x[p]
            n_seed = len(seed)

            missing = (1 - p.astype(np.int32))

            # Learning the 'replace' manipulation in pre-training stage
            # This strategy may can help the manipulator chooses a rich action.
            # We did not use this strategy for current version of the paper.
            """
            if np.random.random() > selection_rate and n_seed > 0:
                _rate = np.random.uniform(low=self.is_present_rate, high=self.max_present_rate, size=1)[0]
                len_replace = int(n_seed * _rate) - 1
                if len_replace > 0:
                    ids = np.random.choice(n_seed, len_replace, replace=False)
                    new_p = missing.copy()
                    for _id in ids:
                        seed[_id] = sample_dummy_token(seed[_id], self.vocab_size)
                        new_p[find_origin_id(missing, _id)] = ACT['replace']
                    missing = new_p
            """

            target_acts[i] = missing

            seeds_len.append(n_seed)
            seed = np.pad(seed, [[0, self.max_seed_len - n_seed]], mode='constant', constant_values=PAD_TOKEN_IDX)
            seeds.append(seed)
        return seeds, seeds_len, target_acts

    def pretrain(self):
        last_epoch = self.load(self.log_dir_MLE)
        if last_epoch is None:
            last_epoch = 0
        for epoch in range(last_epoch + 1, self.pre_epoch_num + 1):
            start = time()
            losses = []
            act_losses = []
            self.gen_data_loader.reset_pointer()
            for it in range(self.gen_data_loader.num_batch):
                batch = self.gen_data_loader.next_batch()
                batch_len = get_batch_seq_len(batch)
                seed, seed_len, target_acts = self.picking_seed(batch, batch_len)
                loss, act_loss, _ = self.sess.run(
                    [self.generator.pretrain_loss, self.generator.pretrain_act_loss, self.generator.pretrain_updates],
                    feed_dict={
                        self.generator.x: batch,
                        self.generator.x_len: batch_len,
                        self.generator.seed: seed,
                        self.generator.acts: target_acts,
                        self.generator.seed_len: seed_len,
                        self.generator.learning_rate: 1e-3
                    })
                losses.append(loss)
                act_losses.append(act_loss)

            end = time()

            print('epoch:' + str(epoch) + '\t time:' + str(int(end - start)) + '\t loss:' + str(
                np.mean(losses)) + '\t act_loss:' + str(np.mean(act_losses)))

            if epoch % 10 == 0:
                samples, acts, seed, seed_len, targets, target_acts = \
                    generate_cond_samples(self, self.oracle_data_loader, output_file=self.generator_file, pass_rate=0.8)
                get_real_test_file(self.generator_file, self._get_samples_fpath('MLE', epoch), self.iw_dict,
                                   seed=seed, acts=acts, targets=targets)
                self.save(self.log_dir_MLE, epoch)

    def pretrain_dis(self):
        for epoch in range(1, self.pre_dis_epoch_num + 1):
            start = time()
            dis_losses = []
            self.gen_data_loader.reset_pointer()
            for index in range(self.gen_data_loader.num_batch):
                batch = self.gen_data_loader.next_batch()
                batch_len = get_batch_seq_len(batch)
                seed, seed_len, target_acts = self.picking_seed(batch, batch_len)
                dis_feed = {
                    self.generator.x: batch,
                    self.generator.seed: seed,
                    self.generator.seed_len: seed_len,
                    self.generator.acts: target_acts,
                    self.generator.learning_rate: 1e-4
                }

                dis_loss_fake, dis_loss_real, _ = self.sess.run(
                    [self.generator.dis_loss_fake, self.generator.dis_loss_real, self.generator.dis_updates],
                    feed_dict=dis_feed)
                dis_losses.append([dis_loss_fake, dis_loss_real])
            end = time()
            print('epoch:' + str(epoch) + '\t elapsed:' + str(int(end - start)) + '\t pre dis loss:' + str(
                np.mean(dis_losses, axis=0)))

    def train_gan(self):
        last_epoch = self.load(self.log_dir_GAN)
        if last_epoch is None:
            last_epoch = 0

        for epoch in range(last_epoch + 1, self.adversarial_epoch_num + 1):
            start = time()
            losses, dis_losses = [], []

            self.gen_data_loader.reset_pointer()
            self.dis_data_loader.reset_pointer()
            for _ in range(self.gen_data_loader.num_batch):
                for _ in range(2):
                    batch = self.dis_data_loader.next_batch()
                    batch_len = get_batch_seq_len(batch)
                    seed, seed_len, target_acts = self.picking_seed(batch, batch_len)
                    dis_feed = {
                        self.generator.x: batch,
                        self.generator.seed: seed,
                        self.generator.seed_len: seed_len,
                        self.generator.acts: target_acts,
                        self.generator.learning_rate: 1e-4
                    }

                    dis_loss_fake, dis_loss_real, critic_loss, _ = self.sess.run(
                        [self.generator.dis_loss_fake, self.generator.dis_loss_real, self.generator.critic_loss,
                         self.generator.dis_updates],
                        feed_dict=dis_feed)
                    dis_losses.append([dis_loss_fake, dis_loss_real, critic_loss])

                batch = self.gen_data_loader.next_batch()
                batch_len = get_batch_seq_len(batch)
                seed, seed_len, _ = self.picking_seed(batch, batch_len)

                g_feed = {
                    self.generator.seed: seed,
                    self.generator.seed_len: seed_len,
                    self.generator.learning_rate: 1e-4
                }

                loss, _ = self.sess.run([self.generator.g_loss, self.generator.g_updates], feed_dict=g_feed)
                losses.append(loss)

            end = time()

            log_tmpl = "epoch:{0:} \t elapsed:{1:.0f}s \t gan loss:{2:.3f} \t dis loss fake:{3:.3f} \t dis loss real:{4:.3f} \t critic loss:{5:.3f}"
            _gan_loss = np.mean(losses)
            _dis_losses = np.mean(dis_losses, axis=0)
            print(log_tmpl.format(epoch, end - start, _gan_loss, _dis_losses[0], _dis_losses[1], _dis_losses[2]))

            if epoch % 10 == 0:
                samples, acts, seed, _, targets, _ = generate_cond_samples(self, self.oracle_data_loader,
                                                                     output_file=self.generator_file, pass_rate=0.8)
                get_real_test_file(self.generator_file, self._get_samples_fpath('GAN', epoch), self.iw_dict, seed=seed,
                                   acts=acts, targets=targets)
                self.save(self.log_dir_GAN, epoch)

    def train(self, pretrain_gen=True, pretrain_dis=True, train_gan=True):
        self.init()
        self.build()

        if pretrain_gen:
            print('start pre-train generator:')
            self.pretrain()
        else:
            self.load(self.log_dir_MLE)

        if pretrain_dis:
            print('start pre-train discriminator:')
            self.pretrain_dis()

        if train_gan:
            print('start adversarial training:')
            self.train_gan()

    def generate(self, batch, seed=None, seed_len=None, target_acts=None, temp=1):
        if seed is None:
            batch_len = get_batch_seq_len(batch)
            seed, seed_len, target_acts = self.picking_seed(batch, batch_len)

        gen_result, gen_act = self.sess.run([self.generator.gen_x, self.generator.gen_act], feed_dict={
            self.generator.seed: seed,
            self.generator.temp: temp
        })

        return gen_result.tolist(), gen_act, seed, seed_len, target_acts


class Helper4MaskGAN(Helper):
    def build(self):
        self.generator = MaskGAN(self.vocab_size, self.batch_size, self.emb_dim, self.hidden_dim, self.sequence_length,
                 is_training=self.is_training, gen_vd_keep_prob=self.gen_vd_keep_prob, dis_vd_keep_prob=self.dis_vd_keep_prob)

        self.t_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=self.t_vars, max_to_keep=50)
        self.sess.run(tf.global_variables_initializer())

        self._build_data_loader()


    def picking_seed(self, batch, batch_len, selection_rate=0.5, mask_strategy='mix'):
        seeds = []
        seeds_len = []
        present_rates = np.random.uniform(low=self.min_present_rate, high=self.max_present_rate, size=len(batch))
        target_acts = np.zeros(batch.shape, dtype=np.int32)
        for i, (x, n_x) in enumerate(list(zip(batch, batch_len))):
            _rate = present_rates[i]
            if mask_strategy == 'cont':
                mask_cond = False
            elif mask_strategy == 'random':
                mask_cond = True
            elif mask_strategy == 'mix':
                mask_cond = np.random.random() > selection_rate

            if mask_cond:
                p = np.random.choice([True, False], size=[n_x], p=[_rate, 1 - _rate])
                while np.sum(p) >= self.max_seed_len:
                    p = np.random.choice([True, False], size=[n_x], p=[_rate, 1 - _rate])

                p = np.concatenate([p, [False] * (len(x) - n_x)])
            else:
                masked_length = int(n_x * _rate) - 1
                if masked_length < 0:
                    masked_length = 0
                start_mask = np.random.randint(1, n_x - masked_length + 1, size=1)[0]

                p = np.zeros_like(x, dtype=np.bool)
                p[start_mask:start_mask + masked_length] = True

            seed = x[p]
            n_seed = len(seed)

            missing = (1 - p.astype(np.int32))
            target_acts[i] = missing

            seeds_len.append(n_seed)
            seed = np.pad(seed, [[0, self.max_seed_len - n_seed]], mode='constant', constant_values=PAD_TOKEN_IDX)
            seeds.append(seed)
        return seeds, seeds_len, target_acts

    def pretrain(self):
        last_epoch = self.load(self.log_dir_MLE)
        if last_epoch is None:
            last_epoch = 0
        for epoch in range(last_epoch + 1, self.pre_epoch_num + 1):
            start = time()
            losses = []
            self.gen_data_loader.reset_pointer()
            for it in range(self.gen_data_loader.num_batch):
                batch = self.gen_data_loader.next_batch()
                batch_len = get_batch_seq_len(batch)
                _, _, missing = self.picking_seed(batch, batch_len)
                loss, _ = self.sess.run(
                    [self.generator.pretrain_loss, self.generator.pretrain_updates],
                    feed_dict={
                        self.generator.x: batch,
                        self.generator.x_len: batch_len,
                        self.generator.missing: missing,
                        self.generator.learning_rate: 1e-3
                    })
                losses.append(loss)

            end = time()

            print('epoch:' + str(epoch) + '\t time:' + str(int(end - start)) + '\t loss:' + str(
                np.mean(losses)) )

            if epoch % 10 == 0:
                samples, _, seed, _, targets, missing = \
                    generate_cond_samples(self, self.oracle_data_loader, output_file=self.generator_file, pass_rate=0.8)
                get_real_test_file(self.generator_file, self._get_samples_fpath('MLE', epoch), self.iw_dict,
                                   seed=seed, acts=missing, targets=targets)
                self.save(self.log_dir_MLE, epoch)

    def pretrain_dis(self):
        for epoch in range(1, self.pre_dis_epoch_num + 1):
            start = time()
            dis_losses = []
            self.gen_data_loader.reset_pointer()
            for index in range(self.gen_data_loader.num_batch):
                batch = self.gen_data_loader.next_batch()
                batch_len = get_batch_seq_len(batch)
                _, _, missing = self.picking_seed(batch, batch_len)
                dis_feed = {
                    self.generator.x: batch,
                    self.generator.missing: missing,
                    self.generator.learning_rate: 1e-4
                }

                dis_loss_fake, dis_loss_real, _ = self.sess.run(
                    [self.generator.dis_loss_fake, self.generator.dis_loss_real, self.generator.dis_updates],
                    feed_dict=dis_feed)
                dis_losses.append([dis_loss_fake, dis_loss_real])
            end = time()
            print('epoch:' + str(epoch) + '\t elapsed:' + str(int(end - start)) + '\t pre dis loss:' + str(
                np.mean(dis_losses, axis=0)))

    def train_gan(self):
        last_epoch = self.load(self.log_dir_GAN)
        if last_epoch is None:
            last_epoch = 0

        for epoch in range(last_epoch + 1, self.adversarial_epoch_num + 1):
            start = time()
            losses, dis_losses = [], []

            self.gen_data_loader.reset_pointer()
            self.dis_data_loader.reset_pointer()
            for _ in range(self.gen_data_loader.num_batch):
                for _ in range(2):
                    batch = self.dis_data_loader.next_batch()
                    batch_len = get_batch_seq_len(batch)
                    _, _, missing = self.picking_seed(batch, batch_len)
                    dis_feed = {
                        self.generator.x: batch,
                        self.generator.missing: missing,
                        self.generator.learning_rate: 1e-4
                    }

                    dis_loss_fake, dis_loss_real, critic_loss, _ = self.sess.run(
                        [self.generator.dis_loss_fake, self.generator.dis_loss_real, self.generator.critic_loss,
                         self.generator.dis_updates],
                        feed_dict=dis_feed)
                    dis_losses.append([dis_loss_fake, dis_loss_real, critic_loss])

                batch = self.gen_data_loader.next_batch()
                batch_len = get_batch_seq_len(batch)
                _, _, missing = self.picking_seed(batch, batch_len)

                g_feed = {
                    self.generator.x: batch,
                    self.generator.missing: missing,
                    self.generator.learning_rate: 1e-4
                }

                loss, _ = self.sess.run([self.generator.g_loss, self.generator.g_updates], feed_dict=g_feed)
                losses.append(loss)

            end = time()

            log_tmpl = "epoch:{0:} \t elapsed:{1:.0f}s \t gan loss:{2:.3f} \t dis loss fake:{3:.3f} \t dis loss real:{4:.3f} \t critic loss:{5:.3f}"
            _gan_loss = np.mean(losses)
            _dis_losses = np.mean(dis_losses, axis=0)
            print(log_tmpl.format(epoch, end - start, _gan_loss, _dis_losses[0], _dis_losses[1], _dis_losses[2]))

            if epoch % 10 == 0:
                samples, _, seed, _, targets, missing = generate_cond_samples(self, self.oracle_data_loader,
                                                                     output_file=self.generator_file, pass_rate=0.8)
                get_real_test_file(self.generator_file, self._get_samples_fpath('GAN', epoch), self.iw_dict, seed=seed,
                                   acts=missing, targets=targets)
                self.save(self.log_dir_GAN, epoch)

    def generate(self, batch, seed=None, seed_len=None, target_acts=None, temp=1):
        if target_acts is None:
            batch_len = get_batch_seq_len(batch)
            seed, seed_len, target_acts = self.picking_seed(batch, batch_len)

        gen_result = self.sess.run(self.generator.gen_x, feed_dict={
            self.generator.x: batch,
            self.generator.missing: target_acts,
            self.generator.temp: temp
        })

        return gen_result.tolist(), np.zeros_like(target_acts), seed, seed_len, target_acts


def get_helper(db, model_name):
    helper = None
    if model_name.startswith('tokmangan'):
        helper = Helper(db=db, save_dir='./save', name=model_name)
    elif model_name.startswith('maskgan'):
        helper = Helper4MaskGAN(db=db, save_dir='./save', name=model_name)

    assert helper is not None
    assert db in ['coco', 'emnlp']

    if db == 'coco':
        data_loc = 'data/image_coco.txt'
        test_data_loc = 'data/testdata/image_coco.txt'
        if os.path.exists('coco_dict.pkl'):
            dicts = joblib.load('coco_dict.pkl')
            sequence_length, vocab_size, wi_dict, iw_dict = \
                init_training(helper.oracle_file, helper.test_oracle_file, data_loc, test_data_loc,
                              wi_dict=dicts['wi_dict'], iw_dict=dicts['iw_dict'], max_seq_len=20)
        else:
            sequence_length, vocab_size, wi_dict, iw_dict =  \
                init_training(helper.oracle_file, helper.test_oracle_file, data_loc, test_data_loc, max_seq_len=20)
            joblib.dump({
               'wi_dict': wi_dict,
                'iw_dict' : iw_dict
            }, 'coco_dict.pkl')

    else:
        data_loc = 'data/emnlp_news.txt'
        test_data_loc = 'data/testdata/emnlp_news.txt'

        if os.path.exists('emnlp_dict.pkl'):
            dicts = joblib.load('emnlp_dict.pkl')
            sequence_length, vocab_size, wi_dict, iw_dict = \
                init_training(helper.oracle_file, helper.test_oracle_file, data_loc, test_data_loc,
                              wi_dict=dicts['wi_dict'], iw_dict=dicts['iw_dict'])
        else:
            sequence_length, vocab_size, wi_dict, iw_dict = \
                init_training(helper.oracle_file, helper.test_oracle_file, data_loc, test_data_loc)
            joblib.dump({
                'wi_dict': wi_dict,
                'iw_dict': iw_dict
            }, 'emnlp_dict.pkl')

    helper.sequence_length = sequence_length
    helper.vocab_size = vocab_size
    helper.wi_dict = wi_dict
    helper.iw_dict = iw_dict

    helper.init()
    return helper


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-g', '--gan_model', default='tokmangan', choices=['tokmangan', 'maskgan'])
    ap.add_argument('-t', '--mode', default='GAN', choices=['GAN', 'MLE'])
    ap.add_argument('-d', '--dataset', default='coco', choices=['coco', 'emnlp'])
    ap.add_argument('-s', '--unit_size', default=32, type=int)
    args = ap.parse_args()

    helper = get_helper(args.dataset, args.gan_model)
    helper.is_training = True

    if args.mode == 'MLE':
        helper.gen_vd_keep_prob = 0.9
    else:
        helper.gen_vd_keep_prob = 0.7
        helper.dis_vd_keep_prob = 0.9

    helper.min_present_rate = 0.0
    helper.max_present_rate = 0.75
    helper.emb_dim = args.unit_size
    helper.hidden_dim = args.unit_size

    helper.pre_epoch_num = 80
    helper.pre_dis_epoch_num = 5
    helper.adversarial_epoch_num = 200

    print(helper.sequence_length, helper.vocab_size)

    if args.mode == 'MLE':
        helper.train(pretrain_gen=True, pretrain_dis=False, train_gan=False)
    else:
        helper.adversarial_epoch_num = 200
        helper.train(pretrain_gen=False, pretrain_dis=False, train_gan=True)
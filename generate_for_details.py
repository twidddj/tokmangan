import os
import argparse
from train import get_helper
from generate_for_eval import seed_extraction
from common.config import START_TOKEN_IDX, END_TOKEN_IDX, PAD_TOKEN_IDX


def code_to_text(a_codes, use_token=False):
    result = ''
    numbers = map(int, a_codes)
    for number in numbers:
        if use_token == False:
            if number == END_TOKEN_IDX:
                break
            if number == START_TOKEN_IDX or number == PAD_TOKEN_IDX:
                continue
        result += (helper.iw_dict[str(number)] + ' ')
    return result


def details_paras(batches, seeds, missings, test_gen_samples, test_gen_acts, n_generate_per_seed):
    paras = ""
    for i in range(len(batches)):
        _batch = batches[i]
        _seed_batch = seeds[i]
        _missing_batch = missings[i]
        for j in range(len(_batch)):
            _target = _batch[j]
            _seed = _seed_batch[j]
            _missing = _missing_batch[j]

            paras += "TARGET: " + code_to_text(_target) + '\n'
            paras += "SEED: " + code_to_text(_seed) + '\n'
            paras += "MISSING: " + ','.join(list(map(str, _missing))) + '\n'

            for k in range(n_generate_per_seed):
                _gen_x = test_gen_samples[i][k][j]
                _gen_act = test_gen_acts[i][k][j]

                paras += "\t gen {}: ".format(k) + code_to_text(_gen_x) + '\n'
                paras += "\t act {}: ".format(k) + ','.join(list(map(str, _gen_act))) + '\n'
    return paras


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-g', '--gan_model', default='tokmangan', choices=['tokmangan', 'maskgan'])
    ap.add_argument('-t', '--mode', default='GAN', choices=['GAN', 'MLE'])
    ap.add_argument('-d', '--dataset', default='coco', choices=['coco', 'emnlp'])
    ap.add_argument('-s', '--unit_size', default=32, type=int)
    ap.add_argument('-n', '--n_generate_per_seed', default=10, type=int)
    ap.add_argument('-r', '--gen_vd_keep_prob', default=.8, type=float)
    args = ap.parse_args()

    db = args.dataset
    h_dim = args.unit_size
    model_name = args.gan_model
    mode = args.mode
    n_generate_per_seed = args.n_generate_per_seed

    seed_info_dict = seed_extraction(db, model_name)
    helper = get_helper(db, model_name)

    helper.gen_vd_keep_prob = args.gen_vd_keep_prob

    helper.is_training = False
    helper.min_present_rate = 0.0
    helper.max_present_rate = 0.75

    helper.emb_dim = h_dim
    helper.hidden_dim = h_dim

    helper.init()
    helper.build()

    if mode == 'GAN':
        epoch = helper.load(helper.log_dir_GAN)
    else:
        epoch = helper.load(helper.log_dir_MLE)

    present_rates = seed_info_dict.keys()

    for key in present_rates:
        present_rate = float(key)

        batches = seed_info_dict[key]['batches']
        seeds = seed_info_dict[key]['seeds']
        seeds_len = seed_info_dict[key]['seeds_len']
        missings = seed_info_dict[key]['missings']

        test_gen_samples = []
        test_gen_acts = []

        helper.min_present_rate = present_rate
        helper.max_present_rate = present_rate

        for i in range(len(seeds)):
            target = batches[i]
            seed = seeds[i]
            seed_len = seeds_len[i]
            missing = missings[i]

            samples_per_seed = []
            acts_per_seed = []

            for k in range(n_generate_per_seed):
                gen_samples, gen_acts, _, _, _ = helper.generate(target, seed, seed_len, missing)
                samples_per_seed.append(gen_samples)
                acts_per_seed.append(gen_acts)
            test_gen_samples.append(samples_per_seed)
            test_gen_acts.append(acts_per_seed)

        paras = details_paras(batches, seeds, missings, test_gen_samples, test_gen_acts, n_generate_per_seed)
        _key = key.replace('.', '')
        _vd = str(args.gen_vd_keep_prob).replace('.', '')
        fpath = os.path.join(helper.save_dir, 'details_{}_{}_{}_[pr-{}_vd-{}].txt'.format(db, mode, epoch, _key, _vd))
        with open(fpath, 'w') as outfile:
            outfile.write(paras)
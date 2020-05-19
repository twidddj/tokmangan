import os
import argparse
from train import get_helper, get_batch_seq_len, get_real_test_file, generate_cond_samples
import joblib
import numpy as np


def seed_extraction(db, model_name, present_rates=[0.25, 0.5, 0.75]):
    helper = get_helper(db, model_name)

    helper.min_present_rate = 0.0
    helper.max_present_rate = 0.75

    helper.init()
    helper._build_data_loader()

    seed_info_fpath = '{}_test_seed_info.pkl'.format(db)

    if os.path.exists(seed_info_fpath):
        seed_info_dict = joblib.load(seed_info_fpath)
    else:
        seed_info_dict = {}
        for seed_rate in present_rates:
            key = str(seed_rate)

            batches, seeds, seeds_len, missings = [], [], [], []

            helper.oracle_data_loader.reset_pointer()
            for _ in range(helper.oracle_data_loader.num_batch):
                batch = helper.oracle_data_loader.next_batch()
                batch_len = get_batch_seq_len(batch)
                helper.min_present_rate = seed_rate
                helper.max_present_rate = seed_rate

                seed, seed_len, missing = helper.picking_seed(batch, batch_len)

                batches.append(batch)
                seeds.append(seed)
                seeds_len.append(seed_len)
                missings.append(missing)

            seed_info_dict[key] = {
                'batches': np.stack(batches),
                'seeds': np.stack(seeds),
                'seeds_len': np.stack(seeds_len),
                'missings': np.stack(missings)
            }

        joblib.dump(seed_info_dict, seed_info_fpath)

    return seed_info_dict


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-g', '--gan_model', default='tokmangan', choices=['tokmangan', 'maskgan'])
    ap.add_argument('-t', '--mode', default='GAN', choices=['GAN', 'MLE'])
    ap.add_argument('-d', '--dataset', default='coco', choices=['coco', 'emnlp'])
    ap.add_argument('-s', '--unit_size', default=32, type=int)
    args = ap.parse_args()

    db = args.dataset
    h_dim = args.unit_size
    model_name = args.gan_model
    mode = args.mode

    helper = get_helper(db, model_name)

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

    TEMPERATURES = [0.8, 0.9, 0.95, 1.0, 1.01, 1.02, 1.03, 1.04, 1.06, 1.07, 1.08, 1.09, 2.25, 2.5, 2.75, 3.25, 3.5, 3.75]

    for temp in TEMPERATURES:
        # generate unconditional samples
        helper.min_present_rate = 0.0
        helper.max_present_rate = 0.0

        samples, acts, seed, seed_len, targets, _ = \
            generate_cond_samples(helper, helper.oracle_data_loader, temp=temp, output_file=helper.generator_file, pass_rate=0.0)

        fpath = helper.get_samples_fpath(mode, epoch, temp=temp)
        get_real_test_file(helper.generator_file, fpath, helper.iw_dict)
        print(fpath)

        # generate conditional samples
        seed_info_dict = seed_extraction(db, model_name, present_rates=[0.25, 0.5, 0.75])
        present_rates = seed_info_dict.keys()
        for key in present_rates:
            present_rate = float(key)

            batches = seed_info_dict[key]['batches']
            seeds = seed_info_dict[key]['seeds']
            seeds_len = seed_info_dict[key]['seeds_len']
            missings = seed_info_dict[key]['missings']

            samples = []

            helper.min_present_rate = present_rate
            helper.max_present_rate = present_rate

            for i in range(len(seeds)):
                target = batches[i]
                seed = seeds[i]
                seed_len = seeds_len[i]
                missing = missings[i]

                gen_samples, gen_acts, _, _, _ = helper.generate(target, seed, seed_len, missing, temp=temp)
                samples.extend(gen_samples)

            fpath = helper.get_samples_fpath(mode, epoch, temp=temp)
            get_real_test_file(samples, fpath, helper.iw_dict)
            print(fpath)


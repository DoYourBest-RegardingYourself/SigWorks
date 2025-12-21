import sys

sys.path.append(".")
import torch
from htcsignet.feature_learning.data import extract_features
import htcsignet.feature_learning.models as models
import argparse
from htcsignet.datasets.util import load_dataset, get_subset
import wd.training as training
import numpy as np
import pickle


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


def main(args):
    exp_users = range(*args.exp_users)
    dev_users = range(*args.dev_users)

    # assert len(
    #     set(exp_users).intersection(set(dev_users))) == 0, 'Exploitation set and Development set must not overlap'

    state_dict, class_weights, forg_weights = torch.load(args.model_path,
                                                         map_location=lambda storage, loc: storage)
    device = get_device()
    # print(state_dict)
    print('Using device: {}'.format(device))

    base_model = models.available_models[args.model]()

    #base_model = torch.nn.DataParallel(base_model, device_ids=[0, 1]).cuda().eval()
    base_model = models.available_models[args.model]().to(device).eval()
    #base_model.load_state_dict(state_dict)

    # def process_fn(batch):
    #     input = batch[0]
    #     return base_model(input)

    x, y, yforg, user_mapping, filenames = load_dataset(args.data_path)

    # features = extract_features(x, process_fn, args.batch_size, args.input_size)

    # data = (features, y, yforg)
    data = (x, y, yforg)

    exp_set = get_subset(data, exp_users)
    # exp_features = extract_features(exp_set[0], process_fn, args.batch_size, args.input_size)
    exp_features = extract_features(exp_set[0], base_model, args.batch_size, device, args.input_size)
    exp_set = tuple((exp_features, exp_set[1], exp_set[2]))

    dev_set = get_subset(data, dev_users)
    # dev_features = extract_features(dev_set[0], base_model, args.batch_size, device, args.input_size)
    rng = np.random.RandomState(1234)

    eer_u_list = []
    eer_list = []
    all_results = []
    ACC = []
    FRR = []
    FAR_random = []
    FAR_skilled = []
    for _ in range(args.folds):
        classifiers, results = training.train_test_all_users(exp_set,
                                                             dev_set,
                                                             svm_type=args.svm_type,
                                                             C=args.svm_c,
                                                             gamma=args.svm_gamma,
                                                             num_gen_train=args.gen_for_train,
                                                             num_forg_from_exp=args.forg_from_exp,
                                                             num_forg_from_dev=args.forg_from_dev,
                                                             num_gen_test=args.gen_for_test,
                                                             rng=rng)
        this_eer_u, this_eer = results['all_metrics']['EER_userthresholds'], results['all_metrics']['EER']
        acc, frr, far_random, far_skilled = results['all_metrics']['mean_AUC'], results['all_metrics']['FRR'], \
            results['all_metrics']['FAR_random'], results['all_metrics']['FAR_skilled']
        all_results.append(results)
        eer_u_list.append(this_eer_u)
        eer_list.append(this_eer)
        ACC.append(acc)
        FRR.append(frr)
        FAR_random.append(far_random)
        FAR_skilled.append(far_skilled)
    print('EER (global threshold): {:.2f} (+- {:.2f})'.format(np.mean(eer_list) * 100, np.std(eer_list) * 100))
    print('EER (user thresholds): {:.2f} (+- {:.2f})'.format(np.mean(eer_u_list) * 100, np.std(eer_u_list) * 100))
    print('mean_AUC:', np.mean(ACC))
    print('FRR:', np.mean(FRR))
    print('FAR_random:', np.mean(FAR_random))
    print('FAR_skilled:', np.mean(FAR_skilled))

    if args.save_path is not None:
        print('Saving results to {}'.format(args.save_path))
        with open(args.save_path, 'wb') as f:
            pickle.dump(all_results, f)
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', choices=models.available_models, default="htcsignet",
                        help='Model architecture', dest='model')
    parser.add_argument('--model-path',
                        default="./model_checkpoint/model_last.pth")
    parser.add_argument('--data-path', default="./datasets/GPDS_1000_256X256.npz")
    parser.add_argument('--save-path')
    parser.add_argument('--input-size', nargs=2, default=(224, 224))

    parser.add_argument('--exp-users', type=int, nargs=2, default=(0, 300))
    parser.add_argument('--dev-users', type=int, nargs=2, default=(0, 300))

    parser.add_argument('--gen-for-train', type=int, default=12)
    parser.add_argument('--gen-for-test', type=int, default=10)
    parser.add_argument('--forg-from_exp', type=int, default=12)
    parser.add_argument('--forg-from_dev', type=int, default=0)

    parser.add_argument('--svm-type', choices=['rbf', 'linear'], default='rbf')
    parser.add_argument('--svm-c', type=float, default=1)
    parser.add_argument('--svm-gamma', type=float, default=2 ** -11)

    parser.add_argument('--gpu-idx', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--folds', type=int, default=10)

    arguments = parser.parse_args()
    print(arguments)

    main(arguments)

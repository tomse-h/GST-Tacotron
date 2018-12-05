from cutoff import cutoff
import os
from os.path import join
import argparse
from Hyperparameters import Hyperparameters as hp


def preprocess(base_dir):
    '''
    base_dir --- data directory
    '''
    data_dir = join(base_dir, 'data')
    train_dir = join(base_dir, 'train')
    test_dir = join(base_dir, 'test')
    dev_dir = join(base_dir, 'dev')

    for d in [data_dir, train_dir, test_dir, dev_dir]:
        for file in os.listdir(d):
            path = join(d, file)
            base, ext = os.path.splitext(path)
            if ext == '.wav' and base[-7:] != '_cutoff':
                output_path = base + '_cutoff' + ext
                cutoff(path, output_path)
                print(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir=", default="")
    args = parser.parse_args
    preprocess(hp.data)

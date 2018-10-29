import argparse
import os
import random

from torch import optim
from tqdm import trange

from model.model import BiLSTM_CRF as Model
from utils.data import Data
from utils.preprocess import preprocess
from utils.score import score
from utils.utils import *

seed_num = 123456
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)


def train(data):
    print('Training model...')
    save_data_setting(data)
    model = Model(data).to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=data.lr, momentum=data.momentum)
    for epoch in range(data.epoch):
        print('Epoch: %s/%s' % (epoch, data.epoch))
        optimizer = lr_decay(optimizer, epoch, data.lr_decay, data.lr)
        total_loss = 0
        random.shuffle(data.ids)
        model.train()
        model.zero_grad()
        train_num = len(data.ids)
        total_batch = train_num // data.batch_size + 1
        for batch in trange(total_batch):
            start, end = slice_set(batch, data.batch_size, train_num)
            instance = data.ids[start:end]
            if not instance: continue
            *model_input, _ = load_batch(instance)
            loss = model.neg_log_likelihood_loss(*model_input)
            total_loss += loss.data.item()
            loss.backward()
            optimizer.step()
            model.zero_grad()
        print('Epoch %d loss = %.3f' % (epoch, total_loss))
    torch.save(model.state_dict(), data.model_path)


def test(data):
    print('Testing model...')
    model = Model(data).to(device)
    model.load_state_dict(torch.load(data.model_path))
    instances = data.ids
    pred_results = []
    model.eval()
    test_num = len(instances)
    total_batch = test_num // data.batch_size + 1
    for batch in trange(total_batch):
        start, end = slice_set(batch, data.batch_size, test_num)
        instance = instances[start:end]
        if not instance: continue
        _, mask, *model_input, char_recover = load_batch(instance, True)
        tag_seq = model(mask, *model_input)
        pred_label = seq2label(tag_seq, mask, data.label_alphabet, char_recover)
        pred_results += pred_label
    return pred_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Setting mode and dataset.')
    parser.add_argument('--mode', choices=['train', 'test'], help='update algorithm', default='train')
    parser.add_argument('--dataset', choices=['baidu', 'dianping', 'mafengwo'], help='select dataset', default='baidu')
    args = parser.parse_args()
    mode = args.mode.lower()
    dataset = args.dataset.lower()
    print('Using dataset', dataset)
    train_file = './dataset/' + dataset + '/train_seg.txt'
    test_file = './dataset/' + dataset + '/test_seg.txt'
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        preprocess(dataset)
    data = Data()
    data.set_dataset(dataset)
    if mode == 'train':
        data.data_loader(train_file, 'train')
        train(data)
    elif mode == 'test':
        data = pickle.load(open(data.config_path, 'rb'))
        data.data_loader(test_file, 'test')
        results = test(data)
        save_results(data, results)
        score(data.result_path, test_file, data.output_path)

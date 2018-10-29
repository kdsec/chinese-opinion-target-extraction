import copy
import pickle
import warnings

import numpy as np
import torch
from torch import nn

warnings.filterwarnings('ignore')

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def init_embedding(vocab_size, embedding_dim):
    emb = nn.Embedding(vocab_size, embedding_dim).to(device)
    pretrain_emb = np.empty([vocab_size, embedding_dim])
    scale = np.sqrt(3.0 / embedding_dim)
    for index in range(vocab_size):
        pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
    emb.weight.data.copy_(torch.from_numpy(pretrain_emb))
    return emb


def seq2label(pred_tensor, mask_tensor, label_alphabet, char_recover):
    pred_tensor = pred_tensor[char_recover]
    mask_tensor = mask_tensor[char_recover]
    seq_len = pred_tensor.size(1)
    mask = mask_tensor.cpu().data.numpy()
    pred_ids = pred_tensor.cpu().data.numpy()
    batch_size = mask.shape[0]
    labels = []
    for i in range(batch_size):
        pred = [label_alphabet.get_item(pred_ids[i][j]) for j in range(seq_len) if mask[i][j] != 0]
        labels.append(pred)
    return labels


def slice_set(batch: int, batch_size: int, max: int):
    start = batch * batch_size
    end = (batch + 1) * batch_size
    if end > max:
        end = max
    return start, end


def load_batch(instances, test_mode=False):
    batch_size = len(instances)
    chars = [instance[0] for instance in instances]
    dict_feats = [instance[1] for instance in instances]
    tags = [instance[2] for instance in instances]
    labels = [instance[3] for instance in instances]
    seq_lengths = torch.tensor(list(map(len, chars)), dtype=torch.long, device=device)
    max_seq_len = seq_lengths.max()
    with torch.set_grad_enabled(test_mode):
        char_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)
        dict_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)
        tag_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)
        label_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)
        mask = torch.zeros((batch_size, max_seq_len), dtype=torch.uint8, device=device)
    for idx, (seq, pos, tag, label, seqlen) in enumerate(zip(chars, dict_feats, tags, labels, seq_lengths)):
        char_seq_tensor[idx, :seqlen] = torch.tensor(seq, dtype=torch.long)
        dict_seq_tensor[idx, :seqlen] = torch.tensor(pos, dtype=torch.long)
        tag_seq_tensor[idx, :seqlen] = torch.tensor(tag, dtype=torch.long)
        label_seq_tensor[idx, :seqlen] = torch.tensor(label, dtype=torch.long)
        mask[idx, :seqlen] = torch.ones(seqlen.item(), dtype=torch.int64)
    seq_lengths, char_perm_idx = seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    dict_seq_tensor = dict_seq_tensor[char_perm_idx]
    tag_seq_tensor = tag_seq_tensor[char_perm_idx]
    label_seq_tensor = label_seq_tensor[char_perm_idx]
    mask = mask[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    return label_seq_tensor, mask, char_seq_tensor, dict_seq_tensor, tag_seq_tensor, seq_lengths, char_seq_recover


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr / (1 + decay_rate * epoch)
    print('learning rate: {0}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def save_data_setting(data):
    _data = copy.deepcopy(data)
    _data.texts, _data.ids = [], []
    pickle.dump(_data, open(data.config_path, 'wb+'))


def save_results(data, results):
    result_file = open(data.result_path, 'w', encoding='utf-8')
    sent_num = len(results)
    content_list = data.texts
    for i in range(sent_num):
        result_file.write('{}{}\n'.format(content_list[i][0], results[i]))
    print('Results have been written into %s' % data.result_path)

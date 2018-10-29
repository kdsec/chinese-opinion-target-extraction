import json
import sys
from itertools import product

from numba import jit
from thulac import thulac
from tqdm import tqdm

segment_tool = None
dictionary = None


@jit
def kmp_search(T, P):
    mapping = [0]
    for x in P[1:]:
        check_index = mapping[-1]
        if P[check_index] == x:
            mapping += [check_index + 1]
        else:
            mapping += [0]
    result = []
    p_pointer = 0
    t_pointer = 0
    while t_pointer < len(T):
        if P[p_pointer] == T[t_pointer]:
            p_pointer += 1
            t_pointer += 1
            if p_pointer >= len(P):
                result += [t_pointer - len(P)]
                p_pointer = 0 if p_pointer == 0 else mapping[p_pointer - 1]
        else:
            t_pointer += 1 if p_pointer == 0 else 0
            p_pointer = 0 if p_pointer == 0 else mapping[p_pointer - 1]
    return result


@jit
def make_char_tag(words: list, target: str) -> list:
    rs = []
    sentence = ''.join(words)
    kmp_rs = kmp_search(sentence, target)
    i = 0
    while i < len(sentence):
        if i in kmp_rs:
            for c_i in range(len(target)):
                if c_i == 0:
                    rs.append('B')
                elif c_i == len(target) - 1:
                    rs.append('E')
                else:
                    rs.append('M')
            i += len(target)
        else:
            rs.append('O')
            i += 1
    return rs


@jit
def n_gram_in_dict(chars: list, char_index: int) -> list:
    rs = []
    for n in range(2, 6):
        if n > len(chars):
            rs += [0, 0]
            continue
        # front n-gram
        if char_index < n - 1:
            rs.append(0)
        else:
            word = ''.join(chars[char_index - n + 1: char_index + 1])
            rs.append(int(word in dictionary))
        # rear n-gram
        if char_index > len(chars) - n:
            rs.append(0)
        else:
            word = ''.join(chars[char_index:char_index + n])
            rs.append(int(word in dictionary))
    return rs


def make_dict_feat(chars):
    vector = []
    for char_index in range(len(chars)):
        vector.append(n_gram_in_dict(chars, char_index))
    return vector


# @jit
def construct_features(origin: dict) -> dict:
    features = {'content': origin['s'], 'label': origin['ot'], 'word': [], 'POS': [], 'char': [], 'char_pos': [],
                'char_pos_tag': [], 'char_word_tag': []}
    sentence = origin['s']
    # Segment
    cut_word, cut_pos = [], []
    tokens = segment_tool.cut(sentence)
    for word, pos in tokens:
        cut_word.append(word)
        cut_pos.append(pos)
    features['word'] += cut_word
    features['POS'] += cut_pos
    for word in features['word']:
        features['char'] += list(word)
    # Build char pos
    for word_index, word in enumerate(features['word']):
        for _ in word:
            features['char_pos'].append(features['POS'][word_index])
    # Build char tag(BMEO)
    features['char_tag'] = make_char_tag(features['word'], origin['ot'])
    # Build char_pos_tag
    for word_index, word in enumerate(features['word']):
        if len(word) == 1:
            features['char_pos_tag'].append('S_' + features['POS'][word_index])
        else:
            for index, char in enumerate(word):
                if index == 0:
                    features['char_pos_tag'].append('B_' + features['POS'][word_index])
                elif index == len(word) - 1:
                    features['char_pos_tag'].append('E_' + features['POS'][word_index])
                else:
                    features['char_pos_tag'].append('M_' + features['POS'][word_index])
    # Build char_word_tag(BEMS)
    for word_index, word in enumerate(features['word']):
        if len(word) == 1:
            features['char_word_tag'].append('S')
        else:
            for index, char in enumerate(word):
                if index == 0:
                    features['char_word_tag'].append('B')
                elif index == len(word) - 1:
                    features['char_word_tag'].append('E')
                else:
                    features['char_word_tag'].append('M')
    # Build dict_feat
    features['dict_feature'] = make_dict_feat(features['char'])
    return features


def handle_data(dataset: str) -> list:
    print('Processing %s ...' % dataset)
    data = [construct_features(json.loads(line)) for line in
            tqdm(open(dataset, 'r', encoding='utf-8').readlines())]
    return data


def preprocess(dataset: str):
    global segment_tool, dictionary
    print('Loading Segment Model...')
    segment_tool = thulac(rm_space=True)
    print('Loading dictionary')
    dictionary = set(map(lambda s: s.rstrip('\n'), open('dataset/dictionary.txt', encoding='utf-8').readlines()))

    dataset_list = (['train', 'test'], [dataset])
    for dataset_type, dataset_name in product(*dataset_list):
        with open('dataset/%s/%s_seg.txt' % (dataset_name, dataset_type), 'w', encoding='utf-8') as f:
            for line in handle_data('dataset/%s/%s.txt' % (dataset_name, dataset_type)):
                f.write(json.dumps(line, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    preprocess(sys.argv[1])

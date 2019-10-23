import json

from keras.preprocessing.text import Tokenizer

from config import Config

Tokenizer.size = lambda x: len(x.word_index) + 1
Tokenizer.get_index = lambda x, item: x.word_index[item] if item in x.word_index else 0
Tokenizer.get_item = lambda x, index: x.index_item[index] if index in x.index_item else None


class Data(Config):
    def __init__(self):
        Config.__init__(self)
        self.char_alphabet, self.dict_alphabet, self.label_alphabet, self.tag_alphabet = [None] * 4
        self.texts, self.ids, self.sentences = [], [], []

    def build_alphabet(self):
        self.label_alphabet = Tokenizer(char_level=True)
        self.label_alphabet.fit_on_texts('OBME')
        self.label_alphabet.index_item = dict(map(reversed, self.label_alphabet.word_index.items()))
        self.char_alphabet = Tokenizer(char_level=True)
        self.char_alphabet.fit_on_texts(map(lambda s: s['char'], self.sentences))
        self.tag_alphabet = Tokenizer(char_level=True)
        self.tag_alphabet.fit_on_texts(map(lambda s: s['char_pos_tag'], self.sentences))
        self.dict_alphabet = Tokenizer(char_level=True)
        self.dict_alphabet.fit_on_texts(map(lambda s: [str(sum([2 ** i * x for i, x in enumerate(word_dict)]))
                                                       for word_dict in s['dict_feature']], self.sentences))

    def read_instance(self):
        instence_texts = []
        instence_id = []
        for sentence in self.sentences:
            chars, labels, dict_feats, tags, char_id, label_id, dict_id, tag_id = [[] for _ in range(8)]
            for i, (char, label, dict_feat, tag) in enumerate(
                    zip(sentence['char'], sentence['char_tag'], sentence['dict_feature'], sentence['char_pos_tag'])):
                if i == self.MAX_SENTENCE_LENGTH: continue
                chars.append(char)
                char_id.append(self.char_alphabet.get_index(char))
                labels.append(label)
                label_id.append(self.label_alphabet.get_index(label.lower()))
                dict_feat = str(sum([2 ** i * x for i, x in enumerate(dict_feat)]))
                dict_feats.append(dict_feat)
                dict_id.append(self.dict_alphabet.get_index(dict_feat))
                tags.append(tag)
                tag_id.append(self.tag_alphabet.get_index(tag.lower()))
            instence_texts.append([chars, dict_feats, tags, labels])
            instence_id.append([char_id, dict_id, tag_id, label_id])
        return instence_texts, instence_id

    def data_loader(self, input_file, name):
        self.sentences = [json.loads(line) for line in open(input_file, 'r', encoding='utf-8')]
        if name == 'train':
            self.build_alphabet()
        self.texts, self.ids = self.read_instance()
        self.sentences = []

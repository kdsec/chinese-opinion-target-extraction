class Config:
    def __init__(self):
        self.epoch = 20
        self.batch_size = 128
        self.MAX_SENTENCE_LENGTH = 250
        self.char_emb_dim = 50
        self.pos_emb_dim = 20
        self.tag_emb_dim = 20
        self.pos_hidden_dim = 50
        self.lstm_hidden_dim = 50
        self.dropout = 0.5
        self.lr = 0.002
        self.lr_decay = 0.03
        self.momentum = 0.01
        self.config_path = ''
        self.model_path = ''
        self.result_path = ''
        self.output_path = ''

    def set_dataset(self, dataset):
        self.model_path = './output/%s/model' % dataset
        self.config_path = './output/%s/setting' % dataset
        self.result_path = './output/%s/result.txt' % dataset
        self.output_path = './output/%s/' % dataset

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from config import Config
from model.crf import CRF
from utils.utils import *


class BiLSTM_CRF(nn.Module):
    def __init__(self, data):
        super(BiLSTM_CRF, self).__init__()
        label_size = data.label_alphabet.size()
        data.label_size = label_size + 2
        self.lstm = BiLSTM(data).to(device)
        self.crf = CRF(target_size=label_size, use_cuda=use_cuda, average_batch=True).to(device)

    def neg_log_likelihood_loss(self, batch_label, mask, *args):
        outs = self.lstm.get_output_score(*args)
        total_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
        return total_loss

    def forward(self, mask, *args):
        outs = self.lstm.get_output_score(*args)
        scores, tag_seq = self.crf(outs, mask)
        return tag_seq


class BiLSTM(nn.Module, Config):
    def __init__(self, data):
        Config.__init__(self)
        super(BiLSTM, self).__init__()
        self.drop = nn.Dropout(self.dropout).to(device)
        self.char_embeddings = init_embedding(data.char_alphabet.size(), self.char_emb_dim)
        self.pos_embeddings = init_embedding(data.dict_alphabet.size(), self.pos_emb_dim)
        self.tag_embeddings = init_embedding(data.tag_alphabet.size(), self.tag_emb_dim)
        self.lstm = nn.LSTM(self.char_emb_dim + self.pos_emb_dim + self.pos_hidden_dim, self.lstm_hidden_dim // 2,
                            batch_first=True, bidirectional=True).to(device)
        self.hidden2tag = nn.Linear(self.lstm_hidden_dim, data.label_size).to(device)
        self.posBiLSTM = PosBiLSTM(data).to(device)

    def get_lstm_features(self, char_inputs, pos_inputs, tag_inputs, seq_lengths):
        char_embs = self.char_embeddings(char_inputs)
        char_embs = self.drop(char_embs)
        pos_embs = self.pos_embeddings(pos_inputs)
        pos_embs = self.drop(pos_embs)
        pos_lstm_out = self.posBiLSTM.get_lstm_features(tag_inputs, seq_lengths)
        emb = torch.cat([char_embs, pos_embs, pos_lstm_out], 2)
        packed_chars = pack_padded_sequence(emb, seq_lengths.cpu().numpy(), True)
        lstm_out, _ = self.lstm(packed_chars)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        lstm_out = self.drop(lstm_out.transpose(1, 0))
        return lstm_out

    def get_output_score(self, *args):
        lstm_out = self.get_lstm_features(*args)
        outputs = self.hidden2tag(lstm_out)
        return outputs

    def forward(self, mask, *args):
        batch_size = args[0].size(0)
        seq_len = args[0].size(1)
        outs = self.get_output_score(*args)
        outs = outs.view(batch_size * seq_len, -1)
        _, tag_seq = torch.max(outs, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)
        decode_seq = mask.long() * tag_seq
        return decode_seq


class PosBiLSTM(nn.Module, Config):
    def __init__(self, data):
        Config.__init__(self)
        super(PosBiLSTM, self).__init__()
        self.drop = nn.Dropout(self.dropout).to(device)
        self.pos_embeddings = init_embedding(data.dict_alphabet.size(), self.pos_emb_dim)
        self.lstm = nn.LSTM(self.pos_emb_dim, self.pos_hidden_dim // 2, batch_first=True, bidirectional=True).to(device)
        self.hidden2tag = nn.Linear(self.pos_hidden_dim, data.label_size).to(device)

    def get_lstm_features(self, pos_inputs, seq_lengths):
        pos_embs = self.pos_embeddings(pos_inputs)
        pos_embs = self.drop(pos_embs)
        packed_words = pack_padded_sequence(pos_embs, seq_lengths.cpu().numpy(), True)
        lstm_out, _ = self.lstm(packed_words)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        lstm_out = self.drop(lstm_out.transpose(1, 0))
        return lstm_out

    def get_output_score(self, *args):
        lstm_out = self.get_lstm_features(*args)
        outputs = self.hidden2tag(lstm_out)
        return outputs

    def forward(self, mask, *args):
        batch_size = args[0].size(0)
        seq_len = args[0].size(1)
        outs = self.get_output_score(*args)
        outs = outs.view(batch_size * seq_len, -1)
        _, tag_seq = torch.max(outs, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)
        decode_seq = mask.long() * tag_seq
        return decode_seq

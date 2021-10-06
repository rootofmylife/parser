# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch import from_numpy
import numpy as np
from supar.modules import (CharLSTM, ELMoEmbedding, IndependentDropout,
                           SharedDropout, TransformerEmbedding,
                           VariationalLSTM, LabelAttention, MultiHeadAttention,
                           PositionwiseFeedForward, PartitionedPositionwiseFeedForward)
from supar.utils import Config
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BatchIndices:
    """
    Batch indices container class (used to implement packed batches)
    """
    def __init__(self, batch_idxs_np):
        self.batch_idxs_np = batch_idxs_np
        self.batch_idxs_torch = from_numpy(batch_idxs_np)

        self.batch_size = int(1 + np.max(batch_idxs_np))

        batch_idxs_np_extra = np.concatenate([[-1], batch_idxs_np, [-1]])
        self.boundaries_np = np.nonzero(batch_idxs_np_extra[1:] != batch_idxs_np_extra[:-1])[0]
        self.seq_lens_np = self.boundaries_np[1:] - self.boundaries_np[:-1]
        assert len(self.seq_lens_np) == self.batch_size
        self.max_len = int(np.max(self.boundaries_np[1:] - self.boundaries_np[:-1]))

class Model(nn.Module):

    def __init__(self,
                 n_words,
                 n_tags=None,
                 n_chars=None,
                 n_lemmas=None,
                 encoder='lstm',
                 feat=['char'],
                 n_embed=100,
                 n_pretrained=100,
                 n_feat_embed=100,
                 n_char_embed=50,
                 n_char_hidden=100,
                 char_pad_index=0,
                 char_dropout=0,
                 elmo_bos_eos=(True, True),
                 elmo_dropout=0.5,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 freeze=False,
                 embed_dropout=.33,
                 n_lstm_hidden=400,
                 n_lstm_layers=3,
                 encoder_dropout=.33,
                 pad_index=0,
                 lal_d_kv=64, # Dimension of Key and Query Vectors in the LAL
                 lal_d_proj=64, # Dimension of the output vector from each label attention head
                 lal_resdrop=True, # True means the LAL uses Residual Dropout
                 lal_pwff=True, # True means the LAL has a Position-wise Feed-forward Layer
                 lal_q_as_matrix=False, # False means the LAL uses learned query vectors
                 lal_partitioned=True, # Partitioned as per the Berkeley Self-Attentive Parser
                 lal_combine_as_self=False,
                 d_model=768, # d_model must be equal with n_embed + n_feat_embed; 200 = 100 + 100
                 num_layers=1,
                 d_l=112,
                 d_k=64,
                 d_v=64,
                 d_ff=2048,
                 num_heads=8,
                 partitioned=True,
                 attention_dropout=0.2,
                 residual_dropout=0.2,
                 relu_dropout=0.2,
                 **kwargs):
        super().__init__()

        self.args = Config().update(locals())

        if encoder != 'bert' and encoder != 'lal':
            self.word_embed = nn.Embedding(num_embeddings=n_words,
                                           embedding_dim=n_embed)

            n_input = n_embed
            if n_pretrained != n_embed:
                n_input += n_pretrained
            if 'tag' in feat:
                self.tag_embed = nn.Embedding(num_embeddings=n_tags,
                                              embedding_dim=n_feat_embed)
                n_input += n_feat_embed
            if 'char' in feat:
                self.char_embed = CharLSTM(n_chars=n_chars,
                                           n_embed=n_char_embed,
                                           n_hidden=n_char_hidden,
                                           n_out=n_feat_embed,
                                           pad_index=char_pad_index,
                                           dropout=char_dropout)
                n_input += n_feat_embed
            if 'lemma' in feat:
                self.lemma_embed = nn.Embedding(num_embeddings=n_lemmas,
                                                embedding_dim=n_feat_embed)
                n_input += n_feat_embed
            if 'elmo' in feat:
                self.elmo_embed = ELMoEmbedding(n_out=n_feat_embed,
                                                bos_eos=elmo_bos_eos,
                                                dropout=elmo_dropout,
                                                requires_grad=(not freeze))
                n_input += self.elmo_embed.n_out
            if 'bert' in feat:
                self.bert_embed = TransformerEmbedding(model=bert,
                                                       n_layers=n_bert_layers,
                                                       n_out=n_feat_embed,
                                                       pooling=bert_pooling,
                                                       pad_index=bert_pad_index,
                                                       dropout=mix_dropout,
                                                       requires_grad=(not freeze))
                n_input += self.bert_embed.n_out
            self.embed_dropout = IndependentDropout(p=embed_dropout)
        if encoder == 'lstm':
            self.encoder = VariationalLSTM(input_size=n_input,
                                           hidden_size=n_lstm_hidden,
                                           num_layers=n_lstm_layers,
                                           bidirectional=True,
                                           dropout=encoder_dropout)
            self.encoder_dropout = SharedDropout(p=encoder_dropout)
            self.args.n_hidden = n_lstm_hidden * 2
        if encoder == 'lal':
            # label attention
            d_positional = (d_model // 2) if partitioned else None
            lal_d_positional = d_positional if lal_partitioned else None
            self.stacks = []

            for _ in range(num_layers):
                attn = MultiHeadAttention(num_heads, d_model, d_k, d_v, residual_dropout=residual_dropout,
                                        attention_dropout=attention_dropout, d_positional=d_positional)
                if d_positional is None:
                    ff = PositionwiseFeedForward(d_model, d_ff, relu_dropout=relu_dropout,
                                                residual_dropout=residual_dropout)
                else:
                    ff = PartitionedPositionwiseFeedForward(d_model, d_ff, d_positional, relu_dropout=relu_dropout,
                                                            residual_dropout=residual_dropout)

                self.stacks.append((attn, ff))

            attn = LabelAttention(lal_combine_as_self, d_model, lal_d_kv, lal_d_kv, d_l, lal_d_proj, use_resdrop=lal_resdrop, q_as_matrix=lal_q_as_matrix,
                                    residual_dropout=residual_dropout, attention_dropout=attention_dropout, d_positional=lal_d_positional)

            ff_dim = lal_d_proj * d_l
            if lal_combine_as_self:
                ff_dim = d_model
            if lal_pwff:
                if d_positional is None or not lal_partitioned:
                    ff = PositionwiseFeedForward(ff_dim, d_ff, relu_dropout=relu_dropout, residual_dropout=residual_dropout)
                else:
                    ff = PartitionedPositionwiseFeedForward(ff_dim, d_ff, d_positional, relu_dropout=relu_dropout, residual_dropout=residual_dropout)
            else:
                ff = None

            self.stacks.append((attn, ff))
            # end label attention

            self.encoder = TransformerEmbedding(model=bert,
                                                n_layers=n_bert_layers,
                                                pooling=bert_pooling,
                                                pad_index=pad_index,
                                                dropout=mix_dropout,
                                                requires_grad=True)
            self.encoder_dropout = nn.Dropout(p=encoder_dropout)
            # self.args.n_hidden = self.encoder.n_out
            self.args.n_hidden = 7168
        else:
            self.encoder = TransformerEmbedding(model=bert,
                                                n_layers=n_bert_layers,
                                                pooling=bert_pooling,
                                                pad_index=pad_index,
                                                dropout=mix_dropout,
                                                requires_grad=True)
            self.encoder_dropout = nn.Dropout(p=encoder_dropout)
            self.args.n_hidden = self.encoder.n_out

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed.to(self.args.device))
            if embed.shape[1] != self.args.n_pretrained:
                self.embed_proj = nn.Linear(embed.shape[1], self.args.n_pretrained).to(self.args.device)
            nn.init.zeros_(self.word_embed.weight)
        return self

    def forward(self):
        raise NotImplementedError

    def loss(self):
        raise NotImplementedError

    def embed(self, words, feats):
        ext_words = words
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.args.unk_index)

        # get outputs from embedding layers
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            pretrained = self.pretrained(words)
            if self.args.n_embed == self.args.n_pretrained:
                word_embed += pretrained
            else:
                word_embed = torch.cat((word_embed, self.embed_proj(pretrained)), -1)

        feat_embeds = []
        if 'tag' in self.args.feat:
            feat_embeds.append(self.tag_embed(feats.pop()))
        if 'char' in self.args.feat:
            feat_embeds.append(self.char_embed(feats.pop(0)))
        if 'elmo' in self.args.feat:
            feat_embeds.append(self.elmo_embed(feats.pop(0)))
        if 'bert' in self.args.feat:
            feat_embeds.append(self.bert_embed(feats.pop(0)))
        if 'lemma' in self.args.feat:
            feat_embeds.append(self.lemma_embed(feats.pop(0)))
        word_embed, feat_embed = self.embed_dropout(word_embed, torch.cat(feat_embeds, -1))
        # concatenate the word and feat representations
        embed = torch.cat((word_embed, feat_embed), -1)

        return embed

    def encode(self, words, feats=None):
        if self.args.encoder == 'lstm':
            x = pack_padded_sequence(self.embed(words, feats), words.ne(self.args.pad_index).sum(1).tolist(), True, False)
            x, _ = self.encoder(x)
            x, _ = pad_packed_sequence(x, True, total_length=words.shape[1])
        else:
            x = self.encoder(words)
        return self.encoder_dropout(x)

    def lal_encode(self, words, feats=None):
        embed = self.encoder(words)
        embed = self.encoder_dropout(embed)

        all_len = [len(w) for w in words]
        packed_len = sum(all_len)
        batch_idxs = np.zeros(packed_len, dtype=int)

        # concatenate the word and feat representations
        flatten_embed = []
        i = 0
        for snum, sentence in enumerate(embed.tolist()):
            for item in sentence:
                flatten_embed.append(item)
                batch_idxs[i] = snum
                i += 1

        assert i == packed_len
        
        batch_idxs = BatchIndices(batch_idxs)
        flatten_embed = torch.Tensor(flatten_embed)

        # enhance word representation by Label Attention layers
        for i, (attn, ff) in enumerate(self.stacks):
            if i == 0:
                res, _ = attn(flatten_embed, batch_idxs)
            else:
                res, _ = attn(res, batch_idxs)
            if ff is not None:
                res = ff(res, batch_idxs)

        split_res = torch.split(res, all_len)

        x = torch.stack(split_res, 0).to(self.args.device)
        # end implement lal
        return x

    def decode(self):
        raise NotImplementedError

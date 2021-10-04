# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch import from_numpy
import numpy as np
from supar.models.model import Model
from supar.modules import MLP, Biaffine, LabelAttention, MultiHeadAttention, PositionwiseFeedForward, PartitionedPositionwiseFeedForward
from supar.utils import Config
from supar.utils.alg import eisner, mst
from supar.utils.transform import CoNLL

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

class LabelAttentionDependencyModel(Model):
    r"""
    The implementation of Biaffine Dependency Parser :cite:`dozat-etal-2017-biaffine`.

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_rels (int):
            The number of labels in the treebank.
        n_tags (int):
            The number of POS tags, required if POS tag embeddings are used. Default: ``None``.
        n_chars (int):
            The number of characters, required if character-level representations are used. Default: ``None``.
        encoder (str):
            Encoder to use.
            ``'lstm'``: BiLSTM encoder.
            ``'bert'``: BERT-like pretrained language model (for finetuning), e.g., ``'bert-base-cased'``.
            Default: ``'lstm'``.
        feat (list[str]):
            Additional features to use, required if ``encoder='lstm'``.
            ``'tag'``: POS tag embeddings.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'bert'``: BERT representations, other pretrained language models like RoBERTa are also feasible.
            Default: [``'char'``].
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_pretrained (int):
            The size of pretrained word embeddings. Default: 100.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if using CharLSTM. Default: 50.
        n_char_hidden (int):
            The size of hidden states of CharLSTM, required if using CharLSTM. Default: 100.
        char_pad_index (int):
            The index of the padding token in the character vocabulary, required if using CharLSTM. Default: 0.
        elmo (str):
            Name of the pretrained ELMo registered in `ELMoEmbedding.OPTION`. Default: ``'original_5b'``.
        elmo_bos_eos (tuple[bool]):
            A tuple of two boolean values indicating whether to keep start/end boundaries of elmo outputs.
            Default: ``(True, False)``.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'``.
            This is required if ``encoder='bert'`` or using BERT features. The full list can be found in `transformers`_.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use, required if ``encoder='bert'`` or using BERT features.
            The final outputs would be weighted sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            The dropout ratio of BERT layers, required if ``encoder='bert'`` or using BERT features. Default: .0.
        bert_pooling (str):
            Pooling way to get token embeddings.
            ``first``: take the first subtoken. ``last``: take the last subtoken. ``mean``: take a mean over all.
            Default: ``mean``.
        bert_pad_index (int):
            The index of the padding token in BERT vocabulary, required if ``encoder='bert'`` or using BERT features.
            Default: 0.
        freeze (bool):
            If ``True``, freezes BERT parameters, required if using BERT features. Default: ``True``.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .33.
        n_lstm_hidden (int):
            The size of LSTM hidden states. Default: 400.
        n_lstm_layers (int):
            The number of LSTM layers. Default: 3.
        encoder_dropout (float):
            The dropout ratio of encoder layer. Default: .33.
        n_arc_mlp (int):
            Arc MLP size. Default: 500.
        n_rel_mlp  (int):
            Label MLP size. Default: 100.
        mlp_dropout (float):
            The dropout ratio of MLP layers. Default: .33.
        scale (float):
            Scaling factor for affine scores. Default: 0.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self,
                 n_words,
                 n_rels,
                 n_tags=None,
                 n_chars=None,
                 encoder='lstm',
                 feat=['char'],
                 n_embed=100,
                 n_pretrained=100,
                 n_feat_embed=100,
                 n_char_embed=50,
                 n_char_hidden=100,
                 char_pad_index=0,
                 elmo='original_5b',
                 elmo_bos_eos=(True, False),
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 freeze=True,
                 embed_dropout=.33,
                 n_lstm_hidden=400,
                 n_lstm_layers=3,
                 encoder_dropout=.33,
                 n_arc_mlp=500,
                 n_rel_mlp=100,
                 mlp_dropout=.33,
                 scale=0,
                 pad_index=0,
                 unk_index=1,
                 lal_d_kv=64, # Dimension of Key and Query Vectors in the LAL
                 lal_d_proj=64, # Dimension of the output vector from each label attention head
                 lal_resdrop=True, # True means the LAL uses Residual Dropout
                 lal_pwff=True, # True means the LAL has a Position-wise Feed-forward Layer
                 lal_q_as_matrix=False, # False means the LAL uses learned query vectors
                 lal_partitioned=True, # Partitioned as per the Berkeley Self-Attentive Parser
                 lal_combine_as_self=False,
                 d_model=200, # d_model must be equal with n_embed + n_feat_embed; 200 = 100 + 100
                 num_layers=3,
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
        super().__init__(**Config().update(locals()))

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

        self.arc_mlp_d = MLP(n_in=self.args.n_hidden, n_out=n_arc_mlp, dropout=mlp_dropout)
        self.arc_mlp_h = MLP(n_in=self.args.n_hidden, n_out=n_arc_mlp, dropout=mlp_dropout)
        self.rel_mlp_d = MLP(n_in=self.args.n_hidden, n_out=n_rel_mlp, dropout=mlp_dropout)
        self.rel_mlp_h = MLP(n_in=self.args.n_hidden, n_out=n_rel_mlp, dropout=mlp_dropout)

        self.arc_attn = Biaffine(n_in=n_arc_mlp, scale=scale, bias_x=True, bias_y=False)
        self.rel_attn = Biaffine(n_in=n_rel_mlp, n_out=n_rels, bias_x=True, bias_y=True)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, words, feats=None):
        r"""
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            feats (list[~torch.LongTensor]):
                A list of feat indices.
                The size is either ``[batch_size, seq_len, fix_len]`` if ``feat`` is ``'char'`` or ``'bert'``,
                or ``[batch_size, seq_len]`` otherwise.
                Default: ``None``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first tensor of shape ``[batch_size, seq_len, seq_len]`` holds scores of all possible arcs.
                The second of shape ``[batch_size, seq_len, seq_len, n_labels]`` holds
                scores of all possible labels on each arc.
        """
        all_len = [len(w) for w in words]
        packed_len = sum(all_len)
        batch_idxs = np.zeros(packed_len, dtype=int)

        x = self.encode(words, feats)

        # concatenate the word and feat representations
        embed = torch.cat(x, -1)

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

        # enhance word representation by Lable Attention layers
        for i, (attn, ff) in enumerate(self.stacks):
            if i == 0:
                res, _ = attn(flatten_embed, batch_idxs)
            else:
                res, _ = attn(res, batch_idxs)
            if ff is not None:
                res = ff(res, batch_idxs)

        split_res = torch.split(res, all_len)

        new_x = torch.stack(split_res, 0).to(self.args.device)
        # end implement lal

        mask = words.ne(self.args.pad_index) if len(words.shape) < 3 else words.ne(self.args.pad_index).any(-1)

        arc_d = self.arc_mlp_d(new_x)
        arc_h = self.arc_mlp_h(new_x)
        rel_d = self.rel_mlp_d(new_x)
        rel_h = self.rel_mlp_h(new_x)

        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h).masked_fill_(~mask.unsqueeze(1), float('-inf'))
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)

        return s_arc, s_rel

    def loss(self, s_arc, s_rel, arcs, rels, mask, partial=False):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            arcs (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard arcs.
            rels (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            partial (bool):
                ``True`` denotes the trees are partially annotated. Default: ``False``.

        Returns:
            ~torch.Tensor:
                The training loss.
        """

        if partial:
            mask = mask & arcs.ge(0)
        s_arc, arcs = s_arc[mask], arcs[mask]
        s_rel, rels = s_rel[mask], rels[mask]
        s_rel = s_rel[torch.arange(len(arcs)), arcs]
        arc_loss = self.criterion(s_arc, arcs)
        rel_loss = self.criterion(s_rel, rels)

        return arc_loss + rel_loss

    def decode(self, s_arc, s_rel, mask, tree=False, proj=False):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.

        Returns:
            ~torch.LongTensor, ~torch.LongTensor:
                Predicted arcs and labels of shape ``[batch_size, seq_len]``.
        """

        lens = mask.sum(1)
        arc_preds = s_arc.argmax(-1)
        bad = [not CoNLL.istree(seq[1:i+1], proj) for i, seq in zip(lens.tolist(), arc_preds.tolist())]
        if tree and any(bad):
            alg = eisner if proj else mst
            arc_preds[bad] = alg(s_arc[bad], mask[bad])
        rel_preds = s_rel.argmax(-1).gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)

        return arc_preds, rel_preds


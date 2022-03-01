import pickle
import sys
from typing import Optional, Dict, List

import torch
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer import TransformerModel, TransformerDecoder, TransformerEncoder, Linear, base_architecture
from torch import Tensor

from ..ETM import etm

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model("topic_transformer")
class TopicTransformerModel(TransformerModel):
    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        parser.add_argument('--add-topic-encoder-pre', default=False, action='store_true',
                            help='')
        parser.add_argument('--add-topic-encoder-post', default=False, action='store_true',
                            help='')
        parser.add_argument('--add-topic-decoder', default=False, action='store_true',
                            help='')

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TopicTransformerEncoder(args, src_dict, embed_tokens, add_topic_pre=args.add_topic_encoder_pre, add_topic_post=args.add_topic_encoder_post)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        if args.add_topic_decoder:
            return TopicTransformerDecoder(
                args,
                tgt_dict,
                embed_tokens,
                no_encoder_attn=getattr(args, "no_cross_attention", False),
            )
        else:
            return TransformerDecoder(
                args,
                tgt_dict,
                embed_tokens,
                no_encoder_attn=getattr(args, "no_cross_attention", False),
            )


class TopicTransformerEncoder(TransformerEncoder):

    def __init__(self, args, dictionary, embed_tokens, add_topic_pre, add_topic_post):
        super().__init__(args, dictionary, embed_tokens)

        self.add_topic_pre, self.add_topic_post = add_topic_pre, add_topic_post

        with open("/cache/code_dir/ETM/checkpoint", 'rb') as f:
            sys.modules["etm"] = etm
            m = torch.load(f)
        m = m.cuda()

        self.topic_embedding = m.rho.weight

        with open('/cache/code_dir/ETM/vocab.pkl', 'rb') as f:
            self.vo = pickle.load(f)

        self.vocab = []
        f = open('/cache/data_dir/dict.txt')
        for row in f.readlines():
            self.vocab.append(row.split(" ")[0])
        f.close()

        self.t = Linear(300, 512)

    def forward(
            self,
            src_tokens,
            src_lengths,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        if self.add_topic_pre:
            x = self.add_topic(x, src_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # add topic start

        if self.add_topic_post:
            x = x.transpose(1, 0)
            x = self.add_topic(x, src_tokens)
            x = x.transpose(0, 1).detach()
        # add topic end

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )

    def add_topic(self, x, src_tokens):
        bsz, src_len, em = x.size()

        topic = []

        for i in range(bsz):
            pre = x.new_zeros(300)
            for tk in src_tokens[i]:
                if tk < len(self.vocab) and self.vocab[tk] in self.vo:
                    i = self.vo.index(self.vocab[tk])
                    pre += self.topic_embedding[i]
            topic.append(pre)

        topic = torch.stack(topic, dim=0)
        topic = topic.unsqueeze(1)
        topic = topic.repeat(1, src_len, 1)

        predict = self.t(topic)

        x += predict
        return x


class TopicTransformerDecoder(TransformerDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super(TopicTransformerDecoder, self).__init__(args, dictionary, embed_tokens, no_encoder_attn)

        with open("/cache/code_dir/ETM/checkpoint", 'rb') as f:
            sys.modules["etm"] = etm
            m = torch.load(f)
        m = m.cuda()

        self.topic_embedding = m.rho.weight

        with open('/cache/code_dir/ETM/vocab.pkl', 'rb') as f:
            self.vo = pickle.load(f)

        self.vocab = []
        f = open('/cache/data_dir/dict.txt')
        for row in f.readlines():
            self.vocab.append(row.split(" ")[0])
        f.close()

    def extract_features_scriptable(
            self,
            prev_output_tokens,
            encoder_out: Optional[EncoderOut] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        # add topic start
        bsz, tgt_len = prev_output_tokens.size()
        topic_dec = []

        for i in range(bsz):
            pre = x.new_zeros(512)

            for tk in prev_output_tokens[i]:
                if tk < len(self.vocab) and self.vocab[tk] in self.vo:
                    i = self.vo.index(self.vocab[tk])
                    pre += self.topic_embedding[i]

            topic_dec.append(pre)

        topic_dec = torch.stack(topic_dec, dim=0)

        topic = topic_dec
        topic = topic.unsqueeze(1)

        topic = topic.repeat(1, tgt_len, 1)

        x = x.transpose(1, 0)
        x += topic
        x = x.transpose(0, 1).detach()
        # add topic end

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}


@register_model_architecture("topic_transformer", "topic_transformer")
def topic_trf_base_architecture(args):
    base_architecture(args)

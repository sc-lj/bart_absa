from fastNLP.io import Pipe, DataBundle, Loader
import os
import json
from fastNLP import DataSet, Instance
from fitlog.fastlog import logger
from transformers import AutoTokenizer,BertTokenizer,BartTokenizer,RobertaTokenizer
import numpy as np
from itertools import chain
from functools import cmp_to_key
import pandas as pd 

def cmp_aspect(v1, v2):
    if v1[0]['from']==v2[0]['from']:
        return v1[1]['from'] - v2[1]['from']
    return v1[0]['from'] - v2[0]['from']

def cmp_opinion(v1, v2):
    if v1[1]['from']==v2[1]['from']:
        return v1[0]['from'] - v2[0]['from']
    return v1[1]['from'] - v2[1]['from']


class BartBPEABSAPipe(Pipe):
    def __init__(self, tokenizer='fnlp/bart-base-chinese', opinion_first=False):
        super(BartBPEABSAPipe, self).__init__()
        # self.tokenizer = BartTokenizer.from_pretrained(tokenizer)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.tokenizer.decoder = {ids:tok for tok, ids in self.tokenizer.vocab.items()}
        self.mapping = {  # so that the label word can be initialized in a better embedding.
            'POS': '<<positive>>',
            'NEG': '<<negative>>',
            'NEU': '<<neutral>>'
        }
        self.opinion_first = opinion_first  # 是否先生成opinion

        cur_num_tokens = self.tokenizer.vocab_size
        self.cur_num_token = cur_num_tokens

        tokens_to_add = sorted(list(self.mapping.values()), key=lambda x:len(x), reverse=True)
        unique_no_split_tokens = self.tokenizer.unique_no_split_tokens
        sorted_add_tokens = sorted(list(tokens_to_add), key=lambda x:len(x), reverse=True)
        for tok in sorted_add_tokens:
            assert self.tokenizer.convert_tokens_to_ids([tok])[0]==self.tokenizer.unk_token_id
        self.tokenizer.unique_no_split_tokens = unique_no_split_tokens + sorted_add_tokens
        self.tokenizer.add_tokens(sorted_add_tokens)
        self.mapping2id = {}
        self.mapping2targetid = {}

        for key, value in self.mapping.items():
            key_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value))
            assert len(key_id) == 1, value
            assert key_id[0] >= cur_num_tokens
            self.mapping2id[key] = key_id[0]
            self.mapping2targetid[key] = len(self.mapping2targetid)

    def process(self, data_bundle: DataBundle) -> DataBundle:
        """
        words: List[str]
        aspects: [{
            'index': int
            'from': int
            'to': int
            'polarity': str
            'term': List[str]
        }],
        opinions: [{
            'index': int
            'from': int
            'to': int
            'term': List[str]
        }]

        输出为[o_s, o_e, a_s, a_e, c]或者[a_s, a_e, o_s, o_e, c]
        :param data_bundle:
        :return:
        """
        target_shift = len(self.mapping) + 2  # 是由于第一位是sos，紧接着是eos, 然后是

        def prepare_target(ins):
            text = ins['text']
            word_bpes = [self.tokenizer.cls_token_id]
            bpes = self.tokenizer.tokenize(text)
            bpes = self.tokenizer.convert_tokens_to_ids(bpes)
            word_bpes += bpes
            word_bpes += [self.tokenizer.sep_token_id]
            if len(word_bpes)>512:
                word_bpes = word_bpes[:512]

            target = [0]  # 特殊的开始
            target_spans = []

            aspects_opinions = [(a, o) for a, o in zip(ins['aspects'], ins['opinions'])]
            if self.opinion_first:
                aspects_opinions = sorted(aspects_opinions, key=cmp_to_key(cmp_opinion))
            else:
                aspects_opinions = sorted(aspects_opinions, key=cmp_to_key(cmp_aspect))

            for aspects, opinions in aspects_opinions:  # 预测bpe的start
                aspect_term = aspects['term']
                opinion_term = opinions['term']
                # aspect_index = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(aspect_term, add_prefix_space=True)[:1])
                # opinion_index = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(opinion_term, add_prefix_space=True)[:1])
                aspect_index = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(str(aspect_term)))
                opinion_index = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(str(opinion_term)))
                a_o_bpe = self.match_index(word_bpes,aspect_index,opinion_index)
                if len(a_o_bpe)==0:
                    print(aspect_term,opinion_term,text)
                    continue
                a_bpe,o_bpe = a_o_bpe
                a_start_bpe,a_end_bpe = a_bpe
                a_end_bpe = a_end_bpe - 1 # 这里由于之前是开区间，刚好取到最后一个word的开头
                o_start_bpe,o_end_bpe = o_bpe
                o_end_bpe = o_end_bpe - 1 # 这里由于之前是开区间，刚好取到最后一个word的开头
                # 这里需要evaluate是否是对齐的
                # for idx, word in zip((o_start_bpe, o_end_bpe, a_start_bpe, a_end_bpe),
                #                      (opinions['term'][0], opinions['term'][-1],aspects['term'][0], aspects['term'][-1])):
                    # assert word_bpes[idx] == self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word, add_prefix_space=True)[:1])[0] or \
                    #        word_bpes[idx] == self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word, add_prefix_space=True)[-1:])[0]
                    # assert word_bpes[idx] == self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word))[0] or \
                    #        word_bpes[idx] == self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word))[-1]
                if self.opinion_first:
                    # target_spans.append([o_start_bpe+target_shift, o_end_bpe+target_shift,
                    #                      a_start_bpe+target_shift, a_end_bpe+target_shift])
                    target_spans.append([o_start_bpe, o_end_bpe,a_start_bpe, a_end_bpe])
                else:
                    # target_spans.append([a_start_bpe+target_shift, a_end_bpe+target_shift,
                    #                      o_start_bpe+target_shift, o_end_bpe+target_shift])
                    target_spans.append([a_start_bpe, a_end_bpe,o_start_bpe, o_end_bpe])
                target_spans[-1].append(self.mapping2targetid[aspects['polarity']]+2)   # 前面有sos和eos
                target_spans[-1] = tuple(target_spans[-1])
            target.extend(list(chain(*target_spans)))
            target.append(1)  # append 1是由于特殊的eos

            return {'tgt_tokens': target, 'target_span': target_spans, 'src_tokens': word_bpes}

        data_bundle.apply_more(prepare_target, use_tqdm=True, tqdm_desc='Pre. tgt.')

        data_bundle.set_ignore_type('target_span')
        data_bundle.set_pad_val('tgt_tokens', 0)  # 设置为pad所在的id
        data_bundle.set_pad_val('src_tokens', self.tokenizer.pad_token_id)

        data_bundle.apply_field(lambda x: len(x), field_name='src_tokens', new_field_name='src_seq_len')
        data_bundle.apply_field(lambda x: len(x), field_name='tgt_tokens', new_field_name='tgt_seq_len')
        data_bundle.set_input('tgt_tokens', 'src_tokens', 'src_seq_len', 'tgt_seq_len')
        data_bundle.set_target('tgt_tokens', 'tgt_seq_len', 'target_span')

        return data_bundle
    
    def match_index(self,word_bpes,aspect_index,opinion_index):
        """[匹配索引]

        Args:
            word_bpes ([type]): [description]
            aspect_index ([type]): [description]
            opinion_index ([type]): [description]
        """
        length = len(word_bpes)
        expect_aspect_index = []
        expect_opinion_index = []
        for i in range(length):
            if aspect_index == word_bpes[i:i+len(aspect_index)]:
                expect_aspect_index.append([i,i+len(aspect_index)])
            if opinion_index == word_bpes[i:i+len(opinion_index)]:
                expect_opinion_index.append([i,i+len(opinion_index)])
        
        min_dis = float('inf')
        min_pair = []
        for a_index in expect_aspect_index:
            for o_index in expect_opinion_index:
                if min(abs(a_index[-1]-o_index[0]),abs(a_index[0]-o_index[-1])) <min_dis:
                    min_pair=(a_index,o_index)
                    min_dis = min(abs(a_index[-1]-o_index[0]),abs(a_index[0]-o_index[-1]))
                    continue
        return min_pair


    def process_from_file(self, paths, demo=False) -> DataBundle:
        """
        :param paths: 支持路径类型参见 :class:`fastNLP.io.loader.ConllLoader` 的load函数。
        :return: DataBundle
        """
        # 读取数据
        data_bundle = ABSALoader(demo=demo).load(paths)
        data_bundle = self.process(data_bundle)

        return data_bundle

sentiment = {"0":"NEG","1":"NEU",'2':"POS"}

class ABSALoader(Loader):
    def __init__(self, demo=False):
        super().__init__()
        self.demo = demo

    def _load(self, path):
        data = pd.read_csv(path)
        ds = DataSet()
        for ins in data.to_dict('records'):
            text = ins['text']
            lines = eval(ins['topic_list'])
            aspects = []
            opinions = []
            for line in lines:
                if len(line['aspect']) == 0 or len(str(line['opinion'])) == 0:
                    continue
                aspect = {"from": line['aspect_index'][0],
                        "to": line['aspect_index'][1],
                        "polarity": sentiment[str(line['sentiment'])],
                        "term": line['aspect']}
                opinion = {"from": line['opinion_index'][0],
                        "to": line['opinion_index'][1],
                        "term": line['opinion']}

                aspects.append(aspect)
                opinions.append(opinion)

            assert len(aspects)==len(opinions)
            ins = Instance(text=text, aspects=aspects, opinions=opinions)
            ds.append(ins)
            if self.demo and len(ds)>30:
                break
        return ds


if __name__ == '__main__':
    data_bundle = BartBPEABSAPipe().process_from_file('pengb/16res')
    print(data_bundle)


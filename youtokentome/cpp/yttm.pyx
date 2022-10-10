from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from libcpp.string cimport string
from libcpp cimport bool
import os
from pathlib import Path
from typing import Collection


cdef extern from "utils.h" namespace "vkcom":
    cdef cppclass SpecialTokens:
        int pad_id
        int unk_id
        int bos_id
        int eos_id

    cdef cppclass Status:
        int code
        string message

    cdef unsigned int DEFAULT_SEED = DEFAULT_SEED


cdef extern from "bpe.h" namespace "vkcom":
    cdef cppclass BpeTrainConfig:
        double character_coverage
        int n_threads
        SpecialTokens special_tokens

    cdef cppclass BpeEncodingConfig:
        bool bos
        bool eos
        bool reverse
        double dropout_prob
        unsigned int dropout_seed

cdef extern from "bpe.h" namespace "vkcom":
    Status train_bpe(const string &source_path, const string& model_path, int vocab_size, BpeTrainConfig bpe_config)

cdef extern from "bpe.h" namespace "vkcom":
    cdef cppclass BpeModel:
        BpeModel(const string& model_path, int n_threads, Status* status)

        Status encode_as_ids(const vector[string] &sentences, vector[vector[int]]* ids, BpeEncodingConfig config) const
        Status encode_as_subwords(const vector[string]& sentences, vector[vector[string]]* subwords, BpeEncodingConfig config) const

        Status encode_cli(string output_type, bool stream, BpeEncodingConfig config) const

        Status decode_cli(const unordered_set[int]* ignore_ids) const

        void vocab_cli(bool verbose) const

        Status id_to_subword(int id, string* subword) const

        int subword_to_id(const string &subword) const
        Status decode(const vector[vector[int]]& ids, vector[string]* output, const unordered_set[int]* ignore_ids) const
        int vocab_size() const
        vector[string] vocabulary() const


cdef class BPE:
    cdef BpeModel* encoder

    def __dealloc__(self):
        del self.encoder

    def __init__(self, model_path, n_threads=-1):
        cdef Status status
        self.encoder = new BpeModel(model_path.encode(), n_threads, &status)
        if status.code != 0:
            raise ValueError(status.message.decode())

    @staticmethod
    def train(data,
              model,
              vocab_size,
              coverage=1.0,
              n_threads=-1,
              pad_id=0,
              unk_id=1,
              bos_id=2,
              eos_id=3):
        cdef BpeTrainConfig bpe_train_config
        bpe_train_config.character_coverage = coverage
        bpe_train_config.n_threads = n_threads
        bpe_train_config.special_tokens.pad_id = pad_id
        bpe_train_config.special_tokens.unk_id = unk_id
        bpe_train_config.special_tokens.bos_id = bos_id
        bpe_train_config.special_tokens.eos_id = eos_id

        cdef Status status = train_bpe(data.encode(), model.encode(), vocab_size, bpe_train_config)
        if status.code != 0:
            raise ValueError(status.message.decode())

    def encode(self, sentences, output_type, bos, eos, reverse, dropout_prob, dropout_seed):
        if dropout_prob < 0 or dropout_prob > 1:
            raise ValueError("dropout_prob value must be in the range [0, 1]. Current value of dropout_prob = " + str(dropout_prob))

        cdef vector[string] s
        cdef vector[vector[string]] ret_subwords
        cdef vector[vector[int]] ret_ids
        cdef Status status

        cdef BpeEncodingConfig config
        config.bos = bos
        config.eos = eos
        config.reverse = reverse
        config.dropout_prob = dropout_prob
        config.dropout_seed = dropout_seed if dropout_seed is not None else DEFAULT_SEED

        if output_type == 'id':
            if isinstance(sentences, str):
                s = [sentences.encode()]
                status = self.encoder.encode_as_ids(s, &ret_ids, config)
                if status.code != 0:
                    raise ValueError(status.message.decode())
                return ret_ids[0]

            assert isinstance(sentences, list) or isinstance(sentences, tuple)
            s = [x.encode() for x in sentences]
            status = self.encoder.encode_as_ids(s, &ret_ids, config)
            if status.code != 0:
                raise ValueError(status.message.decode())
            return ret_ids
        elif output_type == 'subword':
            if isinstance(sentences, str):
                s = [sentences.encode()]
                status = self.encoder.encode_as_subwords(s, &ret_subwords, config)
                if status.code != 0:
                    raise ValueError(status.message.decode())
                assert len(ret_subwords) == 1
                return [piece.decode() for piece in ret_subwords[0]]

            assert isinstance(sentences, list) or isinstance(sentences, tuple)
            s = [x.encode() for x in sentences]
            status = self.encoder.encode_as_subwords(s, &ret_subwords, config)
            if status.code != 0:
                raise ValueError(status.message.decode())
            return [[piece.decode() for piece in sentence] for sentence in ret_subwords]
        else:
            raise ValueError('output_type must be equal to "id" or "subword"')

    def subword_to_id(self, subword):
        return self.encoder.subword_to_id(subword.encode())

    def id_to_subword(self, id):
        cdef string subword
        cdef Status status = self.encoder.id_to_subword(id, &subword)
        if status.code != 0:
            raise ValueError(status.message.decode())
        return subword.decode()

    def decode(self, ids, ignore_ids):

        if not isinstance(ids, list):
            raise TypeError(
                "{} is not a list instance".format(type(ids))
            )

        if not isinstance(ignore_ids, Collection) and ignore_ids is not None:
            raise TypeError(
                "{} is not a Collection instance".format(type(ignore_ids))
            )

        if len(ids) > 0 and isinstance(ids[0], int):
            ids = [ids]
        if ignore_ids is None:
            ignore_ids = set()

        cdef vector[string] sentences
        cdef unordered_set[int] c_ignore_ids = unordered_set[int](ignore_ids)
        cdef Status status = self.encoder.decode(ids, &sentences, &c_ignore_ids)
        if status.code != 0:
            raise ValueError(status.message.decode())
        return [sentence.decode() for sentence in sentences]

    def vocab_size(self):
        return self.encoder.vocab_size();

    def vocab(self):
        cdef vector[string] vocab = self.encoder.vocabulary()
        return [token.decode() for token in vocab]

    def encode_cli(self, output_type, stream, bos, eos, reverse, dropout_prob, dropout_seed):
        cdef BpeEncodingConfig config
        config.bos = bos
        config.eos = eos
        config.reverse = reverse
        config.dropout_prob = dropout_prob
        config.dropout_seed = dropout_seed if dropout_seed is not None else DEFAULT_SEED

        cdef Status status = self.encoder.encode_cli(output_type.encode(), stream, config)
        if status.code != 0:
            raise ValueError(status.message.decode())

    def decode_cli(self, ignore_ids):
        if ignore_ids is None:
            ignore_ids = set()
        cdef unordered_set[int] c_ignore_ids = unordered_set[int](ignore_ids)
        cdef Status status = self.encoder.decode_cli(&c_ignore_ids)
        if status.code != 0:
            raise ValueError(status.message.decode())

    def vocab_cli(self, verbose):
        self.encoder.vocab_cli(verbose)


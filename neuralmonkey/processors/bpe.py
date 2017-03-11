import codecs
import re
from typing import List

from neuralmonkey.logging import log
from lib.subword_nmt.apply_bpe import BPE, encode

# pylint: disable=too-few-public-methods


class BPEPreprocessor(object):
    """Wrapper class for Byte-Pair Encoding.

    Paper: https://arxiv.org/abs/1508.07909
    Code: https://github.com/rsennrich/subword-nmt
    """

    def __init__(self, **kwargs):

        if "merge_file" not in kwargs:
            raise Exception("No merge file for BPE preprocessor")

        log("Initializing BPE preprocessor")

        separator = kwargs.get("separator", "@@")
        merge_file = kwargs["merge_file"]
        merge_type = kwargs["merge_type"]

        with codecs.open(merge_file, "r", "utf-8") as f_data:
            self.bpe = BPE(f_data, separator, merge_type)

    def __call__(self, sentence: List[str]) -> List[str]:
        """Adapted code from BPE.segment """

        output = []
        for word in sentence:

            # Hack. TODO: inspect why there are empty sentences
            if len(word) == 0:
                output.append(word)
                continue

            new_word = encode(word, self.bpe.bpe_codes)

            if self.bpe.merge_type == 'prefix':
                #Prefix
                for item in new_word[:-1]:
                    output.append(item + self.bpe.separator)
                output.append(new_word[-1])

            elif self.bpe.merge_type == 'suffix':
                #Suffix
                output.append(new_word[0])
                for item in new_word[1:]:
                    output.append(self.bpe.separator + item)

            elif self.bpe.merge_type == 'both':
                #Prefix & Suffix
                if len(new_word) > 1:
                    output.append('|@' + new_word[0] + self.bpe.separator)
                    for item in new_word[1:-1]:
                        output.append(self.bpe.separator + item + self.bpe.separator)
                    output.append(self.bpe.separator + new_word[-1] + '@|')
                else:
                    output.append(new_word[0])

        return output


class BPEPostprocessor(object):

    def __init__(self, **kwargs):
        self.separator = kwargs.get("separator", "@@")
        self.merge_type = kwargs["merge_type"]

        esc = re.escape(self.separator)
        esc2 = re.escape("|@")
        esc3 = re.escape("@|")
        if self.merge_type == 'prefix':
            self.pattern = re.compile(esc + r" ")
        elif self.merge_type == 'suffix':
            self.pattern = re.compile(r" " + esc)
        elif self.merge_type == 'both':
            self.pattern = re.compile(r" " + esc)
            self.pattern2 = re.compile(esc + r" ")
            self.pattern3 = re.compile(esc2)
            self.pattern4 = re.compile(esc3)

    def __call__(self, decoded_sentences: List[List[str]]) -> List[List[str]]:
        return [self.decode(s) for s in decoded_sentences]

    def decode(self, sentence: List[str]) -> List[str]:
        joined = " ".join(sentence)
        if self.merge_type == 'both':
            decoded = self.pattern.sub("", joined)
            decoded = self.pattern2.sub("", decoded)
            decoded = self.pattern3.sub("", decoded)
            decoded = self.pattern4.sub("", decoded)
        else:
            decoded = self.pattern.sub("", joined)
        splitted = decoded.split(" ")

        return splitted

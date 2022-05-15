import argparse
import re
import sys
from argparse import RawTextHelpFormatter
import sentencepiece as spm
from nltk.tokenize import TweetTokenizer
import os

trained = False


def wst(sentence):
    _words = re.split(r'\s+', sentence)
    for i, e in enumerate(_words):
        _words[i] = e.lower()
    return _words


def sentpiece(sentence):
    global trained
    if trained is False:
        spm.SentencePieceTrainer.Train(
            f'--input=botchan.txt,file1.txt,file2.txt,file3.txt,file4.txt,file5.txt --model_prefix=m --vocab_size=1908')
        trained = True

    sp = spm.SentencePieceProcessor()
    sp.Load('m.model')

    _words = sp.EncodeAsPieces(sentence)
    for i, e in enumerate(_words):
        _words[i] = e.lower()
    return _words


def ret(sentence):
    _words = re.findall(
        r'[@#$][\d\w]*|http[^\s()\'"]*|\d+%|(?:[A-Z](?!\w)[\.\s]?)+|[:;<xX]+\s*[-]?\s*[PpDd*()/\\380Oo]?|\w+|\S',
        sentence)
    for i, e in enumerate(_words):
        _words[i] = e.lower()
        _words[i] = re.sub(r'\s+', '', _words[i])
    return _words


def twt(sentence):
    tk = TweetTokenizer()
    _words = tk.tokenize(sentence)
    for i, e in enumerate(_words):
        _words[i] = e.lower()
    return _words


def tokenize(file, method, output):
    fmap = {
        'ret': ret,
        'wst': wst,
        'sentpiece': sentpiece,
        'twt': twt
    }
    f = open(file, "r")
    index = re.search(r'\d+', os.path.basename(file)).group(0)
    tweets = f.read().splitlines()
    tokenized = []
    for tweet in tweets:
        tokenized.append(fmap[method](tweet))

    if output is True:
        if os.path.exists(f'output_{index}.txt'):
            os.remove(f'output_{index}.txt')

        o = open(f'output_{index}.txt', "a", encoding='utf-8')

        for i, e in enumerate(tokenized):
            temp = ', '.join(e)
            buffer = f'{tweets[i]}\n{len(e)} tokens\n[{temp}]\n\n'
            o.write(buffer)
    else:
        for i, e in enumerate(tokenized):
            temp = ', '.join(e)
            print(f'{tweets[i]}\n{len(e)} tokens\n[{temp}]\n\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='This program is created by Kamil Agliullin for tweet tokenizing '
                                                 'during NLP Assignment 1', formatter_class=RawTextHelpFormatter)

    required = parser.add_argument_group("required arguments")
    required.add_argument("-method", required=True, help="wst :  White Space Tokenization\n"
                                                         "sentpiece : Sentencepiece tokenizer\n"
                                                         "ret : Tokenizing text using regular expressions\n"
                                                         "twt : Tokenizing text using TweetTokenizer", )

    required.add_argument("-source", required=True, help="path to file where the input source is stored ")
    required.add_argument("-output", action='store_true', help="If specified, stores output in file. If not"
                                                               ", the results are printed in terminal")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    tokenize(args.source, args.method, args.output)

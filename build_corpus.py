# coding: utf-8
#!/usr/bin/python2
import argparse
import codecs
import lxml.etree as ET
import os
import regex
import logging
import subprocess
import math
import time
import traceback
from multiprocessing import Pool, Queue

logger = logging.getLogger('Build_corpus')
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

info    = logger.info
warning = logger.warning
debug   = logger.debug
error   = logger.error

# arguments setting 
parser = argparse.ArgumentParser()
parser.add_argument('--lcode', type=str, default='ko', help='ISO 639-1 code of target language. See `lcodes.txt`.')
parser.add_argument('--date', type=str, default='20161201', help='Wiki document dumped date.')
parser.add_argument('--max_corpus_size', type=int, default=1000000000, help='the maximum size of the corpus. Feel free to adjust it according to your computing power.')
parser.add_argument('--nproc', type=int, default=1, help='the number of processes. It should be a number under the number of your cpu cores.')

args = parser.parse_args()
lcode = args.lcode
date  = args.date
nproc = args.nproc

if lcode == 'ko-kkma':
    from konlpy.tag import Kkma # pip install konlpy. See http://konlpy.org/en/v0.4.4/ for further information.
    komorph = Kkma()
    info("kkma succesfuly loaded!")

elif lcode == 'ko-kiwi':
    from kiwipiepy import Kiwi  # pip install kiwipiepy. See https://github.com/bab2min/kiwipiepy for further information.
    komorph = Kiwi()
    info("kiwi succesfuly loaded!")

elif lcode == 'ja':
    import MeCab # See https://pypi.python.org/pypi/mecab-python/0.996
    mecab = MeCab.Tagger("-Owakati")
    info("mecab succesfuly loaded!")

elif lcode == 'zh':
    import jieba # See https://pypi.python.org/pypi/jieba/
    info("jieba succesfuly loaded!")

elif lcode == 'vi':
    from pyvi.pyvi import ViTokenizer # See https://pypi.python.org/pypi/pyvi
    info("pyvi succesfuly loaded!")

elif lcode == 'th':  
    import pythai # See https://pypi.python.org/pypi/pythai  
    info("pythai succesfuly loaded!")

# elif lcode == 'ar':
#     os.environ['CLASSPATH'] = "../stanford-segmenter-2015-12-09"
#     from nltk.tokenize.stanford_segmenter import StanfordSegmenter
#     segmenter = StanfordSegmenter(path_to_jar="../stanford-segmenter-2015-12-09/stanford-segmenter-3.6.0.jar", 
#                                path_to_sihan_corpora_dict="../stanford-segmenter-2015-12-09/data", 
#                                path_to_model="../stanford-segmenter-2015-12-09/data/pku.gz", 
#                                path_to_dict="../stanford-segmenter-2015-12-09/data/dict-chris6.ser.gz")
#     print "StanfordSegmenter succesfuly loaded!"
    
max_corpus_size = args.max_corpus_size
if lcode in ['ko', 'ko-kkma', 'ko-kiwi']:  # korean
    fname = f'kowiki-{date}-pages-articles-multistream.xml'
else:
    fname = f'{lcode}wiki-{date}-pages-articles-multistream.xml'

def clean_text(text):
    global lcode
    
    # Common
    text = regex.sub("(?s)<ref>.+?</ref>", "", text) # remove reference links
    text = regex.sub("(?s)<[^>]+>", "", text) # remove html tags
    text = regex.sub("&[a-z]+;", "", text) # remove html entities
    text = regex.sub("(?s){{.+?}}", "", text) # remove markup tags
    text = regex.sub("(?s){.+?}", "", text) # remove markup tags
    text = regex.sub("(?s)\[\[([^]]+\|)", "", text) # remove link target strings
    text = regex.sub("(?s)\[\[([^]]+\:.+?]])", "", text) # remove media links
    
    text = regex.sub("[']{5}", "", text) # remove italic+bold symbols
    text = regex.sub("[']{3}", "", text) # remove bold symbols
    text = regex.sub("[']{2}", "", text) # remove italic symbols
    
    if lcode in ['ko', 'ko-kkma', 'ko-kiwi']: # korean
        text = regex.sub(u"[^ \r\n\p{Hangul}.?!]", " ", text) # Replace unacceptable characters with a space.
    elif lcode in ['ja']: # japanese
        text = regex.sub(u"[^\r\n\p{Han}\p{Hiragana}\p{Katakana}ー。！？]", "", text)
    elif lcode in ['zh']: # chinsese
        text = regex.sub(u"[^\r\n\p{Han}。！？]", "", text)
    elif lcode in ['th']: # thai
        text = regex.sub(u"[^ \r\n\p{Thai}.?!]", " ", text)
    elif lcode in ['ru']: # russian
        text = regex.sub(u"[^ \r\n\p{Cyrillic}.?!\-]", " ", text)
        text = text.lower()
#     elif lcode in ['ar']: # arabic
#         text = regex.sub(u"[^ \r\n\p{Arabic}.?!\-]", " ", text)
    elif lcode in ['hi']: # hindi
        text = regex.sub(u"[^ \r\n\p{Devanagari}.।?!\-]", " ", text)
    elif lcode in ['bn']: # bengali
        text = regex.sub(u"[^ \r\n\p{Bengali}.।?!\-]", " ", text)
    elif lcode in ['de']: # german
        text = regex.sub(u"[^ \r\n\p{Latin}\-'‘’.?!]", " ", text)
    else: # Mostly european languages
        text = regex.sub(u"[^ \r\n\p{Latin}\-'‘’.?!]", " ", text)
        text = text.lower()
    
    # Common
    text = regex.sub("[ ]{2,}", " ", text) # Squeeze spaces.
    return text

def sentence_segment(text):
    '''
    Args:
      text: A string. A unsegmented paragraph.
    
    Returns:
      A list of sentences.
    '''
    global lcode
    if lcode in ['ja', 'zh']:
        sents = regex.split(u"([。！？])?[\n]+|[。！？]", text) 
    elif lcode in ['th']:
        sents = text.split("[\n]+") 
    elif lcode in ['hi', 'bn']: # hindi, bengali
        sents = regex.split(u"([.।?!])?[\n]+|[.।?!] ", text)
    elif lcode in ['de']: # german
        sents = regex.split("([.?!])?[\n]+|[.?!] ", text)
        sents = [sent[0].lower() + sent[1:] for sent in sents if sent is not None and len(sent) > 1]
    elif lcode in ['ko', 'ko-kkma']:
        sents = regex.split("([.?!])?[\n]+|[.?!] ", text)
    elif lcode in ['ko-kiwi']:
        sents = komorph.split_into_sents(text)
    return sents
        
def word_segment(sent):
    '''
    Args:
      sent: A string. A sentence.
    
    Returns:
      A list of words.
    '''
    global lcode
    if lcode in ['ko-kkma']:
        words = [word for word, _ in komorph.pos(sent)]
    elif lcode in ['ko-kiwi']:
        words = [token.form for token in komorph.tokenize(sent.text)]
    elif lcode in ['ja']:
        words = mecab.parse(sent.encode('utf8')).split() 
    elif lcode in ['th']:
        words = pythai.split(sent)
    elif lcode in ['vi']:
        words = ViTokenizer.tokenize(sent).split()        
    elif lcode in ['zh']:
        words = list(jieba.cut(sent, cut_all=False)) 
#     elif lcode in ['ar']:
#         words = segmenter.segment(sent).split()
    else: # Mostly european languages
        words = sent.split()
    
    return words

def count_size(file):
    command = f"grep -o '<text ' {file} | wc -l"
    result = subprocess.check_output(command, shell=True, text=True).strip()
    return int(result)


def process_text(text):
    text   = clean_text(text)
    sents  = sentence_segment(text)
    output = ''

    for sent in sents:
        if sent is not None and sent != '' and sent != ' ':
            line = ''
            words = word_segment(sent)
            if len(words) > 10:
                if lcode in ['ja']:
                    line = " ".join(words).decode('utf8') + "\n"
                else:
                    line = " ".join(words) + "\n"
                output += line
    return output


elem_size = 0
job_no = 0
prev_rate = -1
results = Queue()


def progress(result):
    global elem_size, job_no, prev_rate, results
    if type(result) is str:
        results.put(result)
    else:
        debug(f'result : {type(result)}, {result}')
    job_no +=1
    i = job_no
    rate = round(i / elem_size * 100, 1)
    if prev_rate != rate:
        if rate.is_integer():
            info(f'[{i}/{elem_size}]: {int(rate)} %')
        else:
            info(f'[{i}/{elem_size}]: {rate} %')
        prev_rate = rate


def build_corpus():
    global lcode, max_corpus_size, fname, elem_size
    target_file = f"data/{fname}"

    if lcode in ['ja', 'th', 'vi', 'zh']:
        output_file = f"data/{lcode}.txt"
    elif lcode in ['ko', 'ko-kkma', 'ko-kiwi']:
        output_file = "data/ko.txt"
    else:
        info(f'Improper target language code : {lcode}')
        exit(0)

    output = ''
    i = 0
    ns = "{http://www.mediawiki.org/xml/export-0.10/}" # namespace

    info(f'Start loading file : {target_file}')
    elem_iter = ET.iterparse(target_file, tag=ns+"text")
    info(f'Loaded file : {target_file}')

    elem_size = count_size(target_file)
    info(f'Total {elem_size} elements in the target file : {target_file}')

    if nproc == 1:
        for _, elem in elem_iter:
            i += 1

            try:
                output += process_text(elem.text)
            except Exception as e:
                debug(f'Exception occured : {e}')
                traceback.print_exc()
                continue # it's okay as we have a pretty big corpus!
            elem.clear() # We need to save memory!

            rate = round(i/elem_size*100, 1)
            debug(f'[{i}/{elem_size}]: {rate}% elapsed')

            if len(output) > max_corpus_size:
                break

        with codecs.open(output_file, 'w', 'utf-8') as fout:
            fout.write(output)

    elif nproc > 1:
        p = Pool(nproc)
        for _, elem in elem_iter:
            try:
                fout = p.apply_async(process_text, (elem.text,), callback=progress)
            except Exception as e:
                error(f'Exception occured : {e}')
                traceback.print_exc()
                elem.clear()
                continue # it's okay as we have a pretty big corpus!
            elem.clear() # We need to save memory!

        p.close()
        p.join()

        with codecs.open(output_file, 'w', 'utf-8') as fout:
            while True:
                line = results.get()
                fout.write(line)
                if results.empty():
                    break


if __name__ == "__main__":
    build_corpus()
    
    info("Done")

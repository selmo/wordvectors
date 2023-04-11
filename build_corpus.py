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
from multiprocessing import Pool

logger = logging.getLogger('Build_corpus')
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

info    = logger.info
warning = logger.warning
debug   = logger.debug

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

if lcode == 'ko':
    from konlpy.tag import Kkma # pip install konlpy. See http://konlpy.org/en/v0.4.4/ for further information.
    kkma = Kkma()
    info("kkma succesfuly loaded!")
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
    
    if lcode in ['ko']: # korean
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
    else:
        sents = regex.split("([.?!])?[\n]+|[.?!] ", text)
    return sents
        
def word_segment(sent):
    '''
    Args:
      sent: A string. A sentence.
    
    Returns:
      A list of words.
    '''
    global lcode
    if lcode in ['ko']:
        words = [word for word, _ in kkma.pos(sent)]
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


def mprocess_text(text):
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

"""
def mprocess_text(pid, q_in, q_out, i, size):
    while True:
        if q_in.empty():
            debug(f'[{pid}] : queue is empty. wait for queueing...')
            time.sleep(0.1)
        else:
            idx, text = q_in.get()

            if idx == -1 and text == '# EXIT #':
                break
            else:
                info(f'[{pid}] start process_text.')
                try:
                    output = process_text(text)
                except Exception as e:
                    debug(f'[{pid}] Exception occured : {e}')
                    traceback.print_exc()
                    continue
                ci = i.value
                i.value += 1
                progress = round(ci/size*100, 2)
                info(f'[{pid}] [{ci}/{size}]: {progress}% elapsed')
                q_out.append(output)
"""


def callback_func(result):
    text = result.strip()
    if len(text) > 10:
        info(f'{os.getpid()} : {text[:10]}...')
    else:
        info(f'{os.getpid()} : {text}')

def build_corpus():
    global lcode, max_corpus_size, fname
    target_file = f"data/{fname}"

    if lcode in ['ko', 'ja', 'th', 'vi', 'zh']:
        output_file = f"data/{lcode}.txt"
    else:
        info(f'Improper target language code : {lcode}')
        exit(0)

    output = ''
    i = 0
    ns = "{http://www.mediawiki.org/xml/export-0.10/}" # namespace
    info(f'Start loading file : {target_file}')
    elem_iter = ET.iterparse(target_file, tag=ns+"text")
    info(f'Loaded file : {target_file}')

    if nproc == 1:
        info('count total size of elements')
        elem_size = count_size(target_file)
        info(f'Total {elem_size} elements in the target file : {target_file}')
        for _, elem in elem_iter:
            i += 1

            try:
                output += process_text(elem.text)
            except Exception as e:
                debug(f'Exception occured : {e}')
                traceback.print_exc()
                continue # it's okay as we have a pretty big corpus!
            elem.clear() # We need to save memory!

            progress  = round(i/elem_size*100, 2)
            debug(f'[{i}/{elem_size}]: {progress}% elapsed')

            if len(output) > max_corpus_size:
                break

    elif nproc > 1:
        size = 0
        p = Pool(nproc)
        results = []
        for _, elem in elem_iter:
            size += 1
            try:
                fout = p.apply_async(process_text, (elem.text,), callback=callback_func)
            except Exception as e:
                debug(f'Exception occured : {e}')
                traceback.print_exc()
                elem.clear()
                continue # it's okay as we have a pretty big corpus!
            elem.clear() # We need to save memory!

            results.append(fout)

        p.close()
        p.join()

        i = 0
        for r in results:
            i += 1
            line = r.get()
            output += line
            progress  = round(i/size*100, 2)
            debug(f'[{i}/{size}]: {progress}% elapsed')

    with codecs.open(output_file, 'w', 'utf-8') as fout:
        fout.write(output)


if __name__ == "__main__":
    build_corpus()
    
    info("Done")

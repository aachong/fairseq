import jionlp as jio
import jieba
import fastBPE
from mosestokenizer import *
import nltk.data
# from utils import *
from flashtext import KeywordProcessor
keyword_processor = KeywordProcessor()


def load_case_model(path):
    lines = jio.read_file_by_line(path)
    case_dict = {}
    for line in lines:
        content = line.split()
        if len(content) == 2:
            continue
        key, value = content[0].lower(), content[0]
        case_dict[key] = value
    return case_dict


def load_user_dict(path):
    lines = jio.read_file_by_line(path)
    user_dict = {
    }
    for line in lines:
        content = line.split('\t')
        user_dict[content[0]] = content[1]

    return user_dict


jieba.initialize()
base_dir = '/home/rcduan/fairseq/fairseq/examples/_transformer_big/data-bin'
zh_bpe = fastBPE.fastBPE(base_dir+'/codes.zh', base_dir+'/dict.zh.txt')
en_bpe = fastBPE.fastBPE(base_dir+'/codes.en', base_dir+'/dict.en.txt')
tokenizer = MosesTokenizer()
detokenizer = MosesDetokenizer()
punc_norm = MosesPunctuationNormalizer()
en_spliter = nltk.data.load('tokenizers/punkt/english.pickle')
case_model = base_dir+'/truecase-model.en'
user_dict_path = base_dir+'/user_dict'

case_dict = load_case_model(case_model)
user_dict = load_user_dict(user_dict_path)


def sep_lines(lines):
    sep = []
    for line in lines:
        sep.append(line.count('ENTER'))
    return sep


def en_tokenize(text):
    tokens = tokenizer(text)  # tokenize
    for i in range(len(tokens)):
        tokens[i] = case_dict.get(tokens[i].lower()) or tokens[i]
    return ' '.join(tokens)


def preprocess(texts, lang='zh'):
    assert lang in ['zh', 'en']
    results = []
    if lang == 'zh':
        for line in texts:
            text = ' '.join(list(jieba.cut(line)))  # 涓枃鍒嗚瘝
            text = ' '.join(tokenizer(text))    # tokenize
            text = ' '.join(zh_bpe.apply([text]))  # 涓枃BPE
            results.append(text)

    if lang == 'en':
        """鑻辨枃澶勭悊"""
        texts = jio.remove_exception_char(texts)  # 鍘婚櫎寮傚父瀛楃
        # 鍒嗚澶勭悊
        texts = texts.replace('ENTER', '\n').rstrip().replace('\n', ' ENTER ')
        lines = en_spliter.tokenize(texts)
        line_sep = sep_lines(lines)
        for line in lines:
            # normalize punctuation
            text = punc_norm(line).replace('ENTER', '')
            text = en_tokenize(text)    # tokenize
            text = ' '.join(en_bpe.apply([text]))  # 鑻辨枃BPE
            results.append(text)

    return results


def postprocess(texts, line_sep):

    outputs = []
    line = 0
    for text in texts:
        outputs.append(line_sep[line] * '\n' + detokenizer([text]))
        line += 1
    result = ' '.join(outputs)
    # 鐗规畩璇嶆浛鎹?TODO
    result = keyword_processor.replace_keywords(result)
    return result


if __name__ == '__main__':
    preprocess('你是不是对的', 'en')

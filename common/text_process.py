# coding=utf-8
import nltk
from common.config import START_TOKEN_IDX, END_TOKEN_IDX, PAD_TOKEN_IDX, START_TOKEN, END_TOKEN, PAD_TOKEN


def text_to_code(tokens, dictionary, seq_len):
    code_str = ""
    for sentence in tokens:
        index = 0
        for word in sentence:
            code_str += (str(dictionary[word]) + ' ')
            index += 1
        while index < seq_len:
            code_str += (str(PAD_TOKEN_IDX) + ' ')
            index += 1
        code_str += '\n'
    return code_str


def code_to_text(codes, dictionary, use_token=False):
    paras = ""
    for line in codes:
        numbers = map(int, line)
        for number in numbers:
            if use_token == False:
                if number == END_TOKEN_IDX:
                    break
                if number == START_TOKEN_IDX or number == PAD_TOKEN_IDX:
                    continue
            paras += (dictionary[str(number)] + ' ')
        paras += '\n'
    return paras


def get_tokenlized(file, is_text_file=True, add_end_token=True, max_seq_len=None):
    tokenlized = list()
    if is_text_file:
        with open(file) as raw:
            for text in raw:
                text = nltk.word_tokenize(text.lower())
                if max_seq_len is not None and len(text) > max_seq_len:
                    continue

                if add_end_token:
                    text.append(END_TOKEN)
                tokenlized.append(text)
    else:
        for text in file:
            text = nltk.word_tokenize(text.lower())
            if max_seq_len is not None and len(text) > max_seq_len:
                continue
            if add_end_token:
                text.append(END_TOKEN)
            tokenlized.append(text)            
        
    return tokenlized


def get_word_list(tokens):
    word_set = list()
    for sentence in tokens:
        for word in sentence:
            word_set.append(word)
            
    word_set = set(word_set)
    if START_TOKEN in word_set: 
        word_set.remove(START_TOKEN)
    if END_TOKEN in word_set: 
        word_set.remove(END_TOKEN)
    if PAD_TOKEN in word_set:
        word_set.remove(PAD_TOKEN)
    return list(word_set)


def get_dict(word_set):
    word_index_dict = dict()
    word_index_dict[START_TOKEN] = START_TOKEN_IDX
    word_index_dict[END_TOKEN] = END_TOKEN_IDX
    word_index_dict[PAD_TOKEN] = PAD_TOKEN_IDX
    
    index_word_dict = dict()
    index_word_dict[str(START_TOKEN_IDX)] = START_TOKEN
    index_word_dict[str(END_TOKEN_IDX)] = END_TOKEN
    index_word_dict[str(PAD_TOKEN_IDX)] = PAD_TOKEN
    index = 3
    for word in word_set:
        word_index_dict[word] = str(index)
        index_word_dict[str(index)] = word
        index += 1
    return word_index_dict, index_word_dict


def text_precess(train_text_loc, test_text_loc=None, max_seq_len=None):
    train_tokens = get_tokenlized(train_text_loc, max_seq_len=max_seq_len)
    if test_text_loc is None:
        test_tokens = list()
    else:
        test_tokens = get_tokenlized(test_text_loc, max_seq_len=max_seq_len)
    word_set = get_word_list(train_tokens + test_tokens)
    [word_index_dict, index_word_dict] = get_dict(word_set)

    if test_text_loc is None:
        sequence_len = len(max(train_tokens, key=len))
    else:
        sequence_len = max(len(max(train_tokens, key=len)), len(max(test_tokens, key=len)))

    return sequence_len, len(word_index_dict)

from datasets import load_dataset
import random
from transformers import AutoTokenizer
import pandas as pd
import re


def is_digit(text, ratio=0.5):
    length = max(len(text), 1)
    digit_num = sum([t in digits for t in text])
    if digit_num / length < ratio:
        return False
    return True


def is_valid(text):
    out = ''
    for char in text:
        if char in charset or char in digits:
            if ord(char) > 4000:
                out += char
    return out


def disturb(word, degree, p=0.3):
    if len(word) // 2 < degree:
        return word
    if is_digit(word):
        return word
    if random.random() < p:
        return word
    else:
        index = list(range(len(word)))
        random.shuffle(index)
        index = index[:degree]
        new_word = []
        for i in range(len(word)):
            if i not in index:
                new_word.append(word[i])
                continue
            if (word[i] not in charset) and (word[i] not in digits):
                # special token
                new_word.append(word[i])
                continue
            op = random.random()
            if op < 0.1:  # add
                new_word.append(random.choice(charset))
                new_word.append(word[i])
            elif op < 0.2:
                continue  # remove
            else:
                new_word.append(random.choice(charset))  # replace
        return ''.join(new_word)


def th_wiki(data, add_noise, wr):
    for title in data['title']:
        inp = is_valid(title)
        if len(inp) < min_length:
            continue
        wr.write(title)
        wr.write('\t')
        if add_noise:
            wr.write(disturb(inp, 1))
        else:
            wr.write(inp)
        wr.write('\n')
    for text in data['text']:
        tokens = tk(text.lower())['input_ids']
        for token in tokens:
            if token in tk.all_special_ids:
                continue
            dec = tk.decode(token)
            inp = is_valid(dec)
            if len(inp) < min_length:
                continue
            wr.write(dec)
            wr.write('\t')
            if add_noise:
                wr.write(disturb(inp, 1))
            else:
                wr.write(inp)
            wr.write('\n')


def en_wiki(data, add_noise, wr):
    for line in data:
        for text in line['text'].split():
            text = re.sub('[^0-9a-zA-Z]+', '', text)
            text = is_valid(text)
            if len(text) < min_length:
                continue
            wr.write(text)
            wr.write('\t')
            if add_noise:
                wr.write(disturb(text, 1))
            else:
                wr.write(text)
            wr.write('\n')


def opus(data, add_noise, wr):
    for line in data:
        for text in line['translation']['en'].split():
            text = re.sub('[^0-9a-zA-Z]+', '', text)
            text = is_valid(text)
            if len(text) < min_length:
                continue
            wr.write(text)
            wr.write('\t')
            if add_noise:
                wr.write(disturb(text, 1))
            else:
                wr.write(text)
            wr.write('\n')
        tokens = tk(line['translation']['th'].lower())['input_ids']
        for token in tokens:
            if token in tk.all_special_ids:
                continue
            dec = tk.decode(token)
            inp = is_valid(dec)
            if len(inp) < min_length:
                continue
            wr.write(inp)
            wr.write('\t')
            if add_noise:
                wr.write(disturb(inp, 1))
            else:
                wr.write(inp)
            wr.write('\n')


def removes():
    a = open('train.txt').readlines()
    with open('train2.txt', 'w') as wr:
        for line in a:
            for char in line:
                if not ord(char) > 4000:
                    wr.write(char)
    a = open('val.txt').readlines()
    with open('val2.txt', 'w') as wr:
        for line in a:
            for char in line:
                if not ord(char) > 4000:
                    wr.write(char)


if __name__ == '__main__':
    with open('train.txt', 'w') as wr:
        wr.write('inp\tgt\n')
    with open('val.txt', 'w') as wr:
        wr.write('inp\tgt\n')
    min_length = 3

    tk = AutoTokenizer.from_pretrained('airesearch/wangchanberta-base-att-spm-uncased')
    charmap = open('/home/palm/PycharmProjects/ABINet/data/charset_enth.txt').read().split('\n')
    charset = [x.split('	')[1] for x in charmap[39:165]]
    digits = [x.split('	')[1] for x in charmap if x.split('	')[1] not in charset]

    # train_inp = []
    # train_gt = []
    # val_inp = []
    # val_gt = []
    th_data = load_dataset('graelo/wikipedia', '20230601.th')['train']
    with open('train.txt', 'a') as wr:
        th_wiki(th_data[20000:], False, wr)
    with open('val.txt', 'a') as wr:
        th_wiki(th_data[:20000], True, wr)
    en_data = load_dataset('wikitext', 'wikitext-103-v1')
    with open('train.txt', 'a') as wr:
        en_wiki(en_data['train'], False, wr)
    with open('val.txt', 'a') as wr:
        en_wiki(en_data['validation'], True, wr)
        en_wiki(en_data['test'], True, wr)
    opus_data = load_dataset('opus100', 'en-th')
    with open('train.txt', 'a') as wr:
        opus(opus_data['train'], False, wr)
    with open('val.txt', 'a') as wr:
        opus(opus_data['validation'], True, wr)
        opus(opus_data['test'], True, wr)
    # train_inp.append(inp)
    # train_gt.append(gt)
    # inp, gt = en_wiki(en_data['validation'], False)
    # val_inp.append(inp)
    # val_gt.append(gt)
    # inp, gt = en_wiki(en_data['test'], False)
    # val_inp.append(inp)
    # val_gt.append(gt)
    # opus_data = load_dataset('opus100', 'en-th')
    # inp, gt = en_wiki(opus_data['train'], False)
    # train_inp.append(inp)
    # train_gt.append(gt)
    # inp, gt = en_wiki(opus_data['validation'], False)
    # val_inp.append(inp)
    # val_gt.append(gt)
    # inp, gt = en_wiki(opus_data['test'], False)
    # val_inp.append(inp)
    # val_gt.append(gt)
    # pd.DataFrame({'inp': train_inp, 'gt': train_gt}).to_csv('train.txt', index=None, sep='\t')
    # pd.DataFrame({'inp': val_inp, 'gt': val_gt}).to_csv('val.txt', index=None, sep='\t')

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


def th_wiki(data, add_noise):
    inp = []
    gt = []
    for title in data['title']:
        if len(title) < min_length:
            continue
        inp.append(title)
        if add_noise:
            gt.append(disturb(inp[-1], 1))
        else:
            gt.append(inp[-1])
    for text in data['text']:
        tokens = tk(text.lower())['input_ids']
        for token in tokens:
            if token in tk.all_special_ids:
                continue
            dec = tk.decode(token)
            if len(dec) < min_length:
                continue
            inp.append(dec)
            if add_noise:
                gt.append(disturb(inp[-1], 1))
            else:
                gt.append(inp[-1])
    return inp, gt


def en_wiki(data, add_noise):
    inp = []
    gt = []
    for line in data:
        for text in line['text'].split():
            text = re.sub('[^0-9a-zA-Z]+', '', text)
            if len(text) < min_length:
                continue
            inp.append(text)
            if add_noise:
                gt.append(disturb(text, 1))
            else:
                gt.append(text)
    return inp, gt


def opus(data, add_noise):
    inp = []
    gt = []
    for line in data:
        for text in line['translation']['en'].split():
            text = re.sub('[^0-9a-zA-Z]+', '', text)
            if len(text) < min_length:
                continue
            inp.append(text)
            if add_noise:
                gt.append(disturb(text, 1))
            else:
                gt.append(text)
        tokens = tk(line['translation']['th'].lower())['input_ids']
        for token in tokens:
            if token in tk.all_special_ids:
                continue
            dec = tk.decode(token)
            if len(dec) < min_length:
                continue
            inp.append(dec)
            if add_noise:
                gt.append(disturb(inp[-1], 1))
            else:
                gt.append(inp[-1])
    return inp, gt


if __name__ == '__main__':
    min_length = 2

    tk = AutoTokenizer.from_pretrained('airesearch/wangchanberta-base-att-spm-uncased')
    charmap = open('/home/palm/PycharmProjects/ABINet/data/charset_enth.txt').read().split('\n')
    charset = [x.split('	')[1] for x in charmap[39:165]]
    digits = [x.split('	')[1] for x in charmap if x.split('	')[1] not in charset]
    train_inp = []
    train_gt = []
    val_inp = []
    val_gt = []
    th_data = load_dataset('graelo/wikipedia', '20230601.th')['train']
    inp, gt = th_wiki(th_data[20000:], False)
    train_inp.append(inp)
    train_gt.append(gt)
    inp, gt = th_wiki(th_data[:20000], True)
    val_inp.append(inp)
    val_gt.append(gt)
    en_data = load_dataset('wikitext', 'wikitext-103-v1')
    inp, gt = en_wiki(en_data['train'], False)
    train_inp.append(inp)
    train_gt.append(gt)
    inp, gt = en_wiki(en_data['validation'], False)
    val_inp.append(inp)
    val_gt.append(gt)
    inp, gt = en_wiki(en_data['test'], False)
    val_inp.append(inp)
    val_gt.append(gt)
    opus_data = load_dataset('opus100', 'en-th')
    inp, gt = en_wiki(opus_data['train'], False)
    train_inp.append(inp)
    train_gt.append(gt)
    inp, gt = en_wiki(opus_data['validation'], False)
    val_inp.append(inp)
    val_gt.append(gt)
    inp, gt = en_wiki(opus_data['test'], False)
    val_inp.append(inp)
    val_gt.append(gt)
    pd.DataFrame({'inp': train_inp, 'gt': train_gt}).to_csv('train.txt', index=None, sep='\t')
    pd.DataFrame({'inp': val_inp, 'gt': val_gt}).to_csv('val.txt', index=None, sep='\t')

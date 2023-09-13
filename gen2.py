import numpy as np
from PIL import Image, ImageFont, ImageDraw
import glob
import cv2
import albumentations as albu
from gen_img import rotate_bound
import pandas as pd
import json
import os


font_list = glob.glob('ttf/*.ttf')


def generate(
        text,
        font,
        font_size,
        color,  # 50 - 150
):
    # random
    font = ImageFont.truetype(font, font_size)  # load font

    image = Image.new("L", (500, 500), "white")
    draw = ImageDraw.Draw(image)

    # get the size of the text
    # text_size = draw.textsize(text, font)
    bbox = draw.textbbox(xy=(0, 0), text=text, font=font)
    if bbox[2] <= 0 or bbox[3] <= 0:
        print("Error (%s)" % text)
        return None, None

    # resize and draw
    image = image.resize((bbox[2], bbox[3]))
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), text, color, font=font)
    image = np.array(image)
    w = np.where(image.min(0) < 255)[0]
    image = image[:, w[0]:w[-1]]
    offset_l = np.random.randint(10, 20)
    offset_t = np.random.randint(10, 20)
    results = np.ones((image.shape[0]+offset_t*2, image.shape[1]+offset_l*2), dtype='uint8') * 255
    results[offset_t:-offset_t, offset_l:-offset_l] = image
    return results, text


def is_eng(row):
    text = str(row['inp'])
    return ord('A') <= ord(text[0]) <= ord('z')


def is_short(text):
    return len(str(text)) < 10


n = 2
p = 0.5
size_ratio = (0.8, 1.2)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
noise = albu.MultiplicativeNoise(multiplier=[0.9, 1.1], elementwise=True, per_channel=True, p=p)
blur = albu.OneOf([albu.AdvancedBlur(p=p), albu.GaussianBlur(p=p)])
def aug(image):
    if np.random.rand() < p:
        mask = 1-cv2.morphologyEx((image < 255).astype('uint8'), cv2.MORPH_ERODE, kernel)
        image = ((mask-1) * (1-image))-1
    image = rotate_bound(image)
    ori_size = image.shape[1], image.shape[0]
    ratio = np.random.uniform(*size_ratio)
    image = cv2.resize(image, None, None, ratio, ratio)
    image = cv2.resize(image, ori_size)
    image = noise(image=image, )['image']
    image = blur(image=image, )['image']
    return image


def get_data(path):
    data = pd.read_csv(path, delimiter='\t')
    english_idx = data.apply(is_eng, axis=1)
    english = data['inp'][english_idx]
    short_english = english.apply(is_short)
    short_english = english[short_english]
    thai = data['inp'][~english_idx]
    short_thai = thai.apply(is_short)
    short_thai = thai[short_thai]
    return english, short_english, thai, short_thai


def aug_text(english, short_english, thai, short_thai):
    for text in english:
        added = np.random.choice(short_thai)
        number = np.random.randint(9999)
        if np.random.rand() < 0.4:
            text = f'{text} {number} {added}'
        elif np.random.rand() < 0.5:
            text = f'{number} {text} {added}'
        else:
            text = f'{added} {number} {text}'
        yield text
    for text in thai:
        added = np.random.choice(short_english)
        number = np.random.randint(9999)
        if len(text) > 30:
            pass
        elif np.random.rand() < 0.4:
            text = f'{text} {added} {number}'
        elif np.random.rand() < 0.5:
            text = f'{added} {text} {number}'
        else:
            text = f'{number} {added} {text}'
        yield text


if __name__ == '__main__':
    s = 'val'
    path = '/media/palm/Data/ocr/data3'
    os.makedirs(os.path.join(path, f'images/val'))
    os.makedirs(os.path.join(path, f'images/train'))
    english, short_english, thai, short_thai = get_data(f'../ABINet/val.txt')
    wr = open(os.path.join(path, 'val.jsonl'), 'w')
    for idx, text in enumerate(aug_text(english, short_english, thai, short_thai)):
        image, _ = generate(text, np.random.choice(font_list), 40, int(np.random.rand()*150))
        augd_image = aug(image)
        if idx == 100000:
            s = 'train'
            wr = open(os.path.join(path, 'train.jsonl'), 'w')
        elif idx >= 2100000:
            break
        filename = f'images/{s}/{idx}.jpg'
        wr.write(json.dumps({'filename': filename, 'text': text}))
        wr.write('\n')
        cv2.imwrite(os.path.join(path, filename), augd_image)

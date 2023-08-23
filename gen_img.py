import numpy as np
from PIL import Image, ImageFont, ImageDraw
import glob
import cv2
import re
import albumentations as A
import argparse
import glob
from tqdm import tqdm
from multiprocessing import Pool, Value, Array


def deemojify(text):
    "function to remove emojis from text"
    regrex_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U00002500-\U00002BEF"  # chinese char
                                u"\U00002702-\U000027B0"
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                u"\U0001f926-\U0001f937"
                                u"\U00010000-\U0010ffff"
                                u"\u2640-\u2642"
                                u"\u2600-\u2B55"
                                u"\u200d"
                                u"\u23cf"
                                u"\u23e9"
                                u"\u231a"
                                u"\ufe0f"  # dingbats
                                u"\u3030"
                                "\\\\"
                                '\"'
                                "]+", re.UNICODE)
    return regrex_pattern.sub(r'', text)


def create_text_generator_mp(m, txtdir):
    filelist = glob.glob('%s/*.txt' % txtdir)
    for j, f in enumerate(filelist):
        file = open(f, "r")
        for i, line in enumerate(file):
            if i % 8 != m:
                continue
            line = deemojify(line.rstrip('\n'))
            for k in range(0, len(line), 100):  # skip 150 per chunk
                # random length between 10-150
                l = 10 + np.random.randint(90)
                yield line[k:k + l]


def mp_save(m, txtdir, max_num):
    j = 0
    out_text = ''
    for text_org in create_text_generator_mp(m, txtdir):
        img, text = random_example(text_org, font_list)
        if img is None:
            continue
        filename = f"{args.out}/images/%d_%.6d.jpg" % (m, j)
        j += 1
        img = rotate_bound(img)
        img = sepia(img)
        img = noise(img)
        cv2.imwrite(filename, img)
        out_text += "{\"filename\": \"%s\", \"text\": \"%s\"}\n" % (filename, text)
        if j >= max_num:
            return out_text
    return out_text


def random_example(
        text,
        font_list,
        font_size_min=14, font_size_max=40,
        max_height=50, max_width=1200,
):
    # random
    font_idx = np.random.randint(len(font_list))
    font_size = np.random.randint(font_size_min, high=font_size_max)
    x = np.random.randint(0, 100)
    for _ in range(len(text)):
        font = ImageFont.truetype(font_list[font_idx], font_size)  # load font

        image = Image.new("L", (500, 500), "white")
        draw = ImageDraw.Draw(image)

        # get the size of the text
        # text_size = draw.textsize(text, font)
        bbox = draw.textbbox(xy=(0+x, 0), text=text, font=font)
        if bbox[2] <= 0 or bbox[3] <= 0:
            print("Error (%s)" % text)
            return None, None

        # resize and draw
        image = image.resize((bbox[2], bbox[3]))
        draw = ImageDraw.Draw(image)
        draw.text((x, 0), text, 0, font=font)

        w, h = image.size
        H = max_height
        W = int(max_height * w / h)
        if W <= max_width:
            image = image.resize((W, H))
            break
        else:  # too long -> reduce font ize & shorten text
            font_size -= 1
            if font_size < font_size_min:
                font_size = font_size_min
            text = text[:-1]

    result = Image.new("L", (max_width, max_height), "white")
    result.paste(image, (0, 0))

    return np.asarray(result), text


def sepia(image):
    paper3 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    white = paper3 > 0
    paper3[white] = (paper3[white] * (np.random.rand()/2+0.5)).astype('uint8')
    paper3 = sepia_(image=paper3)['image'][..., ::-1]
    # paper3 = color_(image=paper3)['image']
    return paper3


def noise(image):
    size = np.random.rand() + 0.8
    image = cv2.resize(image, None, None, size, size)
    paper4 = noise_(image=image)['image']
    # paper4 = cv2.resize(paper4, (210 * 2, 297 * 2))
    return paper4


def rotate_bound(image):
    angle = np.random.randint(-2, 2)
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH), borderValue=255)


if __name__ == "__main__":
    sepia_ = A.ToSepia(p=0.5)
    noise_ = A.MultiplicativeNoise(multiplier=[0.9, 1.1], elementwise=True, per_channel=True, p=1)

    parser = argparse.ArgumentParser()
    parser.add_argument('txt', type=str, help="Text folder")
    parser.add_argument('ttf', type=str, help="Font folder")
    parser.add_argument('out', type=str, help="Out folder")
    parser.add_argument('num_workers', type=int, help="Num Workers")
    args = parser.parse_args()

    font_list = glob.glob('%s/*.ttf' % args.ttf)
    array = []
    for i in range(args.num_workers):
        array.append((i, args.txt, 1000000 // args.num_workers))
    with open(f"{args.out}/annotations", "w") as file:
        j = 0
        with Pool() as pool:
            res = pool.starmap(mp_save, array)
            for re in res:
                file.write(re)

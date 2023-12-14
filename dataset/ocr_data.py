import warnings

warnings.filterwarnings('ignore')
import os
from torch.utils.data import Dataset
from dataset.transforms import CVColorJitter, CVDeterioration, CVGeometry
from torchvision import transforms
import cv2
import math
import random
from PIL import Image
import numpy as np
import json
import pandas as pd
import glob
import re


class ImageDataset(Dataset):
    def __init__(self,
                 src: str,
                 jsonl: str,
                 is_training: bool = True,
                 img_h: int = 32,
                 img_w: int = 100,
                 data_aug: bool = True,
                 multiscales: bool = True,
                 convert_mode: str = 'RGB',
                 single_jsonl=False,
                 ):
        self.data_aug = data_aug
        self.convert_mode = convert_mode
        self.img_h = img_h
        self.img_w = img_w
        self.multiscales = multiscales
        self.is_training = is_training
        self.src = src
        if not os.path.exists(jsonl):
            self.data = ['{"filename": "./dataset/images/000000.png", "text": "4"}'] * 100
        elif single_jsonl:
            self.data = []
            kw = 'train' if is_training else 'val'
            for data in open(jsonl).read().split('\n')[:-1]:
                d = json.loads(data)
                if kw in d['filename']:
                    self.data.append(data)
        else:
            print(jsonl)
            self.data = open(jsonl).read().split('\n')[:-1]
        if self.data_aug:
            self.augment_tfs = transforms.Compose([
                CVGeometry(degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.), shear=(45, 15), distortion=0.5, p=0.5),
                CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
                CVColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25)
            ])
        self.totensor = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def resize_multiscales(self, img, borderType=cv2.BORDER_CONSTANT):
        def _resize_ratio(img, ratio, fix_h=True):
            if ratio * self.img_w < self.img_h:
                if fix_h:
                    trg_h = self.img_h
                else:
                    trg_h = int(ratio * self.img_w)
                trg_w = self.img_w
            else:
                trg_h, trg_w = self.img_h, int(self.img_h / ratio)
            img = cv2.resize(img, (trg_w, trg_h))
            pad_h, pad_w = (self.img_h - trg_h) / 2, (self.img_w - trg_w) / 2
            top, bottom = math.ceil(pad_h), math.floor(pad_h)
            left, right = math.ceil(pad_w), math.floor(pad_w)
            img = cv2.copyMakeBorder(img, top, bottom, left, right, borderType)
            return img

        if self.is_training:
            if random.random() < 0.5:
                base, maxh, maxw = self.img_h, self.img_h, self.img_w
                h, w = random.randint(base, maxh), random.randint(base, maxw)
                return _resize_ratio(img, h / w)
            else:
                return _resize_ratio(img, img.shape[0] / img.shape[1])  # keep aspect ratio
        else:
            return _resize_ratio(img, img.shape[0] / img.shape[1])  # keep aspect ratio

    def resize(self, img):
        if self.multiscales:
            return self.resize_multiscales(img, cv2.BORDER_REPLICATE)
        else:
            return cv2.resize(img, (self.img_w, self.img_h))

    def get(self, idx):
        data = json.loads(self.data[idx])
        im = data['filename']
        label = data['text']
        # image = Image.open(os.path.join(self.f, im)).convert(self.convert_mode)
        return os.path.join(self.src, im), label, idx

    def _process_training(self, image):
        if self.data_aug:
            image = self.augment_tfs(image)
        image = self.resize(np.array(image))
        return image

    def _process_test(self, image):
        return self.resize(np.array(image))

    def __getitem__(self, idx):
        image, text, idx_new = self.get(idx)
        return image, text


class SynthDataset(Dataset):
    def __init__(self, 
                 tk,
                 mji='/home/palm/data/mjsynth/images/*/*/*.jpg', 
                 stl='/home/palm/data/synthtext/labels.tsv',
                 sti='/home/palm/data/synthtext/cropped',
                 json_mode=None,
                 ):
        if json_mode is not None:
            self.data = json.load(open(json_mode))
        else:
            self.data = []
            stl = pd.read_csv(stl, header=None,  delimiter='\t')
            errors = '4659,"nan"20172,"nan"62493,"nan"70203,"nan"71259,"nan"92683,"nan"110054,"nan"145208,"nan"153744,"nan"210836,"nan"257603,"nan"276227,"nan"304114,"nan"312424,"nan"315707,"nan"358627,"nan"359874,"nan"364185,"nan"377069,"nan"385822,"nan"394234,"nan"401732,"nan"423674,"nan"430230,"nan"437792,"nan"476168,"nan"477352,"nan"484597,"nan"490463,"nan"511820,"nan"533748,"nan"546232,"nan"553056,"nan"578149,"nan"578489,"nan"588944,"nan"611425,"nan"636797,"nan"660472,"nan"687075,"nan"721142,"nan"766734,"nan"770801,"nan"779541,"nan"801070,"nan"802337,"nan"842815,"nan"909277,"nan"921841,"nan"925933,"nan"957805,"nan"998154,"nan"1035867,"nan"1037128,"nan"1044326,"nan"1054010,"nan"1057888,"nan"1080163,"nan"1096132,"nan"1101509,"nan"1118959,"nan"1120800,"nan"1133489,"nan"1140936,"nan"1148881,"nan"1181970,"nan"1216142,"nan"1245823,"nan"1257087,"nan"1276395,"nan"1281007,"nan"1288002,"nan"1304954,"nan"1318860,"nan"1324836,"nan"1327214,"nan"1349478,"nan"1353106,"nan"1378554,"nan"1383308,"nan"1386275,"nan"1399187,"nan"1403045,"nan"1407229,"nan"1424008,"nan"1457083,"nan"1461816,"nan"1474878,"nan"1486091,"nan"1501963,"nan"1553441,"nan"1571924,"nan"1595262,"nan"1601109,"nan"1603231,"nan"1611702,"nan"1627304,"nan"1637932,"nan"1643470,"nan"1663813,"nan"1690286,"nan"1697671,"nan"1710341,"nan"1714512,"nan"1732012,"nan"1739218,"nan"1744075,"nan"1744646,"nan"1762205,"nan"1762698,"nan"1783065,"nan"1796509,"nan"1816064,"nan"1819649,"nan"1842237,"nan"1853283,"nan"1878511,"nan"1895495,"nan"1927202,"nan"1930369,"nan"1948069,"nan"1955255,"nan"1964561,"nan"1973704,"nan"2012652,"nan"2043227,"nan"2050716,"nan"2070611,"nan"2100777,"nan"2110166,"nan"2126030,"nan"2127498,"nan"2129287,"nan"2130237,"nan"2135515,"nan"2137811,"nan"2141847,"nan"2144585,"nan"2145150,"nan"2148162,"nan"2190242,"nan"2218829,"nan"2221153,"nan"2243794,"nan"2256448,"nan"2305537,"nan"2349420,"nan"2379164,"nan"2381386,"nan"2405164,"nan"2405900,"nan"2437449,"nan"2458339,"nan"2487585,"nan"2504885,"nan"2506148,"nan"2510512,"nan"2531954,"nan"2544040,"nan"2591974,"nan"2605532,"nan"2616230,"nan"2652531,"nan"2660312,"nan"2682647,"nan"2694384,"nan"2703697,"nan"2719918,"nan"2739782,"nan"2771711,"nan"2791304,"nan"2792187,"nan"2797744,"nan"2809418,"nan"2812712,"nan"2818542,"nan"2827944,"nan"2831480,"nan"2836419,"nan"2840547,"nan"2860130,"nan"2860969,"nan"2906947,"nan"2940763,"nan"2953079,"nan"2967484,"nan"2968441,"nan"3014989,"nan"3078421,"nan"3089502,"nan"3099509,"nan"3123446,"nan"3123717,"nan"3126858,"nan"3160780,"nan"3198684,"nan"3223102,"nan"3243210,"nan"3249472,"nan"3253442,"nan"3287283,"nan"3294281,"nan"3309159,"nan"3340361,"nan"3371557,"nan"3416533,"nan"3437699,"nan"3462010,"nan"3462964,"nan"3463323,"nan"3478176,"nan"3496425,"nan"3498130,"nan"3510442,"nan"3540543,"nan"3561507,"nan"3574024,"nan"3592682,"nan"3601164,"nan"3622886,"nan"3634222,"nan"3645716,"nan"3657562,"nan"3663016,"nan"3663430,"nan"3669905,"nan"3722227,"nan"3728056,"nan"3731738,"nan"3744046,"nan"3745097,"nan"3751104,"nan"3758926,"nan"3786622,"nan"3796857,"nan"3812103,"nan"3825879,"nan"3833211,"nan"3852397,"nan"3868365,"nan"3887771,"nan"3896984,"nan"3902241,"nan"3911682,"nan"3917733,"nan"3931528,"nan"3932126,"nan"3949509,"nan"3957384,"nan"3963814,"nan"3980791,"nan"3995855,"nan"3997009,"nan"4014006,"nan"4014670,"nan"4032540,"nan"4035731,"nan"4036270,"nan"4048176,"nan"4069298,"nan"4069583,"nan"4089974,"nan"4106711,"nan"4111724,"nan"4116728,"nan"4168593,"nan"4172933,"nan"4213402,"nan"4216831,"nan"4239369,"nan"4246716,"nan"4257519,"nan"4270052,"nan"4278368,"nan"4298463,"nan"4308422,"nan"4361547,"nan"4361761,"nan"4383838,"nan"4388432,"nan"4389622,"nan"4417903,"nan"4426732,"nan"4430207,"nan"4455435,"nan"4457378,"nan"4486479,"nan"4515776,"nan"4520590,"nan"4536313,"nan"4565735,"nan"4597794,"nan"4601311,"nan"4638534,"nan"4685934,"nan"4691526,"nan"4693286,"nan"4714088,"nan"4720364,"nan"4766617,"nan"4816300,"nan"4867409,"nan"4869818,"nan"4883224,"nan"4887093,"nan"4888009,"nan"4891604,"nan"4893044,"nan"4937599,"nan"4950305,"nan"4990071,"nan"4998828,"nan"5006716,"nan"5006736,"nan"5038639,"nan"5047621,"nan"5048630,"nan"5061745,"nan"5139680,"nan"5140976,"nan"5152028,"nan"5159543,"nan"5212392,"nan"5227070,"nan"5233607,"nan"5243131,"nan"5271651,"nan"5303179,"nan"5311773,"nan"5337576,"nan"5351418,"nan"5398016,"nan"5423995,"nan"5435794,"nan"5437604,"nan"5454697,"nan"5464644,"nan"5470526,"nan"5479692,"nan"5482000,"nan"5489234,"nan"5541511,"nan"5555905,"nan"5572757,"nan"5577497,"nan"5578191,"nan"5580113,"nan"5581685,"nan"5599282,"nan"5608291,"nan"5646381,"nan"5646595,"nan"5659018,"nan"5664755,"nan"5665558,"nan"5678728,"nan"5686867,"nan"5688438,"nan"5709721,"nan"5711327,"nan"5723249,"nan"5758872,"nan"5759831,"nan"5772342,"nan"5780761,"nan"5790714,"nan"5798589,"nan"5833622,"nan"5835043,"nan"5876015,"nan"5876318,"nan"5887093,"nan"5892579,"nan"5906090,"nan"5913263,"nan"5932867,"nan"5938896,"nan"5957724,"nan"5960904,"nan"5963678,"nan"5968184,"nan"5993869,"nan"6046791,"nan"6070870,"nan"6072564,"nan"6075522,"nan"6120505,"nan"6130231,"nan"6152190,"nan"6161910,"nan"6210133,"nan"6227410,"nan"6268931,"nan"6315184,"nan"6334463,"nan"6337785,"nan"6342162,"nan"6346493,"nan"6350406,"nan"6417149,"nan"6428730,"nan"6435446,"nan"6469754,"nan"6497547,"nan"6528338,"nan"6551308,"nan"6555138,"nan"6641038,"nan"6660848,"nan"6687713,"nan"6688412,"nan"6700829,"nan"6715173,"nan"6719266,"nan"6738156,"nan"6741583,"nan"6751525,"nan"6769971,"nan"6841537,"nan"6876628,"nan"6885473,"nan"6935486,"nan"6948850,"nan"6952485,"nan"6986340,"nan"6992063,"nan"7029698,"nan"7052845,"nan"7056616,"nan"7073210,"nan"7088726,"nan"7104764,"nan"7132195,"nan"7186431,"nan"7191106,"nan"'
            bad_indice = re.findall(r'[0-9]+', errors)
            for idx, row in stl.iterrows():
                if str(idx) in bad_indice:
                    continue
                if os.path.exists(os.path.join(sti, row[0])):
                    self.data.append(
                        (os.path.join(sti, row[0]), row[1])
                    )
            for file in glob.glob(mji):
                if os.path.exists(file):
                    label = os.path.split(file)[1].split('_')[1]

                    self.data.append((                
                        file,
                        label
                    ))

    def __getitem__(self, idx):
        image, text = self.data[idx]
        return image, text

    def __len__(self):
        return len(self.data)


class FunsdDataset(SynthDataset):
    def __init__(self,
                 tk,
                 labels=('/home/palm/data/funsd/train_label.json', '/home/palm/data/funsd/test_label.json'),
                 folders=('training', 'test'),
                 images='/home/palm/data/funsd/crops',
                 ):
        self.data = []
        for label, folder in zip(labels, folders):
            for instance in json.load(open(label))['data_list']:
                if os.path.exists(os.path.join(images, folder, instance['img_path'])):
                    self.data.append(
                        (os.path.join(images, folder, instance['img_path']), instance['instances'][0]['text'])
                    )


if __name__ == '__main__':
    mji='/project/lt200060-capgen/palm/ocr_data/train/mjsynth/mnt/ramdisk/max/90kDICT32px/*/*/*.jpg' 
    stl='/project/lt200060-capgen/palm/ocr_data/train/synthtext/labels.tsv'
    sti='/project/lt200060-capgen/palm/ocr_data/train/synthtext/crop'
    stl = pd.read_csv(stl, header=None,  delimiter='\t')
    errors = '4659,"nan"20172,"nan"62493,"nan"70203,"nan"71259,"nan"92683,"nan"110054,"nan"145208,"nan"153744,"nan"210836,"nan"257603,"nan"276227,"nan"304114,"nan"312424,"nan"315707,"nan"358627,"nan"359874,"nan"364185,"nan"377069,"nan"385822,"nan"394234,"nan"401732,"nan"423674,"nan"430230,"nan"437792,"nan"476168,"nan"477352,"nan"484597,"nan"490463,"nan"511820,"nan"533748,"nan"546232,"nan"553056,"nan"578149,"nan"578489,"nan"588944,"nan"611425,"nan"636797,"nan"660472,"nan"687075,"nan"721142,"nan"766734,"nan"770801,"nan"779541,"nan"801070,"nan"802337,"nan"842815,"nan"909277,"nan"921841,"nan"925933,"nan"957805,"nan"998154,"nan"1035867,"nan"1037128,"nan"1044326,"nan"1054010,"nan"1057888,"nan"1080163,"nan"1096132,"nan"1101509,"nan"1118959,"nan"1120800,"nan"1133489,"nan"1140936,"nan"1148881,"nan"1181970,"nan"1216142,"nan"1245823,"nan"1257087,"nan"1276395,"nan"1281007,"nan"1288002,"nan"1304954,"nan"1318860,"nan"1324836,"nan"1327214,"nan"1349478,"nan"1353106,"nan"1378554,"nan"1383308,"nan"1386275,"nan"1399187,"nan"1403045,"nan"1407229,"nan"1424008,"nan"1457083,"nan"1461816,"nan"1474878,"nan"1486091,"nan"1501963,"nan"1553441,"nan"1571924,"nan"1595262,"nan"1601109,"nan"1603231,"nan"1611702,"nan"1627304,"nan"1637932,"nan"1643470,"nan"1663813,"nan"1690286,"nan"1697671,"nan"1710341,"nan"1714512,"nan"1732012,"nan"1739218,"nan"1744075,"nan"1744646,"nan"1762205,"nan"1762698,"nan"1783065,"nan"1796509,"nan"1816064,"nan"1819649,"nan"1842237,"nan"1853283,"nan"1878511,"nan"1895495,"nan"1927202,"nan"1930369,"nan"1948069,"nan"1955255,"nan"1964561,"nan"1973704,"nan"2012652,"nan"2043227,"nan"2050716,"nan"2070611,"nan"2100777,"nan"2110166,"nan"2126030,"nan"2127498,"nan"2129287,"nan"2130237,"nan"2135515,"nan"2137811,"nan"2141847,"nan"2144585,"nan"2145150,"nan"2148162,"nan"2190242,"nan"2218829,"nan"2221153,"nan"2243794,"nan"2256448,"nan"2305537,"nan"2349420,"nan"2379164,"nan"2381386,"nan"2405164,"nan"2405900,"nan"2437449,"nan"2458339,"nan"2487585,"nan"2504885,"nan"2506148,"nan"2510512,"nan"2531954,"nan"2544040,"nan"2591974,"nan"2605532,"nan"2616230,"nan"2652531,"nan"2660312,"nan"2682647,"nan"2694384,"nan"2703697,"nan"2719918,"nan"2739782,"nan"2771711,"nan"2791304,"nan"2792187,"nan"2797744,"nan"2809418,"nan"2812712,"nan"2818542,"nan"2827944,"nan"2831480,"nan"2836419,"nan"2840547,"nan"2860130,"nan"2860969,"nan"2906947,"nan"2940763,"nan"2953079,"nan"2967484,"nan"2968441,"nan"3014989,"nan"3078421,"nan"3089502,"nan"3099509,"nan"3123446,"nan"3123717,"nan"3126858,"nan"3160780,"nan"3198684,"nan"3223102,"nan"3243210,"nan"3249472,"nan"3253442,"nan"3287283,"nan"3294281,"nan"3309159,"nan"3340361,"nan"3371557,"nan"3416533,"nan"3437699,"nan"3462010,"nan"3462964,"nan"3463323,"nan"3478176,"nan"3496425,"nan"3498130,"nan"3510442,"nan"3540543,"nan"3561507,"nan"3574024,"nan"3592682,"nan"3601164,"nan"3622886,"nan"3634222,"nan"3645716,"nan"3657562,"nan"3663016,"nan"3663430,"nan"3669905,"nan"3722227,"nan"3728056,"nan"3731738,"nan"3744046,"nan"3745097,"nan"3751104,"nan"3758926,"nan"3786622,"nan"3796857,"nan"3812103,"nan"3825879,"nan"3833211,"nan"3852397,"nan"3868365,"nan"3887771,"nan"3896984,"nan"3902241,"nan"3911682,"nan"3917733,"nan"3931528,"nan"3932126,"nan"3949509,"nan"3957384,"nan"3963814,"nan"3980791,"nan"3995855,"nan"3997009,"nan"4014006,"nan"4014670,"nan"4032540,"nan"4035731,"nan"4036270,"nan"4048176,"nan"4069298,"nan"4069583,"nan"4089974,"nan"4106711,"nan"4111724,"nan"4116728,"nan"4168593,"nan"4172933,"nan"4213402,"nan"4216831,"nan"4239369,"nan"4246716,"nan"4257519,"nan"4270052,"nan"4278368,"nan"4298463,"nan"4308422,"nan"4361547,"nan"4361761,"nan"4383838,"nan"4388432,"nan"4389622,"nan"4417903,"nan"4426732,"nan"4430207,"nan"4455435,"nan"4457378,"nan"4486479,"nan"4515776,"nan"4520590,"nan"4536313,"nan"4565735,"nan"4597794,"nan"4601311,"nan"4638534,"nan"4685934,"nan"4691526,"nan"4693286,"nan"4714088,"nan"4720364,"nan"4766617,"nan"4816300,"nan"4867409,"nan"4869818,"nan"4883224,"nan"4887093,"nan"4888009,"nan"4891604,"nan"4893044,"nan"4937599,"nan"4950305,"nan"4990071,"nan"4998828,"nan"5006716,"nan"5006736,"nan"5038639,"nan"5047621,"nan"5048630,"nan"5061745,"nan"5139680,"nan"5140976,"nan"5152028,"nan"5159543,"nan"5212392,"nan"5227070,"nan"5233607,"nan"5243131,"nan"5271651,"nan"5303179,"nan"5311773,"nan"5337576,"nan"5351418,"nan"5398016,"nan"5423995,"nan"5435794,"nan"5437604,"nan"5454697,"nan"5464644,"nan"5470526,"nan"5479692,"nan"5482000,"nan"5489234,"nan"5541511,"nan"5555905,"nan"5572757,"nan"5577497,"nan"5578191,"nan"5580113,"nan"5581685,"nan"5599282,"nan"5608291,"nan"5646381,"nan"5646595,"nan"5659018,"nan"5664755,"nan"5665558,"nan"5678728,"nan"5686867,"nan"5688438,"nan"5709721,"nan"5711327,"nan"5723249,"nan"5758872,"nan"5759831,"nan"5772342,"nan"5780761,"nan"5790714,"nan"5798589,"nan"5833622,"nan"5835043,"nan"5876015,"nan"5876318,"nan"5887093,"nan"5892579,"nan"5906090,"nan"5913263,"nan"5932867,"nan"5938896,"nan"5957724,"nan"5960904,"nan"5963678,"nan"5968184,"nan"5993869,"nan"6046791,"nan"6070870,"nan"6072564,"nan"6075522,"nan"6120505,"nan"6130231,"nan"6152190,"nan"6161910,"nan"6210133,"nan"6227410,"nan"6268931,"nan"6315184,"nan"6334463,"nan"6337785,"nan"6342162,"nan"6346493,"nan"6350406,"nan"6417149,"nan"6428730,"nan"6435446,"nan"6469754,"nan"6497547,"nan"6528338,"nan"6551308,"nan"6555138,"nan"6641038,"nan"6660848,"nan"6687713,"nan"6688412,"nan"6700829,"nan"6715173,"nan"6719266,"nan"6738156,"nan"6741583,"nan"6751525,"nan"6769971,"nan"6841537,"nan"6876628,"nan"6885473,"nan"6935486,"nan"6948850,"nan"6952485,"nan"6986340,"nan"6992063,"nan"7029698,"nan"7052845,"nan"7056616,"nan"7073210,"nan"7088726,"nan"7104764,"nan"7132195,"nan"7186431,"nan"7191106,"nan"'
    bad_indice = re.findall(r'[0-9]+', errors)
    data = []
    for idx, row in stl.iterrows():
        if str(idx) in bad_indice:
            continue
        if os.path.exists(os.path.join(sti, row[0])):
            data.append(
                (os.path.join(sti, row[0]), row[1])
            )
    for file in glob.glob(mji):
        if os.path.exists(file):
            label = os.path.split(file)[1].split('_')[1]
            data.append((                
                file,
                label
            ))
    json.dump(data, open('synth_train.json', 'w'))


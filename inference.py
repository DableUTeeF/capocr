from transformers import pipeline

if __name__ == '__main__':
    image_captioner = pipeline("image-to-text", model="workdir/mgpt_vit", device='gpu')
    print(image_captioner("imgs/19.jpg"))

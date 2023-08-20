from transformers import pipeline

if __name__ == '__main__':
    image_captioner = pipeline("image-to-text", model="./image-captioning-output")
    image_captioner("sample_image.png")

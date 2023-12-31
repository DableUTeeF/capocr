import torch
from transformers import ViTImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
from PIL import Image


def preprocess(image, image_processor):
    image = Image.open(image).convert('RGB')

    model_inputs = image_processor(images=image, return_tensors='pt')

    return model_inputs


def forward(model, model_inputs, generate_kwargs=None):
    if generate_kwargs is None:
        generate_kwargs = {'max_new_tokens': 100}
    inputs = model_inputs.pop(model.main_input_name)
    model_outputs = model.generate(inputs, **model_inputs, **generate_kwargs)
    return model_outputs


def postprocess(model_outputs):
    records = []
    for output_ids in model_outputs:
        record = {
            "generated_text": tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
            )
        }
        records.append(record)
    return records


def generate(inputs):
    model_inputs = preprocess(inputs, image_processor)
    model_outputs = forward(model, model_inputs)
    outputs = postprocess(model_outputs)


if __name__ == '__main__':
    p = "/media/palm/Data/ocr/cp/mgpt_vit_768_12_12_3072_16_8_pretrained"
    image_path = '/home/palm/PycharmProjects/mmmmocr/imgs/samples/m.png'
    im = Image.open(image_path)
    model = VisionEncoderDecoderModel.from_pretrained(p)
    image_processor = ViTImageProcessor.from_pretrained(p)
    tokenizer = AutoTokenizer.from_pretrained(p)

    generate(image_path)

    image_captioner = pipeline("image-to-text", model="workdir/mgpt_vit", device='gpu')
    print(image_captioner("imgs/19.jpg"))

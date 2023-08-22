import torch
from transformers import ViTImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
from PIL import Image


def preprocess(image, image_processor):
    image = Image.open(image).convert('RGB')

    model_inputs = image_processor(images=image, return_tensors='pt')

    return model_inputs


def forward(model, model_inputs, generate_kwargs=None):
    if generate_kwargs is None:
        generate_kwargs = {}
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
    model = VisionEncoderDecoderModel.from_pretrained("/media/palm/Data/ocr/cap/train/checkpoint-367500")
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    tokenizer = AutoTokenizer.from_pretrained("ai-forever/mGPT")

    generate('/media/palm/Data/ocr/data/images/002346.png')

    image_captioner = pipeline("image-to-text", model="workdir/mgpt_vit", device='gpu')
    print(image_captioner("imgs/19.jpg"))

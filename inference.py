import torch
from transformers import ViTImageProcessor, AutoTokenizer, VisionEncoderDecoderModel


def preprocess(self, image, prompt=None):
    image = load_image(image)

    if prompt is not None:
        if not isinstance(prompt, str):
            raise ValueError(
                f"Received an invalid text input, got - {type(prompt)} - but expected a single string. "
                "Note also that one single text can be provided for conditional image to text generation."
            )

        model_type = self.model.config.model_type

        if model_type == "git":
            model_inputs = self.image_processor(images=image, return_tensors=self.framework)
            input_ids = self.tokenizer(text=prompt, add_special_tokens=False).input_ids
            input_ids = [self.tokenizer.cls_token_id] + input_ids
            input_ids = torch.tensor(input_ids).unsqueeze(0)
            model_inputs.update({"input_ids": input_ids})

        elif model_type == "pix2struct":
            model_inputs = self.image_processor(images=image, header_text=prompt, return_tensors=self.framework)

        elif model_type != "vision-encoder-decoder":
            # vision-encoder-decoder does not support conditional generation
            model_inputs = self.image_processor(images=image, return_tensors=self.framework)
            text_inputs = self.tokenizer(prompt, return_tensors=self.framework)
            model_inputs.update(text_inputs)

        else:
            raise ValueError(f"Model type {model_type} does not support conditional text generation")

    else:
        model_inputs = self.image_processor(images=image, return_tensors=self.framework)

    if self.model.config.model_type == "git" and prompt is None:
        model_inputs["input_ids"] = None

    return model_inputs


def forward(self, model_inputs, generate_kwargs=None):
    if generate_kwargs is None:
        generate_kwargs = {}
    # FIXME: We need to pop here due to a difference in how `generation.py` and `generation.tf_utils.py`
    #  pahrse inputs. In the Tensorflow version, `generate` raises an error if we don't use `input_ids` whereas
    #  te PyTorch version matches it with `self.model.main_input_name` or `self.model.encoder.main_input_name`
    #  in the `_prepare_model_inputs` method.
    inputs = model_inputs.pop(self.model.main_input_name)
    model_outputs = self.model.generate(inputs, **model_inputs, **generate_kwargs)
    return model_outputs


def postprocess(self, model_outputs):
    records = []
    for output_ids in model_outputs:
        record = {
            "generated_text": self.tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
            )
        }
        records.append(record)
    return records


def generate(inputs):
    model_inputs = preprocess(inputs, **preprocess_params)
    model_outputs = forward(model_inputs, **forward_params)
    outputs = postprocess(model_outputs, **postprocess_params)


if __name__ == '__main__':
    model = VisionEncoderDecoderModel.from_pretrained("workdir/mgpt_vit")

    image_captioner = pipeline("image-to-text", model="workdir/mgpt_vit", device='gpu')
    print(image_captioner("imgs/19.jpg"))

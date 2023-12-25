import os
import math
from dataclasses import dataclass, field
from typing import Optional
import torch
from accelerate import Accelerator
import argparse
from dataset.ocr_data import *

from peft import (
    AutoPeftModelForCausalLM,
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    ViTImageProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from PIL import Image
import numpy as np
import evaluate
import nltk
from transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder import shift_tokens_right, VisionEncoderDecoderModel
from transformers.trainer_callback import ProgressCallback
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from torch.nn import CrossEntropyLoss


def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
):
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

    kwargs_decoder = {
        argument[len("decoder_"):]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
    }

    if encoder_outputs is None:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        encoder_outputs = self.encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs_encoder,
        )
    elif isinstance(encoder_outputs, tuple):
        encoder_outputs = BaseModelOutput(*encoder_outputs)

    encoder_hidden_states = encoder_outputs[0]

    # optionally project encoder_hidden_states
    if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
    ):
        encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

    # else:
    encoder_attention_mask = None

    if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
        decoder_input_ids = shift_tokens_right(
            labels, self.config.pad_token_id, self.config.decoder_start_token_id
        )

    # Decode
    decoder_outputs = self.decoder(
        inputs_embeds=encoder_hidden_states,  # n, 50, 768
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        use_cache=use_cache,
        past_key_values=past_key_values,
        return_dict=return_dict,
        **kwargs_decoder,
    )

    # Compute loss independent from decoder (as some shift the logits inside them)
    loss = None
    if labels is not None:
        logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
        loss_fct = CrossEntropyLoss()
        print(logits.size(), labels.size())
        loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.reshape(-1))

    if not return_dict:
        if loss is not None:
            return (loss,) + decoder_outputs + encoder_outputs
        else:
            return decoder_outputs + encoder_outputs

    return Seq2SeqLMOutput(
        loss=loss,
        logits=decoder_outputs.logits,
        past_key_values=decoder_outputs.past_key_values,
        decoder_hidden_states=decoder_outputs.hidden_states,
        decoder_attentions=decoder_outputs.attentions,
        cross_attentions=decoder_outputs.cross_attentions,
        encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        encoder_hidden_states=encoder_outputs.hidden_states,
        encoder_attentions=encoder_outputs.attentions,
    )


def on_log(self, args, state, control, logs=None, **kwargs):
    if state.is_local_process_zero and self.training_bar is not None:
        _ = logs.pop("total_flos", None)


def tokenization_fn(captions, max_target_length=120):
    """Run tokenization on captions."""
    labels = tokenizer(captions,
                       padding="max_length",
                       max_length=max_target_length,
                       return_tensors="pt",
                       truncation=True).input_ids

    return labels


def feature_extraction_fn(image_paths):
    images = [Image.open(image_file).convert('RGB') for image_file in image_paths]

    encoder_inputs = feature_extractor(images=images, return_tensors="pt")

    return encoder_inputs.pixel_values


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels


def collate_fn(batch):
    model_inputs = {'labels': [], 'pixel_values': []}
    for obj in batch:
        model_inputs['labels'].append(obj[1])
        model_inputs['pixel_values'].append(obj[0])
    model_inputs['labels'] = tokenization_fn(model_inputs['labels'])
    model_inputs['pixel_values'] = feature_extraction_fn(model_inputs['pixel_values'])
    return model_inputs


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds,
                                                     decoded_labels)
    rouge_result = rouge.compute(predictions=decoded_preds,
                                 references=decoded_labels,
                                 use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in rouge_result.items()}
    # bleu_result = bleu.compute(predictions=decoded_preds,
    #                            references=decoded_labels)
    # result.update({k: round(v * 100, 4) for k, v in bleu_result.items()})
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result


if __name__ == '__main__':
    ProgressCallback.on_log = on_log
    VisionEncoderDecoderModel.forward = forward
    parser = argparse.ArgumentParser()
    parser.add_argument('expname', type=str)
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--bs', type=int, default=2)
    parser.add_argument('--gradient_steps', type=int, default=4)
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    expname = args.expname + f'_{args.lora_r}_{args.lora_alpha}_{args.lora_dropout}_{args.bs}'
    print(expname, flush=True)
    target_modules = ['enc_to_dec_proj']
    if os.path.exists("/project/lt200060-capgen/palm/"):
        vit_model = "/project/lt200060-capgen/palm/huggingface/vit-base-patch16-224-in21k"
        text_decode_model = "/project/lt200060-capgen/palm/huggingface/typhoon-7b"
        rouge_path = '/home/nhongcha/hf-caption/rouge/'
        output_dir = os.path.join('/project/lt200060-capgen/palm/capocr/workdir/', expname)
        mji = '/project/lt200060-capgen/palm/ocr_data/train/mjsynth/mnt/ramdisk/max/90kDICT32px/*/*/*.jpg'
        stl = '/project/lt200060-capgen/palm/ocr_data/train/synthtext/labels.tsv'
        sti = '/project/lt200060-capgen/palm/ocr_data/train/synthtext/crop'
        funsd_labels = ('/project/lt200060-capgen/palm/ocr_data/funsd/train_label.json', '/project/lt200060-capgen/palm/ocr_data/funsd/test_label.json')
        funsd_images = '/project/lt200060-capgen/palm/ocr_data/funsd/crops'
        json_mode = 'synth_train.json'
        disable_tqdm = True
        tokenizer = AutoTokenizer.from_pretrained(text_decode_model, trust_remote_code=True)
        train_set = SynthDataset(tokenizer, mji, stl, sti, json_mode=json_mode)
        print(len(train_set), flush=True)
        valid_set = FunsdDataset(tokenizer, labels=funsd_labels, images=funsd_images)
        print(len(valid_set), flush=True)

    # elif os.path.exists("/media/palm/Data/capgen/"):
    #     pass

    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        vit_model = "google/vit-base-patch16-224-in21k"
        text_decode_model = "scb10x/typhoon-7b"
        rouge_path = 'rouge'
        output_dir = os.path.join('/tmp/out/mm_dino_8x8')
        mji = '/home/palm/data/mjsynth/images/*/*/*.jpg'
        stl = '/home/palm/data/synthtext/labels.tsv'
        sti = '/home/palm/data/synthtext/cropped'
        funsd_labels = ('/home/palm/data/funsd/train_label.json', '/home/palm/data/funsd/test_label.json')
        funsd_images = '/home/palm/data/funsd/crops'
        disable_tqdm = False
        tokenizer = AutoTokenizer.from_pretrained(text_decode_model, trust_remote_code=True)
        jsonl = 'data2/val.jsonl'
        train_set = ImageDataset(
            'data2',
            jsonl,
            is_training=True,
            single_jsonl=False
        )
        print(len(train_set), flush=True)
        valid_set = ImageDataset(
            'data2',
            jsonl,
            is_training=False,
            single_jsonl=False
        )

    logdir = os.path.join(args.logdir, expname)
    rouge = evaluate.load(rouge_path)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    base_model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(vit_model, text_decode_model)
    base_model.config.eos_token_id = tokenizer.eos_token_id
    base_model.config.decoder_start_token_id = tokenizer.bos_token_id
    base_model.config.pad_token_id = tokenizer.pad_token_id
    feature_extractor = ViTImageProcessor.from_pretrained(vit_model)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        # task_type="CAUSAL_LM",
    )

    # prepare model for training
    base_model = prepare_model_for_kbit_training(base_model)
    base_model = get_peft_model(base_model, peft_config)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.gradient_steps,
        per_device_eval_batch_size=1,
        learning_rate=4e-4,
        logging_steps=100,
        # max_steps=conf.max_steps,
        num_train_epochs=12,
        # report_to=conf.log_with,
        save_steps=5000,
        save_total_limit=1,
        logging_dir=logdir,
        warmup_steps=1000,
        warmup_ratio=1e-3,
        lr_scheduler_type='cosine',
        optim='adamw_torch',
        weight_decay=0.05,
        bf16=True,
        remove_unused_columns=True,
        gradient_checkpointing=True,
        run_name=expname,
        ddp_find_unused_parameters=False,
        disable_tqdm=disable_tqdm,
        evaluation_strategy="steps",
        eval_steps=20000,
    )

    trainer = Seq2SeqTrainer(
        model=base_model,
        train_dataset=train_set,
        eval_dataset=valid_set,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collate_fn,
    )

    trainer.train()
    trainer.save_model(output_dir)

    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(
        output_dir,
        safe_serialization=True,
    )

    # Free memory for merging weights
    del base_model
    torch.cuda.empty_cache()

    model = AutoPeftModelForCausalLM.from_pretrained(
        output_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = model.merge_and_unload()

    output_merged_dir = os.path.join(output_dir, "final_merged_checkpoint")
    model.save_pretrained(
        output_merged_dir,
        safe_serialization=True,
    )

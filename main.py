import os
import torch
from transformers import ViTImageProcessor, AutoTokenizer, VisionEncoderDecoderModel, ViTConfig, AutoModelForCausalLM, ViTModel
import nltk
import evaluate
import numpy as np
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from dataset.ocr_data import ImageDataset
from PIL import Image
import argparse


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
    if ignore_pad_token_for_loss:
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
    parser = argparse.ArgumentParser()
    parser.add_argument('expname', type=str)
    parser.add_argument('--img_w', type=int, default=160)
    parser.add_argument('--img_h', type=int, default=48)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--num_hidden_layers', type=int, default=12)
    parser.add_argument('--num_attention_heads', type=int, default=12)
    parser.add_argument('--intermediate_size', type=int, default=3072)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--logdir', type=str, default='./logs')
    args = parser.parse_args()
    expname = args.expname + f'_{args.img_w}_{args.img_h}_{args.hidden_size}_{args.num_hidden_layers}_{args.num_attention_heads}_{args.intermediate_size}_{args.patch_size}_{args.bs}'
    logdir = os.path.join(args.logdir, expname)

    if os.path.exists("/project/lt200060-capgen/coco"):
        vit_model = "/project/lt200060-capgen/palm/huggingface/vit-base-patch16-224-in21k"
        text_decode_model = "/project/lt200060-capgen/palm/huggingface/mGPT"
        src_dir = "/project/lt200060-capgen/palm/capocr"
        train_jsonl = '/project/lt200060-capgen/palm/capocr/data/train.jsonl'
        val_jsonl = '/project/lt200060-capgen/palm/capocr/data/val.jsonl'
        config_file = '/home/nhongcha/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco.py'
        detector_weight = '/project/lt200060-capgen/palm/pretrained/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth'
        output_dir = os.path.join('/project/lt200060-capgen/palm/capocr/workdir/', expname)
        bleu_path = '/home/nhongcha/hf-caption/bleu/bleu.py'
        rouge_path = '/home/nhongcha/hf-caption/rouge/'
        bs = args.bs
        workers = 4
    elif os.path.exists("/media/palm/Data/capgen/"):
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        vit_model = "google/vit-base-patch16-224-in21k"
        text_decode_model = "ai-forever/mGPT"
        src_dir = "/media/palm/Data/ocr/"
        train_jsonl = '/home/palm/PycharmProjects/capocr/data/train.jsonl'
        val_jsonl = '/home/palm/PycharmProjects/capocr/data/val.jsonl'
        config_file = '/home/palm/PycharmProjects/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco.py'
        detector_weight = ''
        output_dir = os.path.join('/tmp/out/mm_dino_8x8')
        bleu_path = 'bleu'
        rouge_path = 'bleu'
        bs = 1
        workers = 0
    else:
        vit_model = "google/vit-base-patch16-224-in21k"
        text_decode_model = "ai-forever/mGPT"
        src_dir = "/media/palm/Data/ocr/"
        train_jsonl = '/project/lt200060-capgen/coco/annotations/captions_train2017.json'
        val_jsonl = '/project/lt200060-capgen/coco/annotations/captions_val2017.json'
        config_file = '/home/palm/PycharmProjects/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco.py'
        detector_weight = '/home/palm/PycharmProjects/mmdetection/cp/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth'
        output_dir = os.path.join('/tmp/out/mm_dino_8x8')
        bleu_path = 'bleu'
        rouge_path = 'bleu'
        bs = 2
        workers = 0
    rouge = evaluate.load(rouge_path)
    bleu = evaluate.load(bleu_path)
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=args.overwrite)
    os.makedirs(logdir, exist_ok=args.overwrite)
    ignore_pad_token_for_loss = True

    encoder = ViTModel(
        ViTConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            intermediate_size=args.intermediate_size,
            patch_size=args.patch_size,
            add_cross_attention=True,
            image_size=(args.img_h, args.img_w)
        )
    )
    decoder = AutoModelForCausalLM.from_pretrained(text_decode_model, add_cross_attention=True)
    model = VisionEncoderDecoderModel(None, encoder, decoder)
    feature_extractor = ViTImageProcessor(
        size={"height": args.img_h, "width": args.img_w},
        image_mean=0,
        image_std=255,
    )
    tokenizer = AutoTokenizer.from_pretrained(text_decode_model)
    tokenizer.pad_token = tokenizer.eos_token

    # update the model config
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.save_pretrained(os.path.join(output_dir, 'train'))
    feature_extractor.save_pretrained(os.path.join(output_dir, 'train'))
    tokenizer.save_pretrained(os.path.join(output_dir, 'train'))

    train_set = ImageDataset(
        src_dir,
        train_jsonl
    )
    print(len(train_set), flush=True)
    valid_set = ImageDataset(
        src_dir,
        val_jsonl
    )
    print(len(valid_set), flush=True)
    # train_loader = DataLoader(train_set, **train_hyperparams)
    # valid_loader = DataLoader(valid_set, **valid_hyperparams)

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="epoch",
        save_strategy="steps",
        save_steps=5000,
        save_total_limit=1,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        num_train_epochs=12,
        output_dir=os.path.join(output_dir, 'train'),
        logging_dir=logdir,
        dataloader_num_workers=workers,
        logging_strategy='steps',
        logging_steps=100,
        disable_tqdm=True,
        # report_to=['tensorboard']
    )
    trainer = Seq2SeqTrainer(
        model=model,
        # tokenizer=feature_extractor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_set,
        eval_dataset=valid_set,
        data_collator=collate_fn,
    )
    trainer.train()

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import torch
from transformers import ViTImageProcessor, AutoTokenizer, VisionEncoderDecoderModel, ViTConfig, AutoModelForCausalLM, ViTModel
import nltk
import evaluate
import numpy as np
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from dataset.ocr_data import SynthDataset, FunsdDataset
from PIL import Image
import argparse
from transformers.trainer_callback import ProgressCallback


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
    images = []
    mask = []
    for image_file in image_paths:
        try:
            image = Image.open(image_file).convert('RGB')
            images.append(image) 
            mask.append(True)
        except:
            print(image_file)
            mask.append(False)
    try:
        encoder_inputs = feature_extractor(images=images, return_tensors="pt").pixel_values
    except:
        encoder_inputs = []
        for i, m in enumerate(mask):
            if m:
                try:
                    encoder_inputs.append(feature_extractor(images=images[i:i+1], return_tensors="pt").pixel_values)
                except:
                    mask[i] = False
        encoder_inputs = torch.cat(encoder_inputs, 0)

    return encoder_inputs, mask


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
    pixel_values, mask = feature_extraction_fn(model_inputs['pixel_values'])
    model_inputs['pixel_values'] = pixel_values
    model_inputs['labels'] = tokenization_fn(model_inputs['labels'])[mask]
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
                                 references=decoded_labels)
    result = rouge_result
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result


ProgressCallback.on_log = on_log
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
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--decoder', type=str, default='/project/lt200060-capgen/palm/huggingface/mGPT')
    parser.add_argument('--local-rank', type=int, default=0)
    args = parser.parse_args()
    expname = args.expname + f'_{args.hidden_size}_{args.num_hidden_layers}_{args.num_attention_heads}_{args.intermediate_size}_{args.patch_size}_{args.bs}'
    if args.pretrained:
        expname += '_pretrained'
    else:
        expname += f'_{args.img_w}_{args.img_h}'
    logdir = os.path.join(args.logdir, expname)
    print(expname, flush=True)

    vit_model = "google/vit-base-patch16-224-in21k"
    pretrained_vit_model = "google/vit-base-patch16-224-in21k"
    text_decode_model = "gpt2"
    output_dir = os.path.join('workdir/', expname)
    bleu_path = 'bleu'
    rouge_path = 'bleu'
    workers = 0
    rouge = evaluate.load(rouge_path)
    bleu = evaluate.load(bleu_path)
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=args.overwrite or args.resume)
    os.makedirs(logdir, exist_ok=args.overwrite or args.resume)
    ignore_pad_token_for_loss = True
    if not args.pretrained:
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

    else:
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(pretrained_vit_model, text_decode_model)
        feature_extractor = ViTImageProcessor.from_pretrained(vit_model)

    tokenizer = AutoTokenizer.from_pretrained(text_decode_model)
    tokenizer.pad_token = tokenizer.eos_token

    # update the model config
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    if not args.resume:
        model.save_pretrained(os.path.join(output_dir, 'train'))
        feature_extractor.save_pretrained(os.path.join(output_dir, 'train'))
        tokenizer.save_pretrained(os.path.join(output_dir, 'train'))

    train_set = SynthDataset(tokenizer)
    print(len(train_set), flush=True)
    valid_set = FunsdDataset(tokenizer)
    print(len(valid_set), flush=True)
    # train_loader = DataLoader(train_set, **train_hyperparams)
    # valid_loader = DataLoader(valid_set, **valid_hyperparams)

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        eval_steps=20000,
        save_strategy="steps",
        save_steps=5000,
        save_total_limit=1,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        num_train_epochs=12,
        output_dir=os.path.join(output_dir, 'train'),
        logging_dir=logdir,
        dataloader_num_workers=workers,
        logging_strategy='steps',
        logging_steps=100,
        disable_tqdm=False,
        local_rank=args.local_rank,
        warmup_steps=1000,
        warmup_ratio=1e-3,
        learning_rate=1e-4,
        lr_scheduler_type='cosine',
        save_safetensors=False,
        report_to=['tensorboard']
    )
    trainer = Seq2SeqTrainer(
        model=model,
        # tokenizer=feature_extractor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_set,
        eval_dataset=valid_set,
        data_collator=collate_fn,
        # n_gpu=2

    )
    trainer.train(resume_from_checkpoint=args.resume)

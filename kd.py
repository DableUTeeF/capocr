from transformers import AutoTokenizer, AutoConfig
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset, DatasetDict, Dataset
from models.language import DistillTrainGPT2LMHeadModel
import os
import argparse


def tokenize(element):
    outputs = tokenizer.batch_encode_plus(
        element,
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return input_batch


def data_prepare():
    wikipedia = load_dataset(f"graelo/wikipedia", '20230601.th', split='train')
    thaisum = open(os.path.join(txt_path, 'thaisum_train.txt')).read().split('\n')
    lst20 = open(os.path.join(txt_path, 'lst20_train.txt')).read().split('\n')
    wisesight = open(os.path.join(txt_path, 'wisesight_train.txt')).read().split('\n')

    data = wikipedia['text'] + thaisum + lst20 + wisesight

    train_data = data[10000:]
    valid_data = data[:10000]

    valid_tokens = tokenize(valid_data)
    train_tokens = tokenize(train_data)
    return train_tokens, valid_tokens


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('expname', type=str)
    parser.add_argument('--context_length', type=int, default=256)
    parser.add_argument('--grad_accum', type=int, default=8)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--logdir', type=str, default='./logs')
    args = parser.parse_args()
    expname = args.expname + f'_{args.context_length}_{args.grad_accum}_{args.bs}'
    logdir = os.path.join(args.logdir, expname)
    context_length = args.context_length
    if os.path.exists("/project/lt200060-capgen/coco"):
        bs = args.bs
        output_dir = os.path.join('/project/lt200060-capgen/palm/capocr/workdir/', expname)
        txt_path = '/project/lt200060-capgen/peune/ocr/txt/'
        workers = 4
    elif os.path.exists("/media/palm/Data/capgen/"):
        bs = 1
        output_dir = '/tmp/out/'
        txt_path = '/media/palm/Data/ocr/data/txt/'
        workers = 0
    else:
        bs = 2
        output_dir = '/tmp/out/'
        txt_path = '/project/lt200060-capgen/peune/ocr/txt/'
        workers = 0

    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=args.overwrite)
    os.makedirs(logdir, exist_ok=args.overwrite)

    tokenizer = AutoTokenizer.from_pretrained("ai-forever/mGPT")
    train_tokens, valid_tokens = data_prepare()
    config = AutoConfig.from_pretrained(
        "sshleifer/tiny-gpt2",
        vocab_size=len(tokenizer),
        n_ctx=context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = DistillTrainGPT2LMHeadModel(True, config)
    model.cuda()
    model_size = sum(t.numel() for t in model.parameters())
    print(f"GPT-2 size: {model_size / 1000 ** 2:.1f}M parameters")

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
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
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=train_tokens,
        eval_dataset=valid_tokens,
    )
    trainer.train()

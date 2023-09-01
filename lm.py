from transformers import AutoTokenizer, AutoConfig
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset, DatasetDict, Dataset
from models.language import DistillTrainGPT2LMHeadModel
import os
import argparse
from transformers.trainer_callback import ProgressCallback


def on_log(self, args, state, control, logs=None, **kwargs):
    if state.is_local_process_zero and self.training_bar is not None:
        _ = logs.pop("total_flos", None)


def tokenize(element):
    outputs = tokenizer.batch_encode_plus(
        element['content'],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {'input_ids': input_batch}


def data_prepare(debug):
    wikipedia = load_dataset(wiki, '20230601.th', split='train')
    thaisum = open(os.path.join(txt_path, 'thaisum_train.txt')).read().split('\n')
    lst20 = open(os.path.join(txt_path, 'lst20_train.txt')).read().split('\n')
    wisesight = open(os.path.join(txt_path, 'wisesight_train.txt')).read().split('\n')

    data = wikipedia['text'] + thaisum + lst20 + wisesight

    num_val = 1000 if debug else 10000
    valid_data = Dataset.from_dict({"content": data[:num_val]})
    valid_tokens = valid_data.map(
        tokenize, batched=True, remove_columns=valid_data.column_names
    )
    if not debug:
        train_data = Dataset.from_dict({"content": data[10000:]})
        train_tokens = train_data.map(
            tokenize, batched=True, remove_columns=train_data.column_names
        )
    else:
        train_tokens = valid_tokens
    return train_tokens, valid_tokens


ProgressCallback.on_log = on_log
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('expname', type=str)
    parser.add_argument('--context_length', type=int, default=256)
    parser.add_argument('--grad_accum', type=int, default=32)
    parser.add_argument('--worker', type=int, default=1)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--distil', action='store_true', default=False)
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--temperature', type=int, default=8)
    parser.add_argument('--alpha', type=int, default=0.5)
    parser.add_argument('--use_mse', action='store_true', default=False)
    parser.add_argument('--logdir', type=str, default='./logs')
    args = parser.parse_args()
    expname = args.expname + f'_{args.context_length}_{args.temperature}_{args.alpha}_{args.grad_accum}_{args.bs}'
    if args.distil:
        expname += '_distil'
    if args.pretrain:
        expname += '_prerained'
        if args.use_mse:
            expname += '_mse'
        else:
            expname += '_kl-div'
    logdir = os.path.join(args.logdir, expname)
    print(expname, flush=True)
    context_length = args.context_length
    if os.path.exists("/project/lt200060-capgen/coco"):
        wiki = '/home/nhongcha/.cache/huggingface/datasets/graelo___wikipedia/20230601.th/1.1.0/fa7b5c4902ab5a491d3fe295e3bf5c519890262c50a0401dcafd108de622068d'
        bs = args.bs
        output_dir = os.path.join('/project/lt200060-capgen/palm/capocr/workdir/', expname)
        txt_path = '/project/lt200060-capgen/peune/ocr/txt/'
        workers = args.worker
        teacher_path = "/project/lt200060-capgen/palm/huggingface/mGPT"
        config_path = "/project/lt200060-capgen/palm/huggingface/tiny-gpt2"
        disable_tqdm = True
    elif os.path.exists("/tarafs/data/project/proj0174-capgen/palm"):
        wiki = '/tarafs/data/project/proj0174-capgen/palm/graelo___wikipedia/20230601.th/1.1.0/fa7b5c4902ab5a491d3fe295e3bf5c519890262c50a0401dcafd108de622068d'
        bs = args.bs
        output_dir = os.path.join('workdir/', expname)
        txt_path = '/tarafs/data/project/proj0174-capgen/palm/caption/data/txt/'
        workers = args.worker
        teacher_path = "/tarafs/data/project/proj0174-capgen/palm/caption/cp/mGPT"
        config_path = "/tarafs/data/project/proj0174-capgen/palm/caption/cp/tiny-gpt2"
        disable_tqdm = True
    elif os.path.exists("/media/palm/Data/capgen/"):
        wiki = "graelo/wikipedia"
        bs = 1
        output_dir = '/media/palm/Data/ocr/cp/outs'
        txt_path = '/media/palm/Data/ocr/data/txt/'
        workers = 0
        teacher_path = "ai-forever/mGPT"
        config_path = "sshleifer/tiny-gpt2"
        disable_tqdm = False
    else:
        wiki = "graelo/wikipedia"
        bs = 2
        output_dir = '/tmp/out/'
        txt_path = '/project/lt200060-capgen/peune/ocr/txt/'
        workers = 0
        teacher_path = "ai-forever/mGPT"
        config_path = "sshleifer/tiny-gpt2"
        disable_tqdm = False

    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=args.overwrite)
    os.makedirs(logdir, exist_ok=args.overwrite)
    print(teacher_path, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(teacher_path)
    train_tokens, valid_tokens = data_prepare(args.debug)
    config = AutoConfig.from_pretrained(
        config_path,
        vocab_size=len(tokenizer),
        n_ctx=context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = DistillTrainGPT2LMHeadModel(args, teacher_path, config)
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
        logging_steps=1 if args.debug else 100,
        disable_tqdm=disable_tqdm,
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

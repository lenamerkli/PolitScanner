from os import environ
from certifi import where
environ['REQUESTS_CA_BUNDLE'] = where()
environ['SSL_CERT_FILE'] = where()
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import standardize_sharegpt, get_chat_template, train_on_responses_only
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from datetime import datetime

SEED = 2025-8-10
MAX_SEQ_LENGTH = 4096
MAX_STEPS = 1024
DTYPE = None
LOAD_IN_4BIT: bool = True
BASE_MODEL_NAME = 'unsloth/Qwen3-1.7B-unsloth-bnb-4bit'


def formatting_prompts_func(examples):
    """
    Convert chatml into text (Qwen3 only)
    :param examples: a LazyBatch object
    :return: dictionary with texts
    """
    convos = examples['messages']
    texts = [f"<|im_start|>user\n{convo[0]['content']}\n/no_think\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{convo[1]['content']}" for convo in convos]
    return {'text' : texts,}


def main() -> None:
    """
    Train the language model
    :return: None
    """
    dataset = load_dataset(
        'json',
        data_files='train.jsonl',
        split='train',
    )
    dataset = standardize_sharegpt(dataset)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    tokenizer = get_chat_template(tokenizer, chat_template='qwen3')
    # https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3N_(4B)-Conversational.ipynb#scrollTo=1ahE8Ys37JDJ
    dataset = dataset.map(formatting_prompts_func, batched=True)
    print(dataset[100])
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj',],
        lora_alpha=16,
        lora_dropout=0,
        bias='none',
        use_gradient_checkpointing=False,
        random_state=SEED,
        max_seq_length=MAX_SEQ_LENGTH,
        use_rslora=False,
        loftq_config=None,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=16,
            max_steps=MAX_STEPS,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            output_dir='llm_models',
            optim='adamw_8bit',
            seed=SEED,
        ),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part='<|im_start|>user\n',
        response_part='<|im_start|>assistant\n',
    )
    trainer.train()
    model.save_pretrained_merged(f"./models/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/", tokenizer)


if __name__ == '__main__':
    main()

import sys
sys.path.append('/home/lena/Documents/python/PolitScanner/util')

from os import environ
from certifi import where
environ['REQUESTS_CA_BUNDLE'] = where()
environ['SSL_CERT_FILE'] = where()
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import standardize_sharegpt, get_chat_template, CHAT_TEMPLATES
from jinja2 import Template
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from data import SentenceSplitterDataset
from random import seed, sample
from os.path import exists
from json import dumps
from util.debug import get_dict_structure
from sentence_splitter import BASE_MODEL_NAME, INPUT_SIZE, DTYPE, LOAD_IN_4BIT, escape, fix_multibyte_chars


NUM_EXAMPLES = 1_000
SEED = 1337
with open('prompt.md', 'r', encoding='utf-8') as _f:
    PROMPT = _f.read()
TEMPLATE = Template(CHAT_TEMPLATES['qwen3'][0])

FORMATTING_ITERS = 0


def format_dataset() -> None:
    dataset = SentenceSplitterDataset(train=True, transform=None, min_length=512, max_length=INPUT_SIZE, output_size=0, disable_pytorch=True)
    with open('train.jsonl', 'w') as file:
        seed(SEED)
        samples = sample(range(len(dataset)), NUM_EXAMPLES)
        for index in samples:
            datapoint = dataset[index]
            try:
                byte_string: bytes = datapoint[0]
                indices: list = datapoint[1]
                byte_string, indices = fix_multibyte_chars(byte_string, indices)
                indices = list(reversed(indices))
                text = escape(byte_string.decode('utf-8'))
                user = PROMPT.replace('{input}', text)
                sentences = []
                for i in range(len(indices) // 2):
                    a = indices[i * 2 + 1]
                    b = indices[i * 2] + a
                    for j in [a, b]:
                        if i > 0:
                            sentences.append(escape(byte_string[j:].decode('utf-8')))
                            byte_string = byte_string[:j]
                assistant = f"```text\n{'\n'.join(sentences)}\n```"
                # line = '{"messages": [{"role": "user", "content": "' + user + '"}, {"role": "assistant", "content": "' + assistant + '"}]}'
                line = {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}
                line = dumps(line)
                file.write(line + '\n')
            except Exception as e:
                e.add_note(f"Error in index {index} with datapoint ```{datapoint}```")
                raise e


def formatting_func(x, tokenizer):
    global FORMATTING_ITERS
    print(f"---BEGIN ITERATION {FORMATTING_ITERS}---")
    print(get_dict_structure(x))
    out = TEMPLATE.render(messages=x['messages'], add_generation_prompt=True, tools=[], enable_thinking=False)
    print(f"---END ITERATION {FORMATTING_ITERS}---")
    FORMATTING_ITERS += 1
    return out


def main() -> None:
    """
    Train the language model
    :return: None
    """
    format_dataset()
    # https://github.com/unsloth/unsloth/
    dataset = load_dataset(
        'json',
        data_files='train.jsonl',
        split='train',
    )
    dataset = standardize_sharegpt(dataset)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=INPUT_SIZE + 1024,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    tokenizer = get_chat_template(tokenizer, chat_template='qwen3')
    # Do model patching and add fast LoRA weights
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj',],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias='none',  # Supports any, but = 'none' is optimized
        use_gradient_checkpointing=False,  # True or 'unsloth' for very long context
        random_state=SEED,
        max_seq_length=INPUT_SIZE + 1024,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        formatting_func=lambda x: formatting_func(x, tokenizer),
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=16,
            max_steps=256,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            output_dir='llm_models',
            optim='adamw_8bit',
            seed=SEED,
        ),
    )
    trainer.train()
    model.save_pretrained_gguf('./models', tokenizer, quantization_method='q4_k_m')


if __name__ == '__main__':
    main()

import os

HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required but not set")

from huggingface_hub import login
login(token=HF_TOKEN)

from threading import Thread
from typing import Iterator

import gradio as gr
import spaces
import torch
import pyreft
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from pyreft import ReftModel

MAX_MAX_NEW_TOKENS = int(os.getenv("MAX_MAX_NEW_TOKENS", 2048))
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("DEFAULT_MAX_NEW_TOKENS", 1024))
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", 4096))


DESCRIPTION = """\
# Full fine-tune of Llama-3.2 Vision for NLP Project

### This project is designed exclusively for academic purposes, specifically for an NLP class.
It is trained on 10 examples containing my personal information within a timeframe of under 300 seconds.
"""

if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"


if torch.cuda.is_available():
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cuda", torch_dtype=torch.bfloat16
    )
    reft_model = ReftModel.load("kiko2001/Llama-3.2-1B-NLP", model, from_huggingface_hub=True)
    reft_model.set_device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.use_default_system_prompt = True
    tokenizer.pad_token = "[PAD]"  # Not just add_special_tokens
    model.resize_token_embeddings(len(tokenizer))


prompt_no_input_template = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant.
<</SYS>>

%s [/INST]
"""

@spaces.GPU
def generate(
    message: str,
    chat_history: list[tuple[str, str]],
    max_new_tokens: int = 1024,
) -> Iterator[str]:

    # tokenize and prepare the input
    prompt = prompt_no_input_template % message
    prompt = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = prompt["input_ids"]
    attention_mask = prompt["attention_mask"]
    
    if prompt["input_ids"].shape[-1] < 6:  # f3+l3 requires 6 tokens
        raise gr.Error("Input too short for REFT interventions")
    
    if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
        input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
        attention_mask = attention_mask[:, -MAX_INPUT_TOKEN_LENGTH:]
        gr.Warning(f"Trimmed input from conversation as it was longer than {MAX_INPUT_TOKEN_LENGTH} tokens.")

    positions = "f3+l3" 
    first_n, last_n = pyreft.parse_positions(positions)
    share_weights = False 

    unit_locations = torch.IntTensor([
        pyreft.get_intervention_locations(
            last_position=input_ids.shape[-1],
            first_n=first_n,
            last_n=last_n,
            pad_mode="last",
            num_interventions=4,
            share_weights=share_weights
        )
    ]).permute(1, 0, 2).tolist()
    
    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = {
        "base": {"input_ids": prompt["input_ids"], "attention_mask": prompt["attention_mask"]},
        "unit_locations": {"sources->base": (None, unit_locations)},
        "max_new_tokens": max_new_tokens,
        "intervene_on_prompt": True,
        "streamer": streamer,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "early_stopping": True,
        "do_sample": True,
        "temperature": 0.3,       # Add temperature for controlled randomness
        "top_p": 0.9, 
    }

    t = Thread(target=reft_model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs)


chat_interface = gr.ChatInterface(
    fn=generate,
    additional_inputs=[
        gr.Slider(
            label="Max new tokens",
            minimum=1,
            maximum=MAX_MAX_NEW_TOKENS,
            step=1,
            value=DEFAULT_MAX_NEW_TOKENS,
        )
    ],
    stop_btn=None,
    examples=[
        ["What is Kristijan Boshev’s function?"],
        ["Write a rap about Kristijan Boshev."],
    ],
)

with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION)
    chat_interface.render()

if __name__ == "__main__":
    demo.queue(max_size=20).launch(share=True)


# model.py

import os
import json
import uuid
import re
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.messages import convert_to_openai_messages


load_dotenv(override=True)

MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS=  int(os.getenv("MAX_NEW_TOKENS","400"))

SYSTEM_PROMPT= os.getenv(
    "SYSTEM_PROMPT",
"You are a helpful customer support assistant for an online store. "
"When the user asks about an order, use the lookup_order tool with the exact order ID. "
"Answer using ONLY the real values returned by the tool (status, product, dates, tracking number). "
"Never write placeholders such as {{Order Number}} or {{Website URL}}. "
"If the user has not given an order ID, ask for it. Keep answers short and specific.",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
)

def to_chat(messages):
    chat= convert_to_openai_messages(messages)
    for m in chat:
        for tc in m.get("tool_calls",[]):
            if isinstance(tc["function"]["arguments"],str):
                tc["function"]["arguments"]=json.loads(tc["function"]["arguments"])
    return [{'role':'system','content':SYSTEM_PROMPT}]+chat


_TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)

def parse_output(raw:str):
    calls=[]
    for blob in _TOOL_RE.findall(raw):
        try:
            d=json.loads(blob)
            args= d.get("arguments",{})
            if isinstance(args,str):
                args=json.loads(args)
            calls.append({'name':d["name"],'args':args,"id":str(uuid.uuid4()),"type":"tool_call"})

        except (json.JSONDecodeError,KeyError):
            continue

    text= _TOOL_RE.sub("",raw).strip()
    return text,calls

def generate_response(messages,tools=None):
    prompt= tokenizer.apply_chat_template(
        to_chat(messages),
        tools=[convert_to_openai_tool(n) for n in tools] if tools else None,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs= tokenizer(prompt,return_tensors='pt').to(model.device)
    with torch.no_grad():
        output= model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample= False,
            pad_token_id= tokenizer.eos_token_id,

        )
    raw= tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],skip_special_tokens=False)

        
    for tok in ("<|im_end|>", "<|endoftext|>", tokenizer.eos_token or ""):
        if tok:
            raw = raw.replace(tok, "")

    if os.getenv("DEBUG_RAW", "true").lower() == "true":
        print("RAW MODEL OUTPUT:", raw)

    return parse_output(raw.strip())




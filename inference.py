import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from peft import PeftModel

# Argument Parser Setup
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="ZhangFuXi/Zh-MT-LLM",
                    help="The directory of the model")
parser.add_argument("--tokenizer", type=str, default="ZhangFuXi/Zh-MT-LLM", help="Tokenizer path")
parser.add_argument("--adapter", type=str, default=None,
                    help="Path to the LoRA model checkpoint")
args = parser.parse_args()

if args.tokenizer is None:
    args.tokenizer = args.model
# 加载model
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map='auto'
)
# 加载adapter
if args.adapter is not None:
    model = PeftModel.from_pretrained(model, args.adapter)
# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
model = model.eval()
meta_instruction = "你是ITREC实验室开发的郑和，一个海事领域的专家。请为用户尽可能地提供帮助。注意用词的安全性，不要做任何违法的事情。\n"
user_input = "你好。"
prompt = meta_instruction + user_input
response, history = model.chat(tokenizer, prompt, history=None)
print(response)

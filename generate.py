import torch
from transformers import pipeline
import json
import warnings

model_id = "princeton-nlp/Llama-3-Instruct-8B-SimPO"

with open('chat_templates.json', 'r') as f:
    chat_templates = json.load(f)

if "llama-3" in model_id.lower():
    template = chat_templates["llama3"]
elif "mistral-7b-base" in model_id.lower():
    template = chat_templates["mistral-base"]
elif "mistral-7b-instruct" in model_id.lower():
    template = chat_templates["mistral-instruct"]
else:
    warnings.warn("No template set for the given model_id.")

generator = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)
generator.tokenizer.chat_template = template
outputs = generator([{"role": "user", "content": "What's the difference between llamas and alpacas?"}], do_sample=False, max_new_tokens=200)
print(outputs[0]['generated_text'])

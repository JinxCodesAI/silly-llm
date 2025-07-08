from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig
)
import torch
main_model = "Qwen/Qwen3-0.6B"
print(f"Loading tokenizer from {main_model}...")
tokenizer = AutoTokenizer.from_pretrained(main_model)
       
print(f"Loading main model {main_model}...")
model_kwargs = {
    "torch_dtype": torch.bfloat16,
    "device_map": "auto",
    "attn_implementation": "flash_attention_2"
}
        
main_model = AutoModelForCausalLM.from_pretrained(
    main_model, 
    **model_kwargs
)

messages = [{"role": "user", "content": "What is meaaning of life?"}]
texts_batch = [
    tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)]

inputs = tokenizer(
    texts_batch,
    return_tensors="pt",
    padding=True,
    truncation=True
).to(main_model.device)

generation_kwargs = {
    "max_new_tokens": 200,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "use_cache": True
    
}

print("Generating...")
outputs = main_model.generate(**inputs, **generation_kwargs)
input_length = inputs["input_ids"][0].ne(tokenizer.pad_token_id).sum().item()
generated_tokens = outputs[0, input_length:]
generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)


print("What is meaning of life? \n Answer:",generated_text)
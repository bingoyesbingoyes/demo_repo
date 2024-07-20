from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("../Qwen/Qwen-1_8B-Chat", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained("../Qwen/Qwen-1_8B-Chat", device_map="cuda:0", trust_remote_code=True, bf16=True).eval()
# model.generation_config = GenerationConfig.from_pretrained("../Qwen/Qwen-1_8B-Chat", trust_remote_code=True)
# model.generation_config.temperature = 0.9
# model.generation_config.do_sample = False
# model.generation_config = GenerationConfig.from_pretrained("../Qwen/Qwen-1_8B-Chat", trust_remote_code=True)

response, history = model.chat(tokenizer, "你好", history=None)
print(response)

response, history = model.chat(tokenizer, "给我讲一个年轻人奋斗创业最终取得成功的故事。", history=history)
print(response)
import time
import torch
import json
import re
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

print("Loading the Ovis model...")
model_id = "/nlsasfs/home/llmsdil/bsvibhav/adobe_ps/task2/models/Ovis1.6-Gemma2-9B"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    multimodal_max_length=8192,
    trust_remote_code=True
).cuda()
text_tokenizer = model.get_text_tokenizer()
visual_tokenizer = model.get_visual_tokenizer()

print("Model loaded. Watching for input prompts...")

input_file = "/nlsasfs/home/llmsdil/bsvibhav/adobe_ps/task2/input/input.txt"
output_file = "/nlsasfs/home/llmsdil/bsvibhav/adobe_ps/task2/outputs/output.txt"
json_file = "/nlsasfs/home/llmsdil/bsvibhav/adobe_ps/task2/outputs/output.json"

def process_prompt(image_path, text_prompt):
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return f"Error: Could not load image - {e}"

    query = f"<image>\n{text_prompt}"
    prompt, input_ids, pixel_values = model.preprocess_inputs(query, [image])
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    input_ids = input_ids.unsqueeze(0).to(device=model.device)
    attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
    pixel_values = [pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)]

    with torch.inference_mode():
        gen_kwargs = dict(
            max_new_tokens=2048,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            repetition_penalty=None,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=text_tokenizer.pad_token_id,
            use_cache=True
        )
        output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
        output = text_tokenizer.decode(output_ids, skip_special_tokens=True)
        return output

def parse_json_to_list(filename):
    with open(filename, 'r') as file:
        data = json.load(file)

    max_index = max(item["index"] for item in data)
    result = [False] * (max_index + 1)

    for item in data:
        index = item["index"]
        result[index] = (item["prediction"] == "fake")
    
    return result

def parse_output(output_text):
    pattern = r"\*\*(.*?)\*\*\s*:\s*(.*)"
    matches = re.findall(pattern, output_text)

    artifact_dict = {}
    for match in matches:
        artifact_name, explanation = match
        artifact_dict[artifact_name] = explanation.strip()

    return artifact_dict

# import json

def write_to_json(data, json_file):
    try:
        with open(json_file, "r+") as f:
            try:
                content = json.load(f)
            except json.JSONDecodeError:
                content = []

            content.append(data)
            
            f.seek(0)
            json.dump(content, f, indent=4)  
            f.truncate()  
    except Exception as e:
        print(f"Error writing to JSON file: {e}")



task1_json = "/nlsasfs/home/llmsdil/bsvibhav/adobe_ps/task2/input/64_task1.json"
image_list = parse_json_to_list(task1_json)
image_base_path = "/nlsasfs/home/llmsdil/bsvibhav/adobe_ps/task2/perturbed_images_32"

for i in tqdm(range(len(image_list)), desc="Processing images", ncols=100):
    if image_list[i]:
        try:
            with open(input_file, "r") as f:
                text_prompt = f.read()

            if not text_prompt:
                print("Input file is empty or does not contain a valid prompt.")
                time.sleep(5)
                continue

            print(f"Prompt detected for index {i}. Processing...")

            image_path = image_base_path + f"/{i}.png"
            output = process_prompt(image_path, text_prompt)

            with open(output_file, "w") as f:
                f.write(f"{output}")
            print("Output written to file.")

            artifact_data = parse_output(output)

            if artifact_data:
                new_data = {
                    "index": i,
                    "explanation": artifact_data
                }
                write_to_json(new_data, json_file)
                print(f"Output written to JSON file: {json_file}")

        except Exception as e:
            print(f"Error: {e}")

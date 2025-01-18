from scripts.inference import inference_once
model_path = 'wyddmw/WiseAD'
image_file = './demo/'
prompt_str = 'Are there any zebra crossings ahead?\n'

args = type('Args', (), {
    "model_path": model_path,
    "image_file": image_file,
    "prompt": prompt_str,
    "conv_mode": "v1",
    "temperature": 0, 
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512,
    "load_8bit": False,
    "load_4bit": False,
})()
inference_once(args)

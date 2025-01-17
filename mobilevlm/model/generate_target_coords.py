import sys
import torch
import argparse
from PIL import Image
from pathlib import Path
import os
import re
import numpy as np
import time
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from mobilevlm.model.mobilevlm import load_pretrained_model
from mobilevlm.conversation import conv_templates, SeparatorStyle
from mobilevlm.utils import disable_torch_init, process_images, tokenizer_image_token, KeywordsStoppingCriteria
from mobilevlm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN


def inference_once(model, tokenizer, images_tensor, target_waypoints):
    prompt = 'Your goal point coordinate is ({:.1f}, {:.1f}). What are five passing waypoint coordinates.'.format(target_waypoints[0][0], -target_waypoints[0][1])
    disable_torch_init()

    images_tensor = images_tensor[None]
    images_tensor = torch.from_numpy(images_tensor)
    images_tensor = images_tensor.to(model.device, dtype=torch.float16)
    conv = conv_templates['v1'].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    # Input
    input_ids = (tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda())
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
    # Inference
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=False,
            temperature=0,
            top_p=None,
            num_beams=1,
            max_new_tokens=512,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
    # Result-Decode
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    pattern = r'[-+]?\d*\.?\d+'
    try: 
        matches = re.findall(pattern, outputs)
        coords = np.array(matches).reshape(5, 2).astype(np.float)
    except:
        print('Failed to decode')
        coords = np.zeros((5, 2))
    return coords



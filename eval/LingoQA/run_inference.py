import sys
import torch
import argparse
from PIL import Image
from pathlib import Path
import os
import json

sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))
from mobilevlm.model.mobilevlm import load_pretrained_model
from mobilevlm.conversation import conv_templates, SeparatorStyle
from mobilevlm.utils import disable_torch_init, process_images, tokenizer_image_token, KeywordsStoppingCriteria
from mobilevlm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN


def EvalLingoQA(args):
    # read evaluation json file
    eval_json = json.load(open(args.eval_file, 'r'))
    eval_data = args.eval_file.split('/')[-1].split('.')[0]
    disable_torch_init()
    model_name = args.model_path.split('/')[-1]
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.load_8bit, args.load_4bit)
    result = []
    if args.output_path is not None:
        os.makedirs(args.output_path, exist_ok=True)
        f = open(os.path.join(args.output_path, 'lingoqa_results.json'), 'a', encoding='utf-8')
    else:
        f = open('results.json', 'a', encoding='utf-8')
    
    for index, data in enumerate(eval_json):
        images = []
        segment_id = data['image_id']
        question_id = data['question_id']
        question = prompt = data['conversations'][0]['value'].strip()
        answer_gt = data['conversations'][1]['value'].strip()
        if isinstance(data['image'], list):
            image_lists = sorted(data['image'])
            for image in image_lists:
                images.append(Image.open(image).convert('RGB'))
        else:
            images.append(Image.open(data['image']).convert('RGB'))
        images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
        images_tensor = images_tensor[None]
        conv = conv_templates[args.conv_mode].copy()
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
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
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
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        print(f"🚀 {model_name}: {outputs.strip()}\n")
        result.append(
            {
                'question_id': question_id,
                'question': question,
                'segment_id': segment_id,
                'answer': f'{outputs.strip()}',
                'answer_gt': f'{answer_gt.strip()}',
            }
        )
        # break
    json_data = json.dumps(result, indent=4)
    f.write(json_data)
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="wyddmw/WiseAD")
    parser.add_argument("--conv_mode", type=str, default="v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--load_8bit", type=bool, default=False)
    parser.add_argument("--load_4bit", type=bool, default=False)
    parser.add_argument("--eval_file", type=str, default='./data/LingoQA/evaluation_data.json')
    parser.add_argument("--output_path", type=str, default='eval_results/')

    args = parser.parse_args()

    EvalLingoQA(args)

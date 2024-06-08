import os
import time

import torch
from tqdm import tqdm

from llava.model.builder import load_pretrained_model
import requests
from PIL import Image
from io import BytesIO
import re
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)

model_path = "models/llava-v1.5-13b"
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=model_name
)


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(
    #     args.model_path, args.model_base, model_name
    # )

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    images = load_images(image_files)
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

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

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(
            f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
        )
    outputs = tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    # print(outputs)
    return outputs


model_path = "models/llava-v1.5-13b"

prompt_1 = "[Question1:Please describe this picture.]"
prompt_2 = "[Question2:What are the names of both teams? What colors are the uniforms of both teams?]"
prompt_3 = "[Question3:where is the soccer ball?]"
prompt_4 = "[Question4:Which individuals in the picture are conspicuously running towards the soccer ball?]"
prompt_5 = "[Question5:What are the other people in the picture doing?]"
prompt_6 = r"Please answer the questions one by one, and follow the format: {[Answer1]\n[Answer2]\n[Answer3]\n[Answer4]\n[Answer5]}"

prompt = prompt_1 + "\n" + prompt_2 + "\n" + prompt_3 + "\n" + prompt_4 + "\n" + prompt_5 + "\n" + prompt_6


def get_text(image_file, prompt):
    args = type('Args', (), {
        "temperature": 0,
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()
    text = eval_model(args)
    return text


def get_path(root_folder):
    subfolders_and_files = []

    # 获取一级文件夹下的所有文件和文件夹
    for item in os.listdir(root_folder):
        item_path = os.path.join(root_folder, item)

        # 检查是否是文件夹
        if os.path.isdir(item_path):
            # 获取文件夹下的所有文件和文件夹
            sub_items = [os.path.join(item_path, sub_item) for sub_item in os.listdir(item_path)]
            subfolders_and_files.extend(sub_items)

    return subfolders_and_files


def create_file(file_path, content):
    # 确保文件夹存在
    folder_path = os.path.dirname(file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 创建文件并写入内容
    with open(file_path, 'w') as file:
        file.write(content)


if __name__ == "__main__":
    image_folder = get_path("data/UCF101_v2/rgb-images/SoccerJuggling")

    for image_file in tqdm(image_folder):
        text_file = image_file.replace("rgb-images", "sentences").replace(".jpg", ".txt")
        # 判断文件是否存在
        if not os.path.exists(text_file):
            text = get_text(image_file, prompt)

            create_file(text_file, str(text))

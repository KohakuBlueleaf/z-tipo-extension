import os
import re
from pathlib import Path
from typing import Any

import torch
import folder_paths
from comfy.cli_args import args

from ..tipo_installer import install_tipo_kgen, install_llama_cpp

install_llama_cpp()
install_tipo_kgen()

import kgen.models as models
import kgen.executor.tipo as tipo
from kgen.executor.tipo import (
    parse_tipo_request,
    tipo_single_request,
    tipo_runner,
    apply_tipo_prompt,
    parse_tipo_result,
    OPERATION_LIST,
)
from kgen.formatter import seperate_tags, apply_format
from kgen.logging import logger


models.model_dir = Path(folder_paths.models_dir) / "kgen"
os.makedirs(models.model_dir, exist_ok=True)
logger.info(f"Using model dir: {models.model_dir}")

model_list = tipo.models.tipo_model_list
MODEL_NAME_LIST = [
    f"{model_name} | {file}".strip("_")
    for model_name, ggufs in models.tipo_model_list
    for file in ggufs
] + [i[0] for i in models.tipo_model_list]


attn_syntax = (
    r"\\\(|"
    r"\\\)|"
    r"\\\[|"
    r"\\]|"
    r"\\\\|"
    r"\\|"
    r"\(|"
    r"\[|"
    r":\s*([+-]?[.\d]+)\s*\)|"
    r"\)|"
    r"]|"
    r"[^\\()\[\]:]+|"
    r":"
)
re_attention = re.compile(
    attn_syntax,
    re.X,
)

re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)


def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text

    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and round_brackets:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, text)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


def apply_strength(tag_map, strength_map, strength_map_nl):
    for cate in tag_map.keys():
        new_list = []
        # Skip natural language output at first
        if isinstance(tag_map[cate], str):
            # Ensure all the parts in the strength_map are in the prompt
            if all(part in tag_map[cate] for part, strength in strength_map_nl):
                org_prompt = tag_map[cate]
                new_prompt = ""
                for part, strength in strength_map_nl:
                    before, org_prompt = org_prompt.split(part, 1)
                    new_prompt += before.replace("(", "\(").replace(")", "\)")
                    part = part.replace("(", "\(").replace(")", "\)")
                    new_prompt += f"({part}:{strength})"
                new_prompt += org_prompt
            else:
                # Fix: ensure fallback if new_prompt is not constructed
                tag_map[cate] = tag_map[cate]
            continue

        for org_tag in tag_map[cate]:
            tag = org_tag.replace("(", "\(").replace(")", "\)")
            if org_tag in strength_map:
                new_list.append(f"({tag}:{strength_map[org_tag]})")
            else:
                new_list.append(tag)
        tag_map[cate] = new_list
    return tag_map


current_model = None

# Constants
FUNCTION = "execute"
CATEGORY = "utils/promptgen"


class TIPO:
    INPUT_TYPES = lambda: {
        "required": {
            "tags": ("STRING", {"defaultInput": True, "multiline": True}),
            "nl_prompt": ("STRING", {"defaultInput": True, "multiline": True}),
            "ban_tags": ("STRING", {"defaultInput": True, "multiline": True}),
            "tipo_model": (MODEL_NAME_LIST, {"default": MODEL_NAME_LIST[0]}),
            "format": (
                "STRING",
                {
                    "default": """<|special|>, 
<|characters|>, <|copyrights|>, 
<|artist|>, 

<|general|>,

<|extended|>.

<|quality|>, <|meta|>, <|rating|>""",
                    "multiline": True,
                },
            ),
            "width": ("INT", {"default": 1024, "max": 16384}),
            "height": ("INT", {"default": 1024, "max": 16384}),
            "temperature": ("FLOAT", {"default": 0.5, "step": 0.01}),
            "top_p": ("FLOAT", {"default": 0.95, "step": 0.01}),
            "min_p": ("FLOAT", {"default": 0.05, "step": 0.01}),
            "top_k": ("INT", {"default": 80}),
            "tag_length": (
                ["very_short", "short", "long", "very_long"],
                {"default": "long"},
            ),
            "nl_length": (
                ["very_short", "short", "long", "very_long"],
                {"default": "long"},
            ),
            "seed": ("INT", {"default": 1234}),
            "device": (["cpu", "cuda"], {"default": "cuda"}),
        },
    }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "prompt",
        "user_prompt",
        "unformatted_prompt",
        "unformatted_user_prompt",
    )
    FUNCTION = FUNCTION
    CATEGORY = CATEGORY

    def execute(
        self,
        tipo_model: str,
        tags: str,
        nl_prompt: str,
        width: int,
        height: int,
        seed: int,
        tag_length: str,
        nl_length: str,
        ban_tags: str,
        format: str,
        temperature: float,
        top_p: float,
        min_p: float,
        top_k: int,
        device: str,
    ):
        global current_model
        if (tipo_model, device) != current_model:
            if " | " in tipo_model:
                model_name, gguf_name = tipo_model.split(" | ")
                target_file = f"{model_name.split('/')[-1]}_{gguf_name}"
                if str(models.model_dir / target_file) not in models.list_gguf():
                    models.download_gguf(model_name, gguf_name)
                target = os.path.join(str(models.model_dir), target_file)
                gguf = True
            else:
                target = tipo_model
                gguf = False
            if gguf:
                extra = {"main_device": args.cuda_device or 0}
            else:
                extra = {}
                device = f"{torch.device.type}:{args.cuda_device or 0}"
            models.load_model(target, gguf, device=device, **extra)
            current_model = (tipo_model, device)
        aspect_ratio = width / height
        prompt_without_extranet = tags
        prompt_parse_strength = parse_prompt_attention(prompt_without_extranet)

        nl_prompt_parse_strength = parse_prompt_attention(nl_prompt)
        nl_prompt = ""
        strength_map_nl = []
        for part, strength in nl_prompt_parse_strength:
            nl_prompt += part
            if strength == 1:
                continue
            strength_map_nl.append((part, strength))

        black_list = [tag.strip() for tag in ban_tags.split(",") if tag.strip()]
        tipo.BAN_TAGS = black_list
        all_tags = []
        strength_map = {}
        for part, strength in prompt_parse_strength:
            part_tags = [tag.strip() for tag in part.strip().split(",") if tag.strip()]
            all_tags.extend(part_tags)
            if strength == 1:
                continue
            for tag in part_tags:
                strength_map[tag] = strength

        tag_length = tag_length.replace(" ", "_")
        nl_length = nl_length.replace(" ", "_")
        org_tag_map = seperate_tags(all_tags)
        meta, operations, general, nl_prompt = parse_tipo_request(
            org_tag_map,
            nl_prompt,
            tag_length_target=tag_length,
            nl_length_target=nl_length,
            generate_extra_nl_prompt=(not nl_prompt and "<|extended|>" in format)
            or "<|generated|>" in format,
        )
        meta["aspect_ratio"] = f"{aspect_ratio:.1f}"

        org_formatted_prompt = parse_tipo_result(
            apply_tipo_prompt(
                meta,
                general,
                nl_prompt,
                "short_to_tag_to_long",
                tag_length,
                True,
                gen_meta=True,
            )
        )
        org_formatted_prompt = apply_strength(
            org_formatted_prompt, strength_map, strength_map_nl
        )
        formatted_prompt_by_user = apply_format(org_formatted_prompt, format)
        unformatted_prompt_by_user = tags + nl_prompt

        tag_map, _ = tipo_runner(
            meta,
            operations,
            general,
            nl_prompt,
            temperature=temperature,
            seed=seed,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
        )

        addon = {
            "tags": [],
            "nl": "",
        }
        for cate in tag_map.keys():
            if cate == "generated" and addon["nl"] == "":
                addon["nl"] = tag_map[cate]
                continue
            if cate == "extended":
                extended = tag_map[cate]
                addon["nl"] = extended
                continue
            if cate not in org_tag_map:
                continue
            for tag in tag_map[cate]:
                if tag in org_tag_map[cate]:
                    continue
                addon["tags"].append(tag)
        addon = apply_strength(addon, strength_map, strength_map_nl)
        unformatted_prompt_by_tipo = (
            tags + ", " + ", ".join(addon["tags"]) + "\n" + addon["nl"]
        )

        tag_map = apply_strength(tag_map, strength_map, strength_map_nl)
        formatted_prompt_by_tipo = apply_format(tag_map, format)
        return (
            formatted_prompt_by_tipo,
            formatted_prompt_by_user,
            unformatted_prompt_by_tipo,
            unformatted_prompt_by_user,
        )


class TIPOOperation:
    INPUT_TYPES = lambda: {
        "required": {
            "tags": ("STRING", {"defaultInput": True, "multiline": True}),
            "nl_prompt": ("STRING", {"defaultInput": True, "multiline": True}),
            "ban_tags": ("STRING", {"defaultInput": True, "multiline": True}),
            "tipo_model": (MODEL_NAME_LIST, {"default": MODEL_NAME_LIST[0]}),
            "operation": (
                sorted(OPERATION_LIST),
                {"default": sorted(OPERATION_LIST)[0]},
            ),
            "width": ("INT", {"default": 1024, "max": 16384}),
            "height": ("INT", {"default": 1024, "max": 16384}),
            "temperature": ("FLOAT", {"default": 0.5, "step": 0.01}),
            "top_p": ("FLOAT", {"default": 0.95, "step": 0.01}),
            "min_p": ("FLOAT", {"default": 0.05, "step": 0.01}),
            "top_k": ("INT", {"default": 80}),
            "tag_length": (
                ["very_short", "short", "long", "very_long"],
                {"default": "long"},
            ),
            "nl_length": (
                ["very_short", "short", "long", "very_long"],
                {"default": "long"},
            ),
            "seed": ("INT", {"default": 1234}),
            "device": (["cpu", "cuda"], {"default": "cuda"}),
        },
    }

    RETURN_TYPES = ("LIST", "LIST")
    RETURN_NAMES = (
        "full_output",
        "addon_output",
    )
    FUNCTION = FUNCTION
    CATEGORY = CATEGORY

    def execute(
        self,
        tipo_model: str,
        tags: str,
        nl_prompt: str,
        width: int,
        height: int,
        seed: int,
        tag_length: str,
        nl_length: str,
        ban_tags: str,
        operation: str,
        temperature: float,
        top_p: float,
        min_p: float,
        top_k: int,
        device: str,
    ):
        global current_model
        if (tipo_model, device) != current_model:
            if " | " in tipo_model:
                model_name, gguf_name = tipo_model.split(" | ")
                target_file = f"{model_name.split('/')[-1]}_{gguf_name}"
                if str(models.model_dir / target_file) not in models.list_gguf():
                    models.download_gguf(model_name, gguf_name)
                target = os.path.join(str(models.model_dir), target_file)
                gguf = True
            else:
                target = tipo_model
                gguf = False
            models.load_model(target, gguf, device=device)
            current_model = (tipo_model, device)
        aspect_ratio = width / height
        prompt_without_extranet = tags
        prompt_parse_strength = parse_prompt_attention(prompt_without_extranet)

        nl_prompt_wihtout_extranet = nl_prompt
        nl_prompt_parse_strength = parse_prompt_attention(nl_prompt)
        nl_prompt = ""
        strength_map_nl = []
        for part, strength in nl_prompt_parse_strength:
            nl_prompt += part
            if strength == 1:
                continue
            strength_map_nl.append((part, strength))

        black_list = [tag.strip() for tag in ban_tags.split(",") if tag.strip()]
        tipo.BAN_TAGS = black_list
        all_tags = []
        strength_map = {}
        for part, strength in prompt_parse_strength:
            part_tags = [tag.strip() for tag in part.strip().split(",") if tag.strip()]
            all_tags.extend(part_tags)
            if strength == 1:
                continue
            for tag in part_tags:
                strength_map[tag] = strength

        tag_length = tag_length.replace(" ", "_")
        org_tag_map = seperate_tags(all_tags)
        meta, operations, general, nl_prompt = tipo_single_request(
            org_tag_map,
            nl_prompt,
            tag_length_target=tag_length,
            nl_length_target=nl_length,
            operation=operation,
        )
        meta["aspect_ratio"] = f"{aspect_ratio:.1f}"

        tag_map, _ = tipo_runner(
            meta,
            operations,
            general,
            nl_prompt,
            temperature=temperature,
            seed=seed,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
        )

        addon = {
            "tags": [],
            "nl": "",
        }
        for cate in tag_map.keys():
            if cate == "generated" and addon["nl"] == "":
                addon["nl"] = tag_map[cate]
                continue
            if cate == "extended":
                extended = tag_map[cate]
                addon["nl"] = extended
                continue
            if cate not in org_tag_map:
                continue
            for tag in tag_map[cate]:
                if tag in org_tag_map[cate]:
                    continue
                addon["tags"].append(tag)
        addon = apply_strength(addon, strength_map, strength_map_nl)
        addon["user_tags"] = prompt_without_extranet
        addon["user_nl"] = nl_prompt_wihtout_extranet

        tag_map = apply_strength(tag_map, strength_map, strength_map_nl)
        return (
            tag_map,
            addon,
        )


class TIPOFormat:
    INPUT_TYPES = lambda: {
        "required": {
            "full_output": ("LIST", {"default": []}),
            "addon_output": ("LIST", {"default": []}),
            "format": (
                "STRING",
                {
                    "default": """<|special|>, 
<|characters|>, <|copyrights|>, 
<|artist|>, 

<|general|>,

<|extended|>.

<|quality|>, <|meta|>, <|rating|>""",
                    "multiline": True,
                },
            ),
        },
    }
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "prompt",
        "user_prompt",
        "unformatted_prompt",
        "unformatted_user_prompt",
    )
    FUNCTION = FUNCTION
    CATEGORY = CATEGORY

    def execute(
        self,
        full_output: list,
        addon_output: dict[str, Any],
        format: str,
    ):
        tags = addon_output.pop("user_tags", "")
        nl_prompt = addon_output.pop("user_nl", "")
        addon = addon_output
        tag_map = full_output

        prompt_without_extranet = tags
        prompt_parse_strength = parse_prompt_attention(prompt_without_extranet)

        nl_prompt_parse_strength = parse_prompt_attention(nl_prompt)
        nl_prompt = ""
        strength_map_nl = []
        for part, strength in nl_prompt_parse_strength:
            nl_prompt += part
            if strength == 1:
                continue
            strength_map_nl.append((part, strength))

        all_tags = []
        strength_map = {}
        for part, strength in prompt_parse_strength:
            part_tags = [tag.strip() for tag in part.strip().split(",") if tag.strip()]
            all_tags.extend(part_tags)
            if strength == 1:
                continue
            for tag in part_tags:
                strength_map[tag] = strength

        org_tag_map = seperate_tags(all_tags)
        meta, _, general, nl_prompt = parse_tipo_request(
            org_tag_map,
            nl_prompt,
        )

        org_formatted_prompt = parse_tipo_result(
            apply_tipo_prompt(
                meta,
                general,
                nl_prompt,
                "short_to_tag_to_long",
                "long",
                True,
                gen_meta=True,
            )
        )
        org_formatted_prompt = apply_strength(
            org_formatted_prompt, strength_map, strength_map_nl
        )
        formatted_prompt_by_user = apply_format(org_formatted_prompt, format)
        unformatted_prompt_by_user = tags + nl_prompt
        formatted_prompt_by_tipo = apply_format(tag_map, format)
        unformatted_prompt_by_tipo = (
            tags + ", " + ", ".join(addon["tags"]) + "\n" + addon["nl"]
        )

        return (
            formatted_prompt_by_tipo,
            formatted_prompt_by_user,
            unformatted_prompt_by_tipo,
            unformatted_prompt_by_user,
        )


NODE_CLASS_MAPPINGS = {
    "TIPO": TIPO,
    "TIPOOperation": TIPOOperation,
    "TIPOFormat": TIPOFormat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TIPO": "TIPO",
    "TIPOOperation": "TIPO Single Operation",
    "TIPOFormat": "TIPO Format",
}

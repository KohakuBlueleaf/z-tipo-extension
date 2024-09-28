import os
import re

llama_cpp_python_wheel = (
    "llama-cpp-python --prefer-binary "
    "--extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/{}/{}"
)
try:
    import llama_cpp
except:
    print("Attempting to install LLaMA-CPP-Python")
    import torch

    has_cuda = torch.cuda.is_available()
    cuda_version = torch.version.cuda.replace(".", "")
    package = llama_cpp_python_wheel.format(
        "AVX2", f"cu{cuda_version}" if has_cuda else "cpu"
    )
    os.system(f"pip install {package}")

try:
    import kgen
except:
    GH_TOKEN = os.getenv("GITHUB_TOKEN") + "@"
    if GH_TOKEN == "@":
        GH_TOKEN = ""
    git_url = f"https://{GH_TOKEN}github.com/KohakuBlueleaf/TIPO-KGen@tipo"

    ## call pip install
    os.system(f"pip install git+{git_url}")

import torch

import kgen.models as models
import kgen.executor.tipo as tipo
from kgen.executor.tipo import parse_tipo_request, tipo_runner
from kgen.formatter import seperate_tags, apply_format
from kgen.metainfo import TARGET
from kgen.logging import logger


model_name = "KBlueLeaf/TIPO-200M-dev"
gguf_name = "TIPO-200M-40Btok-F16.gguf"
model_name = ""
gguf_name = "TIPO-500M_epoch5-F16.gguf"
try:
    models.load_model(
        f"{model_name.split('/')[-1]}_{gguf_name}".strip("_"), gguf=True, device="cuda"
    )
except:
    models.download_gguf(model_name, gguf_name)
    models.load_model(
        f"{model_name.split('/')[-1]}_{gguf_name}".strip("_"), gguf=True, device="cuda"
    )


re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:\s*([+-]?[.\d]+)\s*\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
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


class TIPO:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "tags": ("STRING", {"default": "", "multiline": True}),
                "nl_prompt": ("STRING", {"default": "", "multiline": True}),
                "ban_tags": ("STRING", {"default": "", "multiline": True}),
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
                "width": ("INT", {"default": 1024}),
                "height": ("INT", {"default": 1024}),
                "temperature": ("FLOAT", {"default": 0.5, "step": 0.01}),
                "tag_length": (
                    ["very_short", "short", "long", "very_long"],
                    {"default": "long"},
                ),
                "seed": ("INT", {"default": 1234}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "execute"
    CATEGORY = "utils/promptgen"

    def execute(
        self,
        tags: str,
        nl_prompt: str,
        width: int,
        height: int,
        seed: int,
        tag_length: str,
        ban_tags: str,
        format: str,
        temperature: float,
    ):
        aspect_ratio = width / height
        prompt_without_extranet = tags
        prompt_parse_strength = parse_prompt_attention(prompt_without_extranet)

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
        print(strength_map)

        tag_length = tag_length.replace(" ", "_")
        tag_map = seperate_tags(all_tags)
        meta, operations, general, nl_prompt = parse_tipo_request(
            tag_map,
            nl_prompt,
            tag_length_target=tag_length,
            generate_extra_nl_prompt=(not nl_prompt and "<|extended|>" in nl_prompt)
            or "<|generated|>" in nl_prompt,
        )
        meta["aspect_ratio"] = f"{aspect_ratio:.1f}"

        tag_map, _ = tipo_runner(
            meta,
            operations,
            general,
            nl_prompt,
            temperature=temperature,
            seed=seed,
        )
        for cate in tag_map.keys():
            new_list = []
            # Skip natural language output at first
            if isinstance(tag_map[cate], str):
                tag_map[cate] = tag_map[cate].replace("(", "\(").replace(")", "\)")
                continue
            for org_tag in tag_map[cate]:
                tag = org_tag.replace("(", "\(").replace(")", "\)")
                if org_tag in strength_map:
                    new_list.append(f"({tag}:{strength_map[org_tag]})")
                else:
                    new_list.append(tag)
            tag_map[cate] = new_list
        prompt_by_tipo = apply_format(tag_map, format)
        result = prompt_by_tipo
        print(result)
        return (result,)


NODE_CLASS_MAPPINGS = {
    "TIPO": TIPO,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TIPO": "TIPO",
}

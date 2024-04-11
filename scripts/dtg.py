import os
import json
import random
from functools import lru_cache

import torch
import gradio as gr
from transformers import LlamaForCausalLM, LlamaTokenizer

import modules.scripts as scripts
from modules.scripts import basedir, OnComponent
from modules.processing import (
    StableDiffusionProcessingTxt2Img,
    StableDiffusionProcessingImg2Img,
)
from modules.prompt_parser import parse_prompt_attention
from modules.extra_networks import parse_prompt

from kgen.formatter import seperate_tags, apply_format, apply_dtg_prompt
from kgen.metainfo import TARGET
from kgen.generate import tag_gen
from kgen.logging import logger


ext_dir = basedir()
all_model_file = [f for f in os.listdir(ext_dir + "/models") if f.endswith(".gguf")]


try:
    from llama_cpp import Llama

    text_model = Llama(
        os.path.join(ext_dir, "models", all_model_file[-1]),
        n_ctx=384,
        n_gpu_layers=100,
        verbose=False,
    )
    logger.info("Llama-cpp-python/gguf model loaded")
except Exception:
    logger.warning(
        "Llama-cpp-python/gguf model not found, using transformers to load model"
    )

    text_model = (
        LlamaForCausalLM.from_pretrained("KBlueLeaf/DanTagGen-beta").eval().half()
    )
tokenizer = LlamaTokenizer.from_pretrained("KBlueLeaf/DanTagGen-beta")


SEED_MAX = 2**31 - 1
QUOTESWAP = str.maketrans("'\"", "\"'")
TOTAL_TAG_LENGTH = {
    "VERY_SHORT": "very short",
    "SHORT": "short",
    "LONG": "long",
    "VERY_LONG": "very long",
}
TOTAL_TAG_LENGTH_TAGS = {
    TOTAL_TAG_LENGTH["VERY_SHORT"]: "<|very_short|>",
    TOTAL_TAG_LENGTH["SHORT"]: "<|short|>",
    TOTAL_TAG_LENGTH["LONG"]: "<|long|>",
    TOTAL_TAG_LENGTH["VERY_LONG"]: "<|very_long|>",
}
PROCESSING_TIMING = {
    "BEFORE": "Before applying other prompt processings",
    "AFTER": "After applying other prompt processings",
}
DEFAULT_FORMAT = """<|special|>, 
<|characters|>, <|copyrights|>, 
<|artist|>, 

<|general|>, 

<|quality|>, <|meta|>, <|rating|>"""
TIMING_INFO_TEMPLATE = (
    "_Prompt upsampling will be applied to {} "
    "sd-dynamic-promps and the webui's styles feature are applied_"
)


def on_process_timing_dropdown_changed(timing: str):
    info = ""
    if timing == PROCESSING_TIMING["BEFORE"]:
        info = "**only the first image in batch**, **before**"
    elif timing == PROCESSING_TIMING["AFTER"]:
        info = "**all images in batch**, **after**"
    else:
        raise Exception(f"Unknown timing: {timing}")
    return TIMING_INFO_TEMPLATE.format(info)


class DTGScript(scripts.Script):
    def __init__(self):
        super().__init__()
        self.prompt_area = [None, None]
        self.on_after_component_elem_id = [
            ("txt2img_prompt", lambda x: self.set_prompt_area(0, x)),
            ("img2img_prompt", lambda x: self.set_prompt_area(1, x)),
        ]

    def set_prompt_area(self, i2i: int, component: OnComponent):
        self.prompt_area[i2i] = component.component

    def title(self):
        return "DanTagGen"

    def show(self, _):
        return scripts.AlwaysVisible

    def after_component(self, component, **kwargs):
        val = kwargs.get("value", "")
        elem_id = kwargs.get("elem_id", "") or ""
        if elem_id == "txt2img_prompt":
            self.orig_prompt_area[0] = component
        elif elem_id == "img2img_prompt":
            self.orig_prompt_area[1] = component

    def ui(self, is_img2img):
        with gr.Accordion(open=False, label=self.title()) as dtg_acc:
            with gr.Column():
                enabled_check = gr.Checkbox(label="Enabled", value=False)

                tag_length_radio = gr.Radio(
                    label="Total tag length",
                    choices=list(TOTAL_TAG_LENGTH.values()),
                    value=TOTAL_TAG_LENGTH["LONG"],
                )
                ban_tags_textbox = gr.Textbox(
                    label="Ban tags",
                    info="Separate with comma. Regex supported.",
                    value="",
                    placeholder="umbrella, official.*, .*text, ...",
                )
                format_textarea = gr.TextArea(
                    label="Prompt Format",
                    info="The format you want to apply to final prompt",
                    value=DEFAULT_FORMAT,
                )

                with gr.Group():
                    with gr.Row():
                        seed_num_input = gr.Number(
                            label="Seed for upsampling tags",
                            minimum=-1,
                            maximum=2**31 - 1,
                            step=1,
                            scale=4,
                            value=-1,
                        )
                        seed_random_btn = gr.Button(value="Randomize")
                        seed_shuffle_btn = gr.Button(value="Shuffle")

                        seed_random_btn.click(lambda: -1, outputs=[seed_num_input])
                        seed_shuffle_btn.click(
                            lambda: random.randint(0, 2**31-1), outputs=[seed_num_input]
                        )

                with gr.Group():
                    process_timing_dropdown = gr.Dropdown(
                        label="Upsampling timing",
                        choices=list(PROCESSING_TIMING.values()),
                        value=PROCESSING_TIMING["AFTER"],
                    )

                    process_timing_md = gr.Markdown(
                        on_process_timing_dropdown_changed(
                            process_timing_dropdown.value
                        )
                    )

                    process_timing_dropdown.change(
                        on_process_timing_dropdown_changed,
                        inputs=[process_timing_dropdown],
                        outputs=[process_timing_md],
                    )

                with gr.Accordion(label="Generation config", open=False):
                    temperature_slider = gr.Slider(
                        label="Temperature",
                        info="← less random | more random →",
                        maximum=2.5,
                        minimum=0.1,
                        step=0.05,
                        value=1.35,
                    )

        self.infotext_fields = [
            (dtg_acc, lambda d: gr.update(open="DTG Parameters" in d)),
            (
                self.prompt_area[is_img2img],
                lambda d: d.get("DTG prompt", ""),
            ),
            (enabled_check, lambda d: "DTG Parameters" in d),
            (seed_num_input, lambda d: self.get_infotext(d, "seed", -1)),
            (tag_length_radio, lambda d: self.get_infotext(d, "tag_length", "long")),
            (ban_tags_textbox, lambda d: self.get_infotext(d, "ban_tags", "")),
            (format_textarea, lambda d: d.get("DTG format", DEFAULT_FORMAT)),
            (
                process_timing_dropdown,
                lambda d: PROCESSING_TIMING[self.get_infotext(d, "timing", "AFTER")],
            ),
            (temperature_slider, lambda d: self.get_infotext(d, "temperature", 1.35)),
        ]

        return [
            enabled_check,
            process_timing_dropdown,
            seed_num_input,
            tag_length_radio,
            ban_tags_textbox,
            format_textarea,
            temperature_slider,
        ]

    def get_infotext(self, d, target, default):
        return d.get("DanTagGen", {}).get(target, default)

    def write_infotext(
        self,
        p: StableDiffusionProcessingTxt2Img | StableDiffusionProcessingImg2Img,
        prompt: str,
        process_timing: str,
        seed: int,
        *args,
    ):
        p.extra_generation_params["DTG Parameters"] = json.dumps(
            {
                "seed": seed,
                "timing": process_timing,
                "tag_length": args[0],
                "ban_tags": args[1],
                "temperature": args[3],
            },
            ensure_ascii=False,
        ).translate(QUOTESWAP)
        p.extra_generation_params["DTG prompt"] = prompt
        if args[2] != DEFAULT_FORMAT:
            p.extra_generation_params["DTG format"] = args[2]

    def process(
        self,
        p: StableDiffusionProcessingTxt2Img | StableDiffusionProcessingImg2Img,
        is_enabled: bool,
        process_timing: str,
        seed: int,
        *args,
    ):
        """This method will be called after sd-dynamic-prompts and the styles are applied."""

        if not is_enabled:
            return

        if process_timing != PROCESSING_TIMING["AFTER"]:
            return
 
        self.original_prompt = p.all_prompts
        self.original_hr_prompt = p.all_hr_prompts
        aspect_ratio = p.width / p.height
        if seed == -1:
            seed = random.randrange(2**31 - 1)
        seed = int(seed)

        self.write_infotext(p, p.prompt, "AFTER", seed, *args)

        if torch.cuda.is_available() and isinstance(text_model, torch.nn.Module):
            text_model.cuda()
        new_all_prompts = []
        for prompt, sub_seed in zip(p.all_prompts, p.all_seeds):
            new_all_prompts.append(
                self._process(prompt, aspect_ratio, seed + sub_seed, *args)
            )

        hr_fix_enabled = getattr(p, "enable_hr", False)

        if hr_fix_enabled:
            if p.hr_prompt != p.prompt:
                new_hr_prompts = []
                for prompt, hr_prompt in zip(p.all_prompts, p.all_hr_prompts):
                    if prompt == hr_prompt:
                        new_hr_prompts.append(prompt)
                    else:
                        new_hr_prompts.append(hr_prompt)
                p.all_hr_prompts = new_hr_prompts
            else:
                p.all_hr_prompts = new_all_prompts
        p.all_prompts = new_all_prompts

        if torch.cuda.is_available() and isinstance(text_model, torch.nn.Module):
            text_model.cpu()
            torch.cuda.empty_cache()

    def before_process(
        self,
        p: StableDiffusionProcessingTxt2Img | StableDiffusionProcessingImg2Img,
        is_enabled: bool,
        process_timing: str,
        seed: int,
        *args,
    ):
        """This method will be called before sd-dynamic-prompts and the styles are applied."""

        if not is_enabled:
            return

        if process_timing != PROCESSING_TIMING["BEFORE"]:
            return

        self.original_prompt = p.prompt
        self.original_hr_prompt = p.hr_prompt
        aspect_ratio = p.width / p.height
        if seed == -1:
            seed = random.randrange(4294967294)
        self.write_infotext(p, p.prompt, "BEFORE", seed, *args)
        seed = int(seed + p.seed)

        if torch.cuda.is_available() and isinstance(text_model, torch.nn.Module):
            text_model.cuda()
        p.prompt = self._process(p.prompt, aspect_ratio, seed, *args)
        if torch.cuda.is_available() and isinstance(text_model, torch.nn.Module):
            text_model.cpu()
            torch.cuda.empty_cache()

    @lru_cache(128)
    def _process(
        self,
        prompt: str,
        aspect_ratio: float,
        seed: int,
        tag_length: str,
        ban_tags: str,
        format: str,
        temperature: float,
    ):
        propmt_preview = prompt.replace("\n", " ")[:40]
        logger.info(f"Processing propmt: {propmt_preview}...")
        logger.info(f"Processing with seed: {seed}")
        prompt_without_extranet, res = parse_prompt(prompt)
        prompt_parse_strength = parse_prompt_attention(prompt_without_extranet)

        rebuild_extranet = ""
        for name, params in res.items():
            for param in params:
                items = ":".join(param.items)
                rebuild_extranet += f" <{name}:{items}>"

        black_list = [tag.strip() for tag in ban_tags.split(",") if tag.strip()]
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
        len_target = TARGET[tag_length]

        tag_map = seperate_tags(all_tags)
        dtg_prompt = apply_dtg_prompt(tag_map, tag_length, aspect_ratio)
        for llm_gen, extra_tokens in tag_gen(
            text_model,
            tokenizer,
            dtg_prompt,
            tag_map["special"] + tag_map["general"],
            len_target,
            black_list,
            temperature=temperature,
            top_p=0.95,
            top_k=100,
            max_new_tokens=256,
            max_retry=5,
            seed=seed % SEED_MAX,
        ):
            pass
        tag_map["general"] += extra_tokens
        for cate in tag_map.keys():
            new_list = []
            for tag in tag_map[cate]:
                tag = tag.replace("(", "\(").replace(")", "\)")
                if tag in strength_map:
                    new_list.append(f"({tag}:{strength_map[tag]})")
                else:
                    new_list.append(tag)
            tag_map[cate] = new_list
        prompt_by_dtg = apply_format(tag_map, format)

        logger.info("Prompt processing done.")
        return prompt_by_dtg + "\n" + rebuild_extranet


def pares_infotext(_, params):
    try:
        params["DTG Parameters"] = json.loads(params["DTG Parameters"].translate(QUOTESWAP))
    except Exception:
        pass


scripts.script_callbacks.on_infotext_pasted(pares_infotext)

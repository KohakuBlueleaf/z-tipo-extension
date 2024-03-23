import os
import random
from functools import lru_cache

import torch
import gradio as gr
from transformers import set_seed
from transformers import LlamaForCausalLM, LlamaTokenizer

import modules.scripts as scripts
from modules.scripts import basedir
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
except:
    logger.warning("Llama-cpp-python/gguf model not found, using transformers to load model")

    text_model = LlamaForCausalLM.from_pretrained(
        "KBlueLeaf/DanTagGen-beta"
    ).eval().half()
tokenizer = LlamaTokenizer.from_pretrained("KBlueLeaf/DanTagGen-beta")


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


class DTGScript(scripts.Script):
    def __init__(self):
        super().__init__()

    def title(self):
        return "DanTagGen"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(open=False, label=self.title()):
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
                    value="""<|special|>, 
<|characters|>, <|copyrights|>, 
<|artist|>, 

<|general|>, 

<|quality|>, <|meta|>, <|rating|>""",
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

                        def click_random_seed_btn():
                            return random.randint(0, 2**31 - 1)

                        seed_random_btn.click(
                            click_random_seed_btn, outputs=[seed_num_input]
                        )

                        def click_shuffle_seed_btn():
                            return -1

                        seed_shuffle_btn.click(
                            click_shuffle_seed_btn, outputs=[seed_num_input]
                        )

                with gr.Group():
                    process_timing_dropdown = gr.Dropdown(
                        label="Upsampling timing",
                        choices=list(PROCESSING_TIMING.values()),
                        value=PROCESSING_TIMING["AFTER"],
                    )

                    def on_process_timing_dropdown_changed(timing: str):
                        if timing == PROCESSING_TIMING["BEFORE"]:
                            return "_Prompt upsampling will be applied to **only the first image in batch**, **before** sd-dynamic-promps and the webui's styles feature are applied_"
                        elif timing == PROCESSING_TIMING["AFTER"]:
                            return "_Prompt upsampling will be applied to **all images in batch**, **after** sd-dynamic-promps and the webui's styles feature are applied_"
                        raise Exception(f"Unknown timing: {timing}")

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

        return [
            enabled_check,
            process_timing_dropdown,
            seed_num_input,
            tag_length_radio,
            ban_tags_textbox,
            format_textarea,
            temperature_slider,
        ]

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

        aspect_ratio = p.width / p.height
        if seed == -1:
            seed = random.randrange(4294967294)

        if torch.cuda.is_available() and isinstance(text_model, torch.nn.Module):
            text_model.cuda()
        new_all_prompts = []
        for prompt in p.all_prompts:
            new_all_prompts.append(self._process(prompt, aspect_ratio, seed, *args))

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

        aspect_ratio = p.width / p.height
        if seed == -1:
            seed = random.randrange(4294967294)

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
        set_seed(seed)
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
        return prompt_by_dtg + "\n" + rebuild_extranet

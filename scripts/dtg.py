import os
import itertools
import json
import pathlib
import random
from functools import lru_cache

import torch
import gradio as gr

import modules.scripts as scripts
from modules import devices
from modules.scripts import basedir, OnComponent
from modules.processing import (
    StableDiffusionProcessingTxt2Img,
    StableDiffusionProcessingImg2Img,
)
from modules.prompt_parser import parse_prompt_attention
from modules.extra_networks import parse_prompt

import kgen.models as models
from kgen.executor.dtg import apply_dtg_prompt, tag_gen
from kgen.formatter import seperate_tags, apply_format
from kgen.metainfo import TARGET
from kgen.logging import logger


ext_dir = basedir()
models.model_dir = pathlib.Path(ext_dir) / "models"


SEED_MAX = 2**31 - 1
QUOTESWAP = str.maketrans("'\"", "\"'")
TOTAL_TAG_LENGTH = {
    "VERY_SHORT": "very short",
    "SHORT": "short",
    "LONG": "long",
    "VERY_LONG": "very long",
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
INFOTEXT_KEY = "DTG Parameters"
INFOTEXT_KEY_PROMPT = "DTG prompt"
INFOTEXT_KEY_FORMAT = "DTG format"

PROMPT_INDICATE_HTML = """
<div style="height: 100%; width: 100%; display: flex; justify-content: center; align-items: center">
    <span>
        Original Prompt Loaded.<br>
        Click "Apply" to apply the original prompt.
    </span>
</div>
"""
RECOMMEND_MARKDOWN = """
### Recommended Model and Settings:
- Model: DanTagGen-delta-rev2
    - gguf quant: Q6 or Q8
    - gguf device: cpu (cuda have reproducibility issue)
- Settings:
    - Temperature: 0.8~1.2
    - Top P: 0.75~0.9
    - Top K: 50 ~ 90
"""


def on_process_timing_dropdown_changed(timing: str):
    info = ""
    if timing == PROCESSING_TIMING["BEFORE"]:
        info = "**only the first image in batch**, **before**"
    elif timing == PROCESSING_TIMING["AFTER"]:
        info = "**all images in batch**, **after**"
    else:
        raise ValueError(f"Unknown timing: {timing}")
    return TIMING_INFO_TEMPLATE.format(info)


class DTGScript(scripts.Script):
    def __init__(self):
        super().__init__()
        self.prompt_area = [None, None]
        self.current_model = None
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

    def ui(self, is_img2img):
        with gr.Accordion(open=False, label=self.title()) as dtg_acc:
            with gr.Column():
                with gr.Row():
                    with gr.Column(scale=1):
                        enabled_check = gr.Checkbox(
                            label="Enabled", value=False, min_width=20
                        )
                        read_orig_prompt_btn = gr.Button(
                            size="sm",
                            value="Apply original prompt",
                            visible=False,
                            min_width=20,
                        )
                    with gr.Column(scale=3):
                        orig_prompt_area = gr.TextArea(visible=False)
                        orig_prompt_light = gr.HTML("")
                    orig_prompt_area.change(
                        lambda x: PROMPT_INDICATE_HTML * bool(x),
                        inputs=orig_prompt_area,
                        outputs=orig_prompt_light,
                    )
                    orig_prompt_area.change(
                        lambda x: gr.update(visible=bool(x)),
                        inputs=orig_prompt_area,
                        outputs=read_orig_prompt_btn,
                    )
                    read_orig_prompt_btn.click(
                        fn=lambda x: x,
                        inputs=[orig_prompt_area],
                        outputs=self.prompt_area[is_img2img],
                    )

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
                            lambda: random.randint(0, 2**31 - 1),
                            outputs=[seed_num_input],
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
                    gr.Markdown(RECOMMEND_MARKDOWN)
                    model_dropdown = gr.Dropdown(
                        label="Model",
                        choices=models.model_list
                        + [
                            f"{model} | {file}"
                            for model, file in itertools.product(
                                models.model_list, models.gguf_name
                            )
                        ],
                        value=models.model_list[0],
                    )
                    gguf_use_cpu = gr.Checkbox(label="Use CPU (GGUF)")
                    no_formatting = gr.Checkbox(label="No formatting", value=False)
                    temperature_slider = gr.Slider(
                        label="Temperature",
                        info="← less random | more random →",
                        maximum=2.5,
                        minimum=0.1,
                        step=0.05,
                        value=1.35,
                    )
                    top_p_slider = gr.Slider(
                        label="Top-p",
                        info="← less unconfident tokens | more unconfident tokens →",
                        maximum=1,
                        minimum=0,
                        step=0.05,
                        value=0.95,
                    )
                    top_k_slider = gr.Slider(
                        label="Top-k",
                        info="← less unconfident tokens | more unconfident tokens →",
                        maximum=200,
                        minimum=0,
                        step=1,
                        value=100,
                    )

        self.infotext_fields = [
            (dtg_acc, lambda d: gr.update(open=INFOTEXT_KEY in d)),
            (
                self.prompt_area[is_img2img],
                lambda d: d.get(INFOTEXT_KEY_PROMPT, d["Prompt"]),
            ),
            (orig_prompt_area, lambda d: d["Prompt"]),
            (enabled_check, lambda d: INFOTEXT_KEY in d),
            (seed_num_input, lambda d: self.get_infotext(d, "seed", None)),
            (tag_length_radio, lambda d: self.get_infotext(d, "tag_length", None)),
            (ban_tags_textbox, lambda d: self.get_infotext(d, "ban_tags", None)),
            (format_textarea, lambda d: d.get(INFOTEXT_KEY_FORMAT, None)),
            (
                process_timing_dropdown,
                lambda d: PROCESSING_TIMING.get(
                    self.get_infotext(d, "timing", None), None
                ),
            ),
            (temperature_slider, lambda d: self.get_infotext(d, "temperature", None)),
            (top_p_slider, lambda d: self.get_infotext(d, "top_p", None)),
            (top_k_slider, lambda d: self.get_infotext(d, "top_k", None)),
            (
                model_dropdown,
                lambda d: self.get_infotext(d, "model", None),
            ),
            (gguf_use_cpu, lambda d: self.get_infotext(d, "gguf_cpu", None)),
            (no_formatting, lambda d: self.get_infotext(d, "no_formatting", None)),
        ]

        return [
            enabled_check,
            process_timing_dropdown,
            seed_num_input,
            tag_length_radio,
            ban_tags_textbox,
            format_textarea,
            temperature_slider,
            top_p_slider,
            top_k_slider,
            model_dropdown,
            gguf_use_cpu,
            no_formatting,
        ]

    def get_infotext(self, d, target, default):
        return d.get(INFOTEXT_KEY, {}).get(target, default)

    def write_infotext(
        self,
        p: StableDiffusionProcessingTxt2Img | StableDiffusionProcessingImg2Img,
        prompt: str,
        process_timing: str,
        seed: int,
        *args,
    ):
        p.extra_generation_params[INFOTEXT_KEY] = json.dumps(
            {
                "seed": seed,
                "timing": process_timing,
                "tag_length": args[0],
                "ban_tags": args[1],
                "temperature": args[3],
                "top_p": args[4],
                "top_k": args[5],
                "model": args[6],
                "gguf_cpu": args[7],
                "no_formatting": args[8],
            },
            ensure_ascii=False,
        ).translate(QUOTESWAP)
        p.extra_generation_params[INFOTEXT_KEY_PROMPT] = prompt
        if args[2] != DEFAULT_FORMAT:
            p.extra_generation_params[INFOTEXT_KEY_FORMAT] = args[2]

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
        self.original_hr_prompt = getattr(p, "all_hr_prompts", None)
        aspect_ratio = p.width / p.height
        if seed == -1:
            seed = random.randrange(2**31 - 1)
        seed = int(seed)

        self.write_infotext(p, p.prompt, "AFTER", seed, *args)

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

        p.prompt = self._process(p.prompt, aspect_ratio, seed, *args)

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
        top_p: float,
        top_k: int,
        model: str,
        gguf_use_cpu: bool,
        no_formatting: bool,
    ):
        if model != self.current_model:
            if " | " in model:
                model_name, gguf_name = model.split(" | ")
                target_file = f"{model_name.split('/')[-1]}_{gguf_name}"
                if str(models.model_dir / target_file) not in models.list_gguf():
                    models.download_gguf(model_name, gguf_name)
                target = os.path.join(str(models.model_dir), target_file)
                gguf = True
            else:
                target = model
                gguf = False
            models.load_model(target, gguf, device="cpu" if gguf_use_cpu else "cuda")
            self.current_model = model
        prompt_preview = prompt.replace("\n", " ")[:40]
        logger.info(f"Processing prompt: {prompt_preview}...")
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
        break_map = set()
        for part, strength in prompt_parse_strength:
            part_tags = [tag.strip() for tag in part.strip().split(",") if tag.strip()]
            if part == "BREAK" and strength == -1:
                break_map.add(all_tags[-1])
                continue
            all_tags.extend(part_tags)
            if strength == 1:
                continue
            for tag in part_tags:
                strength_map[tag] = strength

        tag_length = tag_length.replace(" ", "_")
        len_target = TARGET[tag_length]

        tag_map = seperate_tags(all_tags)
        dtg_prompt = apply_dtg_prompt(tag_map, tag_length, aspect_ratio)

        if isinstance(models.text_model, torch.nn.Module):
            models.text_model.to(devices.device)
        for current in tag_gen(
            models.text_model,
            models.tokenizer,
            dtg_prompt,
            tag_map["special"] + tag_map["general"],
            len_target,
            black_list,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=512,
            max_retry=20,
            max_same_output=15,
            seed=seed % SEED_MAX,
        ):
            _, extra_tokens, iter_count = current
        if isinstance(models.text_model, torch.nn.Module):
            models.text_model.cpu()
            devices.torch_gc()
        tag_map["general"] += extra_tokens
        logger.info(
            f"Total general tags: {len(tag_map['general']+tag_map['special'])} | "
            f"Total iterations: {iter_count}"
        )
        if no_formatting:
            result = prompt + ", " + ", ".join(extra_tokens)
        else:
            for cate in tag_map.keys():
                new_list = []
                for tag in tag_map[cate]:
                    tag = tag.replace("(", "\(").replace(")", "\)")
                    if tag in strength_map:
                        new_list.append(f"({tag}:{strength_map[tag]})")
                    else:
                        new_list.append(tag)
                    if tag in break_map:
                        new_list.append("BREAK")
                tag_map[cate] = new_list
            prompt_by_dtg = apply_format(tag_map, format).replace("BREAK,", "BREAK")
            result = prompt_by_dtg + "\n" + rebuild_extranet

        logger.info("Prompt processing done.")
        return result


def pares_infotext(_, params):
    try:
        params[INFOTEXT_KEY] = json.loads(params[INFOTEXT_KEY].translate(QUOTESWAP))
    except Exception:
        pass


scripts.script_callbacks.on_infotext_pasted(pares_infotext)

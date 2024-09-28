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
from modules.shared import opts

if hasattr(opts, "hypertile_enable_unet"):  # webui >= 1.7
    from modules.ui_components import InputAccordion
else:
    InputAccordion = None

import kgen.models as models
import kgen.executor.tipo as tipo
from kgen.executor.tipo import parse_tipo_request, tipo_runner
from kgen.formatter import seperate_tags, apply_format
from kgen.metainfo import TARGET, TIPO_DEFAULT_FORMAT
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
INFOTEXT_KEY = "TIPO Parameters"
INFOTEXT_KEY_PROMPT = "TIPO prompt"
INFOTEXT_NL_PROMPT = "TIPO nl prompt"
INFOTEXT_KEY_FORMAT = "TIPO format"

PROMPT_INDICATE_HTML = """
<div style="height: 100%; width: 100%; display: flex; justify-content: center; align-items: center">
    <span>
        Original Prompt Loaded.<br>
        Click "Apply" to apply the original prompt.
    </span>
</div>
"""
RECOMMEND_MARKDOWN = """
### Rcommended Model and Settings:

"""


def on_process_timing_dropdown_changed(timing: str):
    info = ""
    if timing == PROCESSING_TIMING["BEFORE"]:
        info = "**only the first image in batch**, **before**"
    elif timing == PROCESSING_TIMING["AFTER"]:
        info = "**all images in batch**, **after**"
    else:
        raise Exception(f"Unknown timing: {timing}")
    return TIMING_INFO_TEMPLATE.format(info)


class TIPOScript(scripts.Script):
    def __init__(self):
        super().__init__()
        self.prompt_area = [None, None, None, None]
        self.prompt_area_row = [None, None]
        self.current_model = None
        self.on_after_component_elem_id = [
            ("txt2img_prompt_row", lambda x: self.create_new_prompt_area(0)),
            ("txt2img_prompt", lambda x: self.set_prompt_area(0, x)),
            ("img2img_prompt_row", lambda x: self.create_new_prompt_area(1)),
            ("img2img_prompt", lambda x: self.set_prompt_area(1, x)),
        ]

    def create_new_prompt_area(self, i2i: int):
        self.prompt_area_row[i2i] = gr.Row()
        with self.prompt_area_row[i2i]:
            new_propmt_area = gr.Textbox(
                label="Natural Language Prompt",
                lines=3,
                placeholder="Natural Language Prompt for TIPO (Put Tags to Propmt region)",
            )
        self.prompt_area[i2i * 2 + 1] = new_propmt_area

    def set_prompt_area(self, i2i: int, component: OnComponent):
        self.prompt_area[i2i * 2] = component.component

    def title(self):
        return "TIPO"

    def show(self, _):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with self.prompt_area_row[is_img2img]:
            with (
                InputAccordion(False, open=False, label=self.title())
                if InputAccordion
                else gr.Accordion(open=False, label=self.title())
            ) as tipo_acc:
                with gr.Column():
                    with gr.Row():
                        with gr.Column(scale=1):
                            if InputAccordion is None:
                                enabled_check = gr.Checkbox(
                                    label="Enabled", value=False, min_width=20
                                )
                            else:
                                enabled_check = tipo_acc
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
                        label="Length target",
                        choices=list(TOTAL_TAG_LENGTH.values()),
                        value=TOTAL_TAG_LENGTH["LONG"],
                    )
                    ban_tags_textbox = gr.Textbox(
                        label="Ban tags",
                        info="Separate with comma. Regex supported.",
                        value="",
                        placeholder="umbrella, official.*, .*text, ...",
                    )
                    format_dropdown = gr.Dropdown(
                        label="Prompt Format",
                        info="The format you want to apply to final prompt",
                        choices=list(TIPO_DEFAULT_FORMAT.keys()) + ["custom"],
                        value="Both, tag first (recommend)",
                    )
                    format_textarea = gr.TextArea(
                        value=TIPO_DEFAULT_FORMAT["Both, tag first (recommend)"],
                        label="Custom Prompt Format",
                        visible=False,
                        placeholder="<|extended|>. <|general|>",
                    )
                    format_dropdown.change(
                        lambda x: gr.update(
                            visible=x == "custom",
                            value=TIPO_DEFAULT_FORMAT.get(
                                x, list(TIPO_DEFAULT_FORMAT.values())[0]
                            ),
                        ),
                        inputs=format_dropdown,
                        outputs=format_textarea,
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
                            choices=[i[0] for i in models.tipo_model_list]
                            + [
                                f"{model_name} | {file}"
                                for model_name, ggufs in models.tipo_model_list
                                for file in ggufs
                            ],
                            value=models.tipo_model_list[0][0],
                        )
                        gguf_use_cpu = gr.Checkbox(label="Use CPU (GGUF)")
                        no_formatting = gr.Checkbox(label="No formatting", value=False)
                        temperature_slider = gr.Slider(
                            label="Temperature",
                            info="← less random | more random →",
                            maximum=1.5,
                            minimum=0.1,
                            step=0.05,
                            value=0.5,
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
                            maximum=150,
                            minimum=0,
                            step=1,
                            value=80,
                        )

        self.infotext_fields = [
            (
                (tipo_acc, lambda d: gr.update(value=INFOTEXT_KEY in d))
                if InputAccordion
                else (tipo_acc, lambda d: gr.update(open=INFOTEXT_KEY in d))
            ),
            (
                self.prompt_area[is_img2img * 2],
                lambda d: d.get(INFOTEXT_KEY_PROMPT, d["Prompt"]),
            ),
            (
                self.prompt_area[is_img2img * 2 + 1],
                lambda d: d.get(INFOTEXT_NL_PROMPT, ""),
            ),
            (orig_prompt_area, lambda d: d["Prompt"]),
            (enabled_check, lambda d: INFOTEXT_KEY in d),
            (seed_num_input, lambda d: self.get_infotext(d, "seed", None)),
            (tag_length_radio, lambda d: self.get_infotext(d, "tag_length", None)),
            (ban_tags_textbox, lambda d: self.get_infotext(d, "ban_tags", None)),
            (format_dropdown, lambda d: self.get_infotext(d, "format", None)),
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
            format_dropdown,
            format_textarea,
            temperature_slider,
            top_p_slider,
            top_k_slider,
            model_dropdown,
            gguf_use_cpu,
            no_formatting,
            self.prompt_area[is_img2img * 2 + 1],
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
                "format": args[2],
                "temperature": args[4],
                "top_p": args[5],
                "top_k": args[6],
                "model": args[7],
                "gguf_cpu": args[8],
                "no_formatting": args[9],
            },
            ensure_ascii=False,
        ).translate(QUOTESWAP)
        p.extra_generation_params[INFOTEXT_KEY_PROMPT] = prompt
        p.extra_generation_params[INFOTEXT_NL_PROMPT] = args[-1]
        if args[3] != DEFAULT_FORMAT:
            p.extra_generation_params[INFOTEXT_KEY_FORMAT] = args[3]

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

        args = list(args)
        nl_prompt = args.pop()
        new_all_prompts = []
        for prompt, sub_seed in zip(p.all_prompts, p.all_seeds):
            new_all_prompts.append(
                self._process(prompt, nl_prompt, aspect_ratio, seed + sub_seed, *args)
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

        args = list(args)
        p.prompt = self._process(p.prompt, args.pop(), aspect_ratio, seed, *args)

    @lru_cache(128)
    def _process(
        self,
        prompt: str,
        nl_prompt: str,
        aspect_ratio: float,
        seed: int,
        tag_length: str,
        ban_tags: str,
        format_select: str,
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
        tipo.BAN_TAGS = black_list
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
        tag_map = seperate_tags(all_tags)

        meta, operations, general, nl_prompt = parse_tipo_request(
            tag_map,
            nl_prompt,
            tag_length_target=tag_length,
            generate_extra_nl_prompt=(not nl_prompt and "<|extended|>" in nl_prompt)
            or "<|generated|>" in nl_prompt,
        )
        meta["aspect_ratio"] = f"{aspect_ratio:.1f}"

        if isinstance(models.text_model, torch.nn.Module):
            models.text_model.to(devices.device)
        tag_map, _ = tipo_runner(
            meta,
            operations,
            general,
            nl_prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
        )
        if isinstance(models.text_model, torch.nn.Module):
            models.text_model.cpu()
            devices.torch_gc()
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
                if tag in break_map or org_tag in break_map:
                    new_list.append("BREAK")
            tag_map[cate] = new_list
        prompt_by_tipo = apply_format(tag_map, format).replace("BREAK,", "BREAK")
        result = prompt_by_tipo + "\n" + rebuild_extranet

        logger.info("Prompt processing done.")
        return result


def pares_infotext(_, params):
    try:
        params["TIPO Parameters"] = json.loads(
            params["TIPO Parameters"].translate(QUOTESWAP)
        )
    except Exception:
        pass


scripts.script_callbacks.on_infotext_pasted(pares_infotext)

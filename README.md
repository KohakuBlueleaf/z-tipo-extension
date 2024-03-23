# A1111-SD-WebUI-DTG

A sd-webui extension for utilizing DanTagGen to "upsample prompts".

It can generate the detail tags/core tags about the character you put in the prompts. It can also add some extra elements into your prompt.


## What is DanTagGen

DanTagGen(Danbooru Tag Generator) is a LLM model designed for generating Danboou Tags with provided informations.
It aims to provide user a more convinient way to make prompts for Text2Image model which is trained on Danbooru datasets.

More information about model arch and training data can be found in the HuggingFace Model card:

[KBlueLeaf/DanTagGen-beta · Hugging Face](https://huggingface.co/KBlueLeaf/DanTagGen-beta)


## How to use it

After install it into the sd-webui or sd-webui-forge. Just enable it in the acordion. It will automatically read the content of your prompt and generate more tags based on your prompt.

### Options

* tag length:
  * very short: around 10 tags
  * short: around 20 tags
  * long: around 40 tags
  * very long: around 60 tags
  * ***short or long is recommended***
* Ban tags: The black list of tags you don't want to see in final prompt. Regex supported.
* Prompt Format: The format of final prompt. Default value is the recommended format of [Kohaku XL Delta](https://civitai.com/models/332076/kohaku-xl-delta)
* Seed: the seed of prompt generator. Since we use temperature/top k/top p sampling, so it is not deterministic unless you use same seed. -1 for random seed.
* Upsampling timing:
  * After: after other prompt processings, for example: after dynamic prompts/wildcard.
  * Before: Before other prompt processings.
* Temperature: Higher = more dynamic result, Lower = better coherence between tags.


## Faster inference

If you think the transformers implementation is slow and want to get better speed. You can install `llama-cpp-python` by yourself and then download the gguf model from HuggingFace and them put them into the `models` folder.

(Automatic installation/download script for llama-cpp-python and gguf model are WIP)

More information about `llama-cpp-python`:

* [abetlen/llama-cpp-python: Python bindings for llama.cpp (github.com)](https://github.com/abetlen/llama-cpp-python)
* [jllllll/llama-cpp-python-cuBLAS-wheels: Wheels for llama-cpp-python compiled with cuBLAS support (github.com)](https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels)

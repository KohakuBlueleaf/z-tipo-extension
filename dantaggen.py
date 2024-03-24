import gradio_client, pandas as pd, requests, io, random
CL = gradio_client.Client("KBlueLeaf/DTG-demo")
from pathlib import Path
# todo, make this better
if Path(".tags.csv").exists():
    csv = Path(".tags.csv").read_text()
else:
    csv = requests.get("https://huggingface.co/SmilingWolf/wd-vit-tagger-v3/resolve/main/selected_tags.csv").text
    Path(".tags.csv").write_text(csv)

TAGS = pd.read_csv(io.StringIO(csv))
class DanTagGen:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "blacklist": ("STRING", {"multiline": True}),
                "length": (["very_short", "short", "long", "very_long"], {"default": "long"}),
                "width": ("INT", {
                    "min": 64,
                    "max": 2048,
                    "default": 1024
                }),
                "height": ("INT", {
                    "min": 64,
                    "max": 2048,
                    "default": 1024
                }),
                "temp": ("FLOAT", {
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "default": 0.75
                }),
                "escape_bracket": (["enable", "disable"], {"default": "enable"}),
                "model": (["alpha", "beta"], {"default": "beta"}),
                "rating": (["safe", "sensitive", "nsfw", "nsfw, explicit"], {"default": "safe"}),
                "regenerate": (["enable", "disable", "plap"], {"default": "enable"})
            }
        }

    RETURN_TYPES = ("STRING",)

    FUNCTION = "predict"

    CATEGORY = "DTG"
    def predict(self, prompt, blacklist, length, width, height, temp, escape_bracket, model, rating, regenerate):
        artists, characters, general = "", "", ""
        if "," in prompt:
            prmpt = prompt.replace(", ", ",")
            prmpt = prmpt.replace(" ", "_")
            prmpt = prmpt.replace(",", " ")
        else:
            prmpt = prompt.replace("-", "_")
        for tag in prmpt.split(" "):
            if TAGS["name"].str.contains(tag).any():
                category = TAGS["category"][TAGS["name"] == tag].values
                if category.shape[0] != 0:
                    category = category[0]
                if str(category) == "0":
                    general += " " + tag
                elif str(category) == "4":
                    characters += " " + tag
                elif str(category) == "9":
                    artists += " " + tag
            else:
                general += " " + tag 
        general = general.strip(" _")
        characters = characters.strip(" _")
        artists = artists.strip(" _")
        specials = []
        for tag in general.split(" "):
            if tag in ['1girl', '2girls', '3girls', '4girls', '5girls', '6+girls', 'multiple_girls', '1boy', '2boys', '3boys', '4boys', '5boys', '6+boys', 'multiple_boys', 'male_focus', '1other', '2others', '3others', '4others', '5others', '6+others', 'multiple_others']:
                specials.append(tag)
        result = CL.predict(
            "KBlueLeaf/DanTagGen-" + model,
            rating,
            artists,
            characters,
            '',
            length,
            specials,
            prmpt,
            width,
            height,
            blacklist,
            escape_bracket,	
            temp,
            api_name="/wrapper"
        )[0]
        result = result.replace("\n", " ")
        result = result.replace("  ", " ")
        return (result,)
    
    @classmethod
    def IS_CHANGED(self, prompt, blacklist, length, width, height, temp, escape_bracket, model, rating, regenerate):
        if regenerate == "plap":
            return random.randint(1, 200) * regenerate
        return random.randint(1, 2675376) * bool(regenerate) 

# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "DTG": DanTagGen
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DTG": "Danbooru Tag Generator (HF Space)"
}

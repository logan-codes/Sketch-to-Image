from dotenv import load_dotenv
load_dotenv()
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from PIL import Image

cn=ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-scribble",
    torch_dtype=torch.float16
    )

pipe= StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=cn,
    torch_dtype=torch.float16,
    safety_checker=None 
)

pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config
)
pipe.enable_attention_slicing()
pipe.enable_model_cpu_offload()

sk=Image.open("house.png").convert("RGB")
sk=sk.resize((512,512))

img=pipe(
    prompt="an house beside the hills on a sunny morning",
    negative_prompt="blurry, bad quality, distorted, ugly",
    image=sk,
    num_inference_steps=20,
    guidance_scale=7.5,
    controlnet_conditioning_scale=1.0
).images[0]

img.save("output.png")
print("done!")
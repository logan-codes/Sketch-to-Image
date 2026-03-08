#from dotenv import load_dotenv
# load_dotenv()
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from PIL import Image
import gradio as gr
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

logger.info(f"Device: {device}")

logger.info("Loading Controlnet...")
cn= ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-scribble",
    torch_dtype=dtype
)
logger.info("Controlnet Loaded.")

logger.info("Loading Stable Diffusion Controlnet Pipeline...")
pipe= StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=cn,
    torch_dtype=dtype,
    safety_checker=None
)
logger.info("Stable Diffusion Controlnet Pipeline Loaded.")

pipe.scheduler= UniPCMultistepScheduler.from_config(
    pipe.scheduler.config
)
if torch.cuda.is_available():
    pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload()
else:
    pipe= pipe.to(device) 
logger.info("Config setup completed")

def generate(sketch_data, prompt, num_inference, guidance):
    logger.info("Image generation requested")
    if sketch_data is None:
        logger.warning("user tried generating without sketch")
        raise gr.Error("Draw an sketch to generate an image.")
    if not prompt.strip():
        logger.warning("user tried generating without prompt")
        raise gr.Error("Enter an prompt to generate an image.")
    
    if isinstance(sketch_data,dict):
        sketch_data=sketch_data.get("composite",sketch_data.get("image"))

    logger.info(f"Prompt: {prompt}")
    logger.info(f"Steps: {num_inference}, Guidance: {guidance}")
    sk= Image.fromarray(sketch_data.astype(np.uint8)).convert("RGB")
    sk = sk.resize((512,512))

    try:
        op = pipe(
            prompt=prompt+", high quality, detailed and 4K",
            negative_prompt="blurry, bad anatomy, worst quality, low quality, ugly",
            guidance_scale=guidance,
            num_inference_steps=int(num_inference),
            image=sk,
            controlnet_conditioning_scale=1.0
        ).images[0]
        logger.info("Image generated successfully")
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise
    return op

css = """
#title { text-align: center; margin-bottom: 4px; }
#subtitle { text-align: center; color: #666; margin-bottom: 20px; font-size: 0.95em; }
#generate-btn { width: 100%; margin-top: 12px; }
footer { display: none !important; }
"""

with gr.Blocks(css=css, theme=gr.themes.Soft(), title="Sketch to Image") as demo:
    gr.Markdown(" Sketch to Image", elem_id="title")
    gr.Markdown(
        "Draw an sketch and describe it, and watch AI bring it to life using Controlnet + Stable Diffusion",
        elem_id="subtitle"
    )
    with gr.Column(scale=1) as col:
        sketch_input = gr.Sketchpad(
            label="Your Sketch",
            type="numpy",
            canvas_size=(512, 512),
        )
        prompt_input = gr.Textbox(
            label="Describe your sketch",
            placeholder="e.g. a cozy cabin in the forest at sunset, realistic",
            lines=2
        )

        with gr.Accordion("⚙️ Advanced Settings", open=False):
            steps_slider = gr.Slider(
                minimum=10, maximum=50, value=20, step=1,
                label="Inference Steps",
                info="More steps = better quality but slower"
            )
            guidance_slider = gr.Slider(
                minimum=1.0, maximum=15.0, value=7.5, step=0.5,
                label="Guidance Scale",
                info="How strictly to follow your prompt"
            )

        generate_btn = gr.Button(
            "Generate 🎨", variant="primary", elem_id="generate-btn"
        )

        with gr.Column(scale=1):
            output_image = gr.Image(
                label="Generated Image",
                type="pil",
                height=512
            )
            gr.Markdown(
                "💡 **Tips:** Keep sketches simple and clear. "
                "Detailed prompts give better results.",
            )

    generate_btn.click(
        fn=generate,
        inputs=[sketch_input, prompt_input, steps_slider, guidance_slider],
        outputs=output_image
    )

demo.launch(server_name="0.0.0.0", server_port=7860)

---
title: Sketch To Image
emoji: 😻
colorFrom: yellow
colorTo: blue
sdk: docker
pinned: false
---

# 🎨 Sketch to Image AI

An interactive web application that transforms your rough sketches into detailed, high-quality images using **Stable Diffusion v1.5** and **ControlNet (Scribble)**.

## ✨ Features

- **Live Sketchpad:** Draw directly in the browser using the integrated Gradio Sketchpad.
- **Prompt-to-Image:** Describe your sketch with natural language to guide the AI's generation.
- **Advanced Controls:** Fine-tune the output with adjustable inference steps and guidance scales.
- **Optimized Performance:** Uses CPU offloading and attention slicing to run efficiently.

## 🚀 How it Works

1. **ControlNet (Scribble):** Processes your hand-drawn sketch as a structural guide.
2. **Stable Diffusion v1.5:** Generates a realistic or artistic image based on your prompt, while adhering to the structure provided by the sketch.
3. **UniPCMultistepScheduler:** Ensures fast and high-quality image sampling.

## 🛠️ Tech Stack

- **Framework:** [Gradio](https://gradio.app/)
- **Model:** [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- **Control:** [lllyasviel/sd-controlnet-scribble](https://huggingface.co/lllyasviel/sd-controlnet-scribble)
- **Library:** [Diffusers](https://github.com/huggingface/diffusers) by Hugging Face
- **Language:** Python

## 💻 Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Sketch-Image.git
   cd Sketch-Image
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python app.py
   ```

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Built with ❤️ by [logan-codes](https://github.com/logan-codes)
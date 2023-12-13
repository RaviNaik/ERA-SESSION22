import gradio as gr
import torch
from pathlib import Path

torch.set_float32_matmul_precision("high")

from generate.base import main


def generate(prompt, max_new_tokens, temperature, num_samples):
    prompt = prompt.strip()

    responses = main(
        prompt=prompt,
        checkpoint_dir=Path("out/redpajama"),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        num_samples=num_samples,
    )
    return {output: responses}


with gr.Blocks() as app:
    gr.Markdown("## ERA Session22 - Pythia-160M Pre-training with LitGPT")
    gr.Markdown(
        """This is an implementation of Pythia-160M using [LitGPT](https://github.com/Lightning-AI/lit-gpt) by LightningAI.
        
        Please find the source code and training details [here](https://github.com/RaviNaik/ERA-SESSION22).
        
        Dataset used to train: [RedPajama](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T).
        """
    )
    with gr.Row():
        with gr.Column():
            prompt_box = gr.Textbox(label="Initial Prompt", interactive=True)
            max_new_tokens = gr.Slider(
                minimum=10,
                maximum=200,
                value=50,
                step=10,
                label="Select Number of Tokens to be Generated",
                interactive=True,
            )
            temperature = gr.Slider(
                minimum=0.1,
                maximum=1,
                value=0.7,
                step=0.1,
                label="Select Temperature",
                interactive=True,
            )
            num_samples = gr.Dropdown(
                choices=[1, 2, 5, 10],
                value=1,
                interactive=True,
                label="Select No. of outputs to be generated",
            )
            submit_btn = gr.Button(value="Generate")

        with gr.Column():
            output = gr.JSON(label="Generated Text")

        submit_btn.click(
            generate,
            inputs=[prompt_box, max_new_tokens, temperature, num_samples],
            outputs=[output],
        )

app.launch()

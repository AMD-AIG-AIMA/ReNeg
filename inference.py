import os
from pathlib import Path
import torch
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
)
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
    )
    # parser.add_argument("--prompt_path", type=str, default="prompts/prompts.txt")
    parser.add_argument(
        "--prompt",
        type=str,
        default="A girl in a school uniform playing an electric guitar.",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="neg_emb",
        choices=["neg_emb", "neg_prompt", "only_pos"],
    )
    parser.add_argument(
        "--neg_prompt",
        type=str,
        default="distorted, ugly, blurry, low resolution, low quality, bad, deformed, disgusting, Overexposed, Simple background, Plain background, Grainy, Underexposed, too dark, too bright, too low contrast, too high contrast, Broken, Macabre, artifacts, oversaturated",
    )
    parser.add_argument(
        "--neg_embeddings_path",
        type=str,
        default="checkpoints/sd1.5_reneg_emb.bin",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs",
    )
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_path,
        safety_checker=None,
    )
    pipe.scheduler = DDIMScheduler.from_pretrained(
        args.model_path, subfolder="scheduler"
    )
    device = "cuda"
    pipe.to(device)
    generator = torch.Generator().manual_seed(args.seed)

    os.makedirs(args.output_path, exist_ok=True)
    if args.prompt_type == "neg_emb":
        neg_embeddings = torch.load(args.neg_embeddings_path).to(device)
        output = pipe(
            args.prompt,
            negative_prompt_embeds=neg_embeddings,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=7.5,
            generator=generator,
        )
    elif args.prompt_type == "neg_prompt":
        neg_prompt = args.neg_prompt
        output = pipe(
            args.prompt,
            negative_prompt=neg_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=7.5,
            generator=generator,
        )
    elif args.prompt_type == "only_pos":
        output = pipe(
            args.prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=7.5,
            generator=generator,
        )
    image = output.images[0]
    # TextToImageModel is the model you want to evaluate
    file_name = args.prompt.replace(" ", "_")
    output_file = Path(args.output_path) / f"{args.prompt_type}_{file_name}.jpg"
    image.save(output_file)
    print(f"Saved image to {output_file}")


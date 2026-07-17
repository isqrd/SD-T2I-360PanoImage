import argparse
import torch
from diffusers.utils import load_image
from img2panoimg import Image2360PanoramaImagePipeline

# Set up command line argument parsing
parser = argparse.ArgumentParser(description="Image to 360 Panorama Pipeline")
parser.add_argument("-i", "--image", type=str, default="./data/i2p-image.jpg",
                    help="Path or URL to the input source image")
parser.add_argument("-p", "--prompt", type=str,
                    default="Image of my messy desk and office, create additional detail and 360 pano of the room",
                    help="Prompt to guide the outpainting generation")
parser.add_argument("-o", "--output", type=str, default="result.png",
                    help="Path to save the generated equirectangular image")
args = parser.parse_args()

print(f"Loading input image: {args.image}")
image = load_image(args.image).resize((512, 512))
mask = load_image("./data/i2p-mask.jpg")

# set up inputs for 12GB GPU (upscale=False)
input = {'prompt': args.prompt, 'image': image, 'mask': mask, 'upscale': False}

print("Initializing Image to 360 Panorama Pipeline...")
model_id = 'models'
img2panoimg = Image2360PanoramaImagePipeline(model_id, torch_dtype=torch.float16)

print(f"Running generation with prompt: '{args.prompt}'...")
output = img2panoimg(input)

output.save(args.output)
print(f"Successfully saved generated 360 panorama to: {args.output}")

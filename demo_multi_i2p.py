import torch
from diffusers.utils import load_image
from img2panoimg import Image2360PanoramaImagePipeline

# Load FRONT image (face 0) and BACK image (face 1)
image1 = load_image("./data/i2p-image.jpg").resize((512, 512))
image2 = load_image("./data/the-times-square.png").resize((512, 512))

# Load mask for FRONT image
mask1 = load_image("./data/i2p-mask.jpg")

# Put them in lists (multi-image support!)
image_list = [image1, image2]
mask_list = [mask1]  # Second mask defaults to all-black (preserved/unmasked)

prompt = 'The futuristic neon office room with digital holographic elements'

# Set up inputs for 12GB GPU (upscale=False)
input = {'prompt': prompt, 'image': image_list, 'mask': mask_list, 'upscale': False}

model_id = 'models'
img2panoimg = Image2360PanoramaImagePipeline(model_id, torch_dtype=torch.float16)
output = img2panoimg(input)
output.save('result_multi.png')
print("Successfully generated result_multi.png using multi-image outpainting!")

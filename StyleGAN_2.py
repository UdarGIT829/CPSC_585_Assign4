import sys
import torch
import pickle

# # Add paths to PYTHONPATH
sys.path.append('torch_utils')
sys.path.append('dnnlib')

# Assuming ffhq.pkl is in the current directory
with open('ffhq-256.pkl', 'rb') as f:
    data = pickle.load(f)
    G = data['G_ema'].cuda()  # Load the model and send it to GPU
    D = data['D'].cuda()      # Discriminator

# Generate random latent codes
z = torch.randn([1, G.z_dim]).cuda()

# Class labels, not used in this example
c = None

# Generate an image
img = G(z, c)  # Outputs in NCHW format, float32, dynamic range [-1, +1]

import numpy as np
from PIL import Image

# Assuming `img` is your image tensor output from the model

# Convert the range from [-1, 1] to [0, 1]
img = (img.clamp(min=-1, max=1) + 1) / 2.0

# Move the tensor dimension from NCHW to NHWC
img = img.permute(0, 2, 3, 1)

# Remove the batch dimension (assuming batch size of 1)
img = img.squeeze(0)

# Convert to numpy array and then to uint8
img_np = img.cpu().numpy()
img_np = (img_np * 255).astype(np.uint8)

# Create a PIL Image from our numpy array
image = Image.fromarray(img_np)

# Save the image
image.save('output/fake_samples_styleGAN-2.png')

print(f"Raw image shape: {img.shape}")

if img.dim() == 3:
    img = img.unsqueeze(0)  # This makes it [1, channels, height, width]
img = img.permute(0, 3, 1, 2)  # Reorder the dimensions

# Assuming `c` needs to be a tensor of zeros with the same batch size as img
dummy_c = torch.zeros(img.size(0), dtype=torch.int64).cuda()  # Adjust dtype as necessary

print(f"Modified shape: {img.shape}")
# No idea what the dummy is, its supposed to be the classes for the discriminator, but we're passing all zeros?
print(f"Dummy shape?:{dummy_c.shape}")


fake_pred = D(img, dummy_c)
print("Fake prediction:", fake_pred)
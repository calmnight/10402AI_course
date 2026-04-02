import numpy as np
from PIL import Image

# Open the image and convert it to RGB format
im = Image.open('C:/Users/calmnight/python3/10402AI_course/DIP/enlarge/bird1.jpg').convert('RGB')

# Get the width and height of the original image
width, height = im.size
print(f"width: {width}, height: {height}")

# Set the enlargement factor
enlarge = 10

# Convert the original image into a NumPy array
sample_rgb = np.array(im, dtype=np.uint8)

# Create an empty array for the nearest neighbor result
rgb_nn = np.zeros((height * enlarge, width * enlarge, 3), dtype=np.uint8)

# Perform nearest neighbor interpolation
for y in range(height * enlarge):
    for x in range(width * enlarge):
        src_x = min(round(x / enlarge), width - 1)
        src_y = min(round(y / enlarge), height - 1)
        rgb_nn[y, x] = sample_rgb[src_y, src_x]

# Print the shape of the enlarged image array
print("Image processing done.")
print("New image size: ", rgb_nn.shape)

# Convert the NumPy array back to an image and save it
nn_img = Image.fromarray(rgb_nn, 'RGB')
nn_img.save('C:/Users/calmnight/python3/10402AI_course/DIP/NNbird1.png')

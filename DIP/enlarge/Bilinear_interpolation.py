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

# Create an empty array for the bilinear result
rgb_bi = np.zeros((height * enlarge, width * enlarge, 3), dtype=np.uint8)

# Perform bilinear interpolation
for y in range(height * enlarge):
    for x in range(width * enlarge):
         # Map the target pixel back to the source image
        src_x = x / enlarge
        src_y = y / enlarge

        # Find the four neighboring pixels
        x1 = int(np.floor(src_x))
        y1 = int(np.floor(src_y))
        x2 = min(x1 + 1, width - 1)
        y2 = min(y1 + 1, height - 1)

        # Compute the horizontal and vertical distances
        dx = src_x - x1
        dy = src_y - y1

        # Get the RGB values of the four neighboring pixels
        Q11 = sample_rgb[y1, x1].astype(np.float32)
        Q21 = sample_rgb[y1, x2].astype(np.float32)
        Q12 = sample_rgb[y2, x1].astype(np.float32)
        Q22 = sample_rgb[y2, x2].astype(np.float32)

        # Perform bilinear interpolation
        pixel_value = (1 - dx)*(1 - dy)*Q11 + dx*(1 - dy)*Q21 + (1 - dx)*dy*Q12 + dx*dy*Q22

        rgb_bi[y, x] = np.clip(pixel_value, 0, 255).astype(np.uint8)

# Print the shape of the enlarged image array
print("Image processing done.")
print("New image size: ", rgb_bi.shape)

# Convert the NumPy array back to an image and save it
bi_img = Image.fromarray(rgb_bi, 'RGB')
bi_img.save('C:/Users/calmnight/python3/10402AI_course/DIP/BIbird1.png')

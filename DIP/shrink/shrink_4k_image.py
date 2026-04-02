import cv2
import matplotlib.pyplot as plt

# Read the 4K image
img = cv2.imread('C:/Users/calmnight/python3/10402AI_course/DIP/shrink/4k_image.jpg')
if img is None:
    raise FileNotFoundError('Image not found. Please check whether the file path is correct.')

# OpenCV loads images in BGR format, convert to RGB for matplotlib display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Target size: reduce to 1/4 of the original size (can be modified)
h, w = img_rgb.shape[:2]
target_size = (w // 4, h // 4)

# Method 1: direct resize (without Gaussian pyramid)
direct_resize = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)

# Method 2: Gaussian pyramid (each pyrDown applies Gaussian smoothing before downsampling)
pyramid_resize = img_rgb.copy()
while pyramid_resize.shape[1] > target_size[0] * 2 and pyramid_resize.shape[0] > target_size[1] * 2:
    pyramid_resize = cv2.pyrDown(pyramid_resize)

# If the size does not exactly match the target, apply one more resize to align it
pyramid_resize = cv2.resize(pyramid_resize, target_size, interpolation=cv2.INTER_LINEAR)

# Enlarge the images for easier visual comparison of detail differences
# Note: this is only for visualization and is not a required step
zoom_size = (w // 2, h // 2)
direct_zoom = cv2.resize(direct_resize, zoom_size, interpolation=cv2.INTER_NEAREST)
pyramid_zoom = cv2.resize(pyramid_resize, zoom_size, interpolation=cv2.INTER_NEAREST)

# Display the results
plt.figure(figsize=(15, 8))

plt.subplot(1, 3, 1)
plt.imshow(cv2.resize(img_rgb, zoom_size, interpolation=cv2.INTER_LINEAR))
plt.title('Original (scaled for display)')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(direct_zoom)
plt.title('Direct resize (no Gaussian pyramid)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(pyramid_zoom)
plt.title('Gaussian pyramid + resize')
plt.axis('off')

plt.tight_layout()
plt.show()

# Save the results
cv2.imwrite('C:/Users/calmnight/python3/10402AI_course/DIP/shrink/direct_resize.png', cv2.cvtColor(direct_resize, cv2.COLOR_RGB2BGR))
cv2.imwrite('C:/Users/calmnight/python3/10402AI_course/DIP/shrink/pyramid_resize.png', cv2.cvtColor(pyramid_resize, cv2.COLOR_RGB2BGR))

print('Output saved: direct_resize.png and pyramid_resize.png')

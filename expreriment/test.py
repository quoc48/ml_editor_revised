import numpy as np
import matplotlib.pyplot as plt

# 4x4 RGB Image Array
image_array = np.array([
    [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]],   # Row 1
    [[255, 128, 0], [128, 255, 128], [0, 128, 255], [255, 0, 255]],  # Row 2
    [[128, 0, 255], [0, 255, 128], [128, 128, 128], [64, 64, 64]],   # Row 3
    [[0, 0, 0], [255, 255, 255], [192, 192, 192], [64, 128, 192]]    # Row 4
], dtype=np.uint8)

# Display the image
plt.imshow(image_array)
plt.axis('off')  # Hide the axes
plt.show()

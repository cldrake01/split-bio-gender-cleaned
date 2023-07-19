# Facial Landmark Detection and Data Resampling

This repository contains Python code for facial landmark detection and data resampling techniques. It includes functions
for processing frames, aligning faces, overlaying landmarks, and performing random oversampling on training data.

## Requirements

- Python 3.7 or above
- OpenCV
- NumPy
- tqdm
- Torch
- torchvision
- imbalanced-learn

## Installation

1. Clone the repository:
    
   git clone https://github.com/your-username/facial-landmark-detection.git

2. Install the required dependencies:

    pip install -r requirements.txt

## Usage

### Facial Landmark Detection

The facial landmark detection functionality is provided by the `align_face_in_frame()`, `draw_bounding_box()`,
and `overlay_landmarks()` functions. These functions take input frames, bounding box coordinates, and landmark data, and
perform the corresponding operations.

Example usage:

```python
import cv2
from utils import align_face_in_frame, draw_bounding_box, overlay_landmarks

# Load input frame, bounding box coordinates, and landmark data
frame = cv2.imread('frame.jpg')
rect = {'left': 100, 'top': 100, 'width': 200, 'height': 200}
landmarks = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

# Align face in frame
aligned_rect, landmark_confidence = align_face_in_frame(frame, rect, landmarks)

# Draw bounding box on frame
draw_bounding_box(frame, aligned_rect)

# Overlay landmarks on frame
overlay_landmarks(frame, aligned_rect, landmarks)

# Display the resulting frame
cv2.imshow('Output', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

# Perform random oversampling on the training data

The data resampling functionality is provided by the random_over_sampling() function. It performs random oversampling on training data to handle class imbalance issues.

Make sure to provide the correct paths to the training data directory (data_dir) and the directory where the resampled data should be saved (save_dir).

```python
from utils import random_over_sampling

data_dir = '/path/to/training/data'
save_dir = '/path/to/save/resampled/data'

random_over_sampling(data_dir, save_dir)
```
___

### Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

### License
This project is licensed under the MIT License. See the LICENSE file for details.
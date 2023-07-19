import os
import cv2
import numpy as np
from tqdm import tqdm
import torch.utils.data
from lumeopipeline import VideoFrame
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from imblearn.over_sampling import RandomOverSampler
from typing import List, Dict, Tuple, Union, Any


def parse_regression(result_array: List[List[float]]) -> np.ndarray:
    """
    Parse the regression result array into a NumPy array of landmarks.

    :param result_array: List of lists containing the regression result.
    :return: np.ndarray: Array of landmarks.
    """
    result_array = np.array(result_array).reshape(-1, 2)

    # Map values to the range [0, 1]
    # landmarks = (result_array * 2) - 1
    landmarks = result_array

    return landmarks


def align_face_in_frame(frame: np.ndarray, rect: Dict[str, int], landmarks: np.ndarray) -> Tuple[Dict[str, int], float]:
    """
    Aligns the face in the given frame using the bounding box coordinates and landmarks.

    :param frame: The frame in which the face is aligned.
    :param rect: The bounding box coordinates of the face.
    :param landmarks: The landmarks of the face.
    :return: Tuple[Dict[str, int], float]: A tuple containing the updated bounding box coordinates and landmark confidence.
    """
    left, top, width, height = rect['left'], rect['top'], rect['width'], rect['height']
    original_rect_center = np.array([left + width // 2, top + height // 2])

    # Select 17 points
    selected_points = [0, 1, 2, 17, 18, 35, 38, 39, 49, 86, 88, 89, 93, 104]
    src_points = landmarks[selected_points]

    dst_points = np.array([
        (0.0, 0.8632707774798927),
        (-0.8484848484848485, -0.5603217158176943),
        (-0.6151515151515152, 0.3378016085790885),
        (0.8484848484848485, -0.5603217158176943),
        (0.6151515151515152, 0.3378016085790885),
        (-0.6060606060606061, -0.4772117962466488),
        (-0.44545454545454544, -0.47989276139410186),
        (-0.2909090909090909, -0.4396782841823056),
        (-0.48484848484848486, -0.7962466487935657),
        (0.0, 0.0),
        (0.44545454545454544, -0.47989276139410186),
        (0.2909090909090909, -0.4396782841823056),
        (0.6060606060606061, -0.4772117962466488),
        (0.48484848484848486, -0.7962466487935657)], dtype=np.float32)

    M = cv2.estimateAffine2D(src_points, dst_points)[0]

    # Calculate the deviation from the identity matrix
    deviation = np.sqrt((M[0, 0] - 1) ** 2 + M[0, 1] ** 2 + M[1, 0] ** 2 + (M[1, 1] - 1) ** 2)

    # Normalize the deviation to a range between 0 and 1
    max_deviation = 3
    normalized_deviation = deviation / max_deviation

    # Calculate the landmark confidence
    landmark_confidence = 1 - normalized_deviation

    face_roi = frame[top:top + height, left:left + width]
    transformed_face = cv2.warpAffine(face_roi, M, (width, height), borderMode=cv2.BORDER_REPLICATE)

    # Luminance correction
    lab_transformed_face = cv2.cvtColor(transformed_face, cv2.COLOR_BGR2Lab)
    l_channel, a_channel, b_channel = cv2.split(lab_transformed_face)
    l_channel = cv2.normalize(l_channel, None, 0, 255, cv2.NORM_MINMAX)
    corrected_transformed_face = cv2.merge((l_channel, a_channel, b_channel))
    corrected_transformed_face = cv2.cvtColor(corrected_transformed_face, cv2.COLOR_Lab2BGR)

    # Calculate new bounding box corners
    corners = np.array([
        [left, top, 1],
        [left + width, top, 1],
        [left, top + height, 1],
        [left + width, top + height, 1]
    ])

    M_extended = np.vstack([M, [0, 0, 1]])
    transformed_corners = np.matmul(corners, M_extended.T)

    updated_left_top = np.min(transformed_corners, axis=0)[:2]
    updated_right_bottom = np.max(transformed_corners, axis=0)[:2]
    updated_width, updated_height = updated_right_bottom - updated_left_top

    # Calculate the scaling factors
    width_scale = width / updated_width
    height_scale = height / updated_height
    max_scale = max(width_scale, height_scale)
    max_scale = max(1, max_scale)

    # Scale the dimensions of the updated bounding box
    scaled_width = int(updated_width * max_scale)
    scaled_height = int(updated_height * max_scale)

    # Calculate the top-left corner of the new bounding box by moving the transformed face's center
    # to the original bounding box's center
    new_left_top = original_rect_center - np.array([scaled_width // 2, scaled_height // 2])

    # Ensure the new bounding box is within the frame boundaries
    frame_height, frame_width, _ = frame.shape
    new_left_top[0] = max(0, min(frame_width - scaled_width, new_left_top[0]))
    new_left_top[1] = max(0, min(frame_height - scaled_height, new_left_top[1]))

    # Resize the transformed face to fit the scaled bounding box
    resized_transformed_face = cv2.resize(corrected_transformed_face, (scaled_width, scaled_height))

    # Convert the resized face to BGRA
    resized_transformed_face_bgra = cv2.cvtColor(resized_transformed_face, cv2.COLOR_BGR2BGRA)

    frame[int(new_left_top[1]):int(new_left_top[1]) + int(scaled_height),
    int(new_left_top[0]):int(new_left_top[0]) + int(scaled_width)] = resized_transformed_face_bgra

    updated_rect = {
        'left': int(new_left_top[0]),
        'top': int(new_left_top[1]),
        'width': int(scaled_width),
        'height': int(scaled_height)
    }

    return updated_rect, landmark_confidence


def draw_bounding_box(frame: np.ndarray, rect: Dict[str, int]) -> None:
    """
    Draws a bounding box on the given frame using the rectangle coordinates.

    :param frame: The frame on which the bounding box is drawn.
    :param rect: The bounding box coordinates.
    """
    left, top, width, height = rect['left'], rect['top'], rect['width'], rect['height']
    cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)


def overlay_landmarks(frame: np.ndarray, rect: Dict[str, int], landmarks: np.ndarray) -> None:
    """
    Overlays landmarks on the given frame using the bounding box coordinates and landmarks.

    :param frame: The frame on which the landmarks are overlaid.
    :param rect: The bounding box coordinates of the face.
    :param landmarks: The landmarks of the face.
    """
    # Select specific landmarks and their labels
    selected_landmarks = [
        (86, "Nose tip"),
        (93, "Left eye left corner"),
        (89, "Left eye right corner"),
        (39, "Right eye left corner"),
        (35, "Right eye right corner"),
    ]

    left, top, width, height = rect['left'], rect['top'], rect['width'], rect['height']

    for index, label in selected_landmarks:
        x, y = landmarks[index]

        # Convert relative coordinates to absolute
        x = int(left + (x + 1) * width / 2)
        y = int(top + (y + 1) * height / 2)

        # Draw landmark and label on the frame
        cv2.circle(frame, (x, y), 2, (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    for i, coord in enumerate(landmarks):
        x, y = coord

        # Convert relative coordinates to absolute
        x = int(left + (x + 1) * width / 2)
        y = int(top + (y + 1) * height / 2)

        # Draw landmark and label on the frame
        cv2.circle(frame, (x, y), 2, (0, 255, 0), 2)
        cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def process_frame(frame: VideoFrame, node_id: Any = None, **kwargs: Any) -> bool:
    """
    Process a single frame, aligning faces and overlaying landmarks.

    :param frame: The frame to process.
    :param node_id: Identifier of the node (default: None).
    :param kwargs: Additional keyword arguments.
    :return: True if the frame was processed successfully, False otherwise.
    """
    try:
        meta = frame.meta()
        all_meta = meta.get_all()
        objects = all_meta['objects']
        object_tensors = frame.object_tensors()

        facial_landmark_results = process_object_tensors(objects, object_tensors)

        with frame.data() as mat:
            for idx, result in enumerate(facial_landmark_results):
                rect = result['rect']
                landmarks = result['landmarks']
                new_rect, confidence = align_face_in_frame(mat, rect, landmarks)
                if new_rect is not None:
                    objects[idx]['rect'] = new_rect
                    objects[idx]['attributes'].append({
                        'label': "landmarks",
                        'class_id': 21999,
                        'probability': confidence
                    })

        meta.set_field('objects', objects)
        meta.save()

    except Exception as e:
        print(e)

    return True


def process_object_tensors(objects: List[Dict[str, Union[str, Dict[str, int]]]],
                           object_tensors: List[torch.Tensor]) -> List[Dict[str, Union[np.ndarray, Dict[str, int]]]]:
    """
    Process the object tensors to extract facial landmarks.

    :param objects: List of objects in the frame.
    :param object_tensors: List of object tensors.
    :return: List of facial landmark results.
    """
    ret = []
    for obj, obj_tensor in zip(objects, object_tensors):
        if obj['label'] == 'face':
            for tensor in obj_tensor.tensors:
                for layer in tensor.layers:
                    if layer.name == 'fc1' and layer.dimensions == [212]:
                        to_add = {}
                        to_add['landmarks'] = parse_regression(layer.data)
                        to_add['rect'] = obj['rect']
                        ret.append(to_add)
    return ret


def random_over_sampling(data_dir: str, save_dir: str = 'post-ros-split-bio-gender-cleaned/train') -> None:
    """
    Resample the training data to handle class imbalance issue. The resampled data will be saved to the specified
    directory. Note: This function is only for the training data. The validation and test data should not be resampled.

    :param data_dir: The directory of the training data, e.g. '/Users/collindrake/Downloads/pre-ros-split-bio-gender-cleaned'.
    :param save_dir: The directory to which the final results will be saved. The default parameter is sufficient for
                     Linux and macOS users.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_data = ImageFolder(root=data_dir + '/train', transform=transform)

    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)

    # Handle class imbalance issue by using weighted loss function or other techniques
    ros = RandomOverSampler(sampling_strategy='not majority')

    train_loader_dataset = np.fromiter(train_loader.dataset, dtype=object)
    train_loader_dataset = np.reshape(train_loader_dataset, (-1, 1))
    train_data.targets = np.array(train_data.targets)
    train_data.targets = np.reshape(train_data.targets, (-1, 1))

    print(train_loader_dataset.shape)
    print(train_data.targets.shape)

    print("Original training data size: ", len(train_data.targets))

    X, y = ros.fit_resample(train_loader_dataset, train_data.targets)

    print("Resampled training data size: ", len(X))

    os.makedirs(save_dir, exist_ok=True)

    file: int = 0

    for folder in tqdm(X, desc="Resampling", total=len(X)):
        file += 1
        str_file: str = f'{file}'

        for image_index in range(len(folder)):
            image, label = folder[image_index]
            str_label: str = f'{label}'

            os.makedirs(save_dir + '/' + str_label, exist_ok=True)

            image *= 255

            if image.shape[0] != 3:
                image = image.repeat(3, 1, 1)

            image_bgr = image.permute(1, 2, 0).numpy()
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            filename: str = f'{str_file}.jpg'
            final_path = os.path.join(save_dir, str_label, filename)

            cv2.imwrite(final_path, image_rgb)

    print("Resampling: Complete")

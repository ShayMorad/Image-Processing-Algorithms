import cv2
import numpy as np
import os

# Constants and Hyperparameters
CANVAS_X_START = 20  # Initial X-position on the canvas for placing strips
CANVAS_Y_START = 20  # Initial Y-position on the canvas for placing strips
SCALE = 0.3  # Scaling factor for resizing frames
FPS = 240  # Frames per second for output video
COL_START = 20  # Starting column for strip extraction
COL_END = 20  # Ending column for strip extraction
RESIZE_FACTOR = 0.5  # Resize factor for pre-processing video frames
MIN_TRANSLATION_THRESHOLD = 1  # Minimum translation required to include a strip



def draw_matches(frame1, frame2, keypoints1, keypoints2, matches, inliers_mask):
    """
    Draw matches between keypoints of two frames with custom colors for inliers and outliers.

    Args:
        frame1 (np.ndarray): The first frame.
        frame2 (np.ndarray): The second frame.
        keypoints1 (List[cv2.KeyPoint]): Keypoints from the first frame.
        keypoints2 (List[cv2.KeyPoint]): Keypoints from the second frame.
        matches (List[cv2.DMatch]): List of matched keypoints.
        inliers_mask (np.ndarray): Boolean array indicating which matches are inliers (1 for inliers, 0 for outliers).

    Returns:
        np.ndarray: Image showing matched keypoints with inliers in yellow and outliers in blue.
    """
    # Combine frames horizontally
    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]
    output_image = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    output_image[:h1, :w1] = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR) if len(frame1.shape) == 2 else frame1
    output_image[:h2, w1:w1 + w2] = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR) if len(frame2.shape) == 2 else frame2

    for i, match in enumerate(matches):
        pt1 = tuple(map(int, keypoints1[match.queryIdx].pt))
        pt2 = tuple(map(int, keypoints2[match.trainIdx].pt))
        pt2 = (pt2[0] + w1, pt2[1])  # Adjust for horizontal combination

        # Draw in yellow for inliers, blue for outliers
        color = (0, 255, 255) if inliers_mask[i] else (255, 0, 0)
        cv2.line(output_image, pt1, pt2, color, 2)
        cv2.circle(output_image, pt1, 5, color, -1)
        cv2.circle(output_image, pt2, 5, color, -1)

    return output_image


def load_video(video_path):
    """
    Load video frames from a video file and convert them to RGB format.

    Args:
        video_path (str): Path to the video file.

    Returns:
        List[np.ndarray]: List of video frames in RGB format.
    """
    cap = cv2.VideoCapture(video_path)  # Open the video file
    frames = []  # List to store the frames
    while cap.isOpened():
        ret, frame = cap.read()  # Read the frame
        if not ret:
            break  # If no more frames, stop the loop
        frames.append(frame)  # Append the RGB frame to the list
    cap.release()  # Release the video capture object
    return frames  # Return the list of RGB frames


def calculate_rigid_transform(src, dst, visualize=False, no_rotation=False):
    """
    Calculate the rigid transformation (rotation and translation) between two frames.

    Args:
        src (np.ndarray): Source frame.
        dst (np.ndarray): Destination frame.
        visualize (bool): If True, display matched keypoints.
        no_rotation (bool): If True, remove rotation from the transformation.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Affine transformation matrix and inliers mask.
    """

    # Convert images to grayscale if not already
    if len(src.shape) == 3:
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    if len(dst.shape) == 3:
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    # Detect Harris corners using goodFeaturesToTrack
    src_corners = cv2.goodFeaturesToTrack(src, maxCorners=500, qualityLevel=0.01, minDistance=10, blockSize=7)
    dst_corners = cv2.goodFeaturesToTrack(dst, maxCorners=500, qualityLevel=0.01, minDistance=10, blockSize=7)

    # Convert corners to cv2.KeyPoint objects
    keypoints1 = [cv2.KeyPoint(x=float(pt[0][0]), y=float(pt[0][1]), size=7) for pt in src_corners]
    keypoints2 = [cv2.KeyPoint(x=float(pt[0][0]), y=float(pt[0][1]), size=7) for pt in dst_corners]

    # Create SIFT object and compute descriptors
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.compute(src, keypoints1)
    keypoints2, descriptors2 = sift.compute(dst, keypoints2)

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 4:
        raise Exception("Not enough good matches to calculate rotation.")

    # Extract points from good matches
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    # Use RANSAC to find the best transformation (rotation + translation)
    matrix, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)

    if matrix is None:
        raise Exception("RANSAC failed to find a valid transformation.")

    if visualize:
        inliers_mask = inliers.ravel() > 0  # Convert inliers mask to boolean array
        output_image = draw_matches(src, dst, keypoints1, keypoints2, matches, inliers_mask)
        cv2.imshow("Keypoint Matches (Inliers: Yellow, Outliers: Blue)", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if no_rotation:
        matrix[0][0] = 1
        matrix[0][1] = 0
        matrix[1][0] = 0
        matrix[1][1] = 1
    return matrix, inliers


def calculate_forward_transformations(frames, no_rotation=False):
    """
        Calculate transformations between consecutive frames.

        Args:
            frames (List[np.ndarray]): List of video frames.
            no_rotation (bool): If True, disable rotation in the transformations.

        Returns:
            List[np.ndarray]: List of 2x3 affine transformation matrices.
        """
    stabilized_frames = [frames[0]]  # Start with the first frame as reference
    clean_forward_transformations = []
    for i in range(1, len(frames)):
        print(f"Calculating transformation from frame: {i - 1} to frame: {i}")
        frame0 = frames[i - 1]  # Previous frame
        frame1 = frames[i]  # Current frame

        # Calculate transformation matrix between consecutive frames, frame0 -> frame1 for example.
        transformation_matrix, inliers = calculate_rigid_transform(frame0, frame1, False, no_rotation)
        clean_forward_transformations.append(transformation_matrix)

    return clean_forward_transformations


def calculate_cumulative_transformations(forward_transformations, idx, zero_x=False, zero_y=False):
    """
    Calculate cumulative transformations relative to a reference frame.

    Args:
        forward_transformations (List[np.ndarray]): List of transformations between consecutive frames.
        idx (int): Index of the reference frame.
        zero_x (bool): If True, zero out the X translation.
        zero_y (bool): If True, zero out the Y translation.

    Returns:
        List[np.ndarray]: List of cumulative 3x3 homography matrices.
    """
    print("calculating cumulative transformations")
    transformations = [np.eye(3) for _ in range(len(forward_transformations) + 1)]
    forward_trans = np.eye(3)
    backward_trans = np.eye(3)

    # Accumulate homographies for frames before the middle frame
    for i in range(idx - 1, -1, -1):
        next = np.vstack([forward_transformations[i], [0, 0, 1]])
        if zero_x:
            next[0, 2] = 0
        if zero_y:
            next[1, 2] = 0
        forward_trans = forward_trans @ next
        forward_trans /= forward_trans[2, 2]
        transformations[i] = forward_trans

    # Accumulate homographies for frames after the middle frame
    for i in range(idx, len(forward_transformations)):
        next = np.vstack([forward_transformations[i], [0, 0, 1]])
        if zero_x:
            next[0, 2] = 0
        if zero_y:
            next[1, 2] = 0
        next_inv = np.linalg.inv(next)
        backward_trans = backward_trans @ next_inv
        backward_trans /= backward_trans[2, 2]
        transformations[i + 1] = backward_trans
    transformations[idx] = np.eye(3)
    return transformations


def calculate_global_canvas_size_and_offset(frames, transformations):
    """
    Compute the required canvas size and offsets.

    Args:
        frames (List[np.ndarray]): List of video frames.
        transformations (List[np.ndarray]): List of 3x3 transformation matrices for each frame.

    Returns:
        Tuple[Tuple[int, int], Tuple[int, int]]: Canvas size (width, height) and offsets (x_min, y_min).
    """
    # Get dimensions of the first frame
    height, width = frames[0].shape[:2]

    # Define the corners of the base frame in homogeneous coordinates
    frame_corners = np.array([
        [0, 0, 1],  # Top-left corner
        [width, 0, 1],  # Top-right corner
        [0, height, 1],  # Bottom-left corner
        [width, height, 1]  # Bottom-right corner
    ])

    # Track transformed corners for all frames
    all_transformed_corners = []

    for transform in transformations:
        transformed_corners = (transform @ frame_corners.T).T  # Apply transformation to corners
        all_transformed_corners.append(transformed_corners)

    all_transformed_corners = np.vstack(all_transformed_corners)  # Stack all corners

    # Calculate global min and max coordinates
    x_min, y_min = np.floor(all_transformed_corners[:, :2].min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_transformed_corners[:, :2].max(axis=0)).astype(int)

    # Calculate canvas width and height
    canvas_width = x_max - x_min
    canvas_height = y_max - y_min

    return (canvas_width, canvas_height), np.array([x_min, y_min])

#
# def stabilize_frames(frames, transformations):
#     """
#     Stabilize frames by neutralizing the X translation in the transformation and applying only Y translations and rotations.
#
#     Args:
#         frames (List[np.ndarray]): List of input video frames (images).
#         transformations (List[np.ndarray]): List of 3x3 transformation matrices for each frame.
#
#     Returns:
#         List[np.ndarray]: List of stabilized frames.
#     """
#     output_dir = "stab_frames"
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     stabilized_frames = []
#
#     for i, frame in enumerate(frames):
#         stabilized_transform = transformations[i].copy()
#         print(f"Stabilizing frame: {i}, with transformation: {stabilized_transform}")
#         # Apply the transformation to stabilize the frame
#         height, width = frame.shape[:2]
#         stabilized_frame = cv2.warpAffine(frame, stabilized_transform[:2, :], (width, height))
#         stabilized_frames.append(stabilized_frame)
#
#         # Save each stabilized frame as an image
#         frame_filename = os.path.join(output_dir, f"stabilized_frame_{i:03d}.png")
#         cv2.imwrite(frame_filename, stabilized_frame)
#
#     return stabilized_frames


def create_mosaics_video(frames, transformations, reference_index=0, start_col=5, end_col=400,
                         min_translation_threshold=5, output_dir="panorama_outputs", input_type="moving"):
    """
    Create multiple panoramas by extracting vertical strips from specified columns of each frame and assembling them into a single canvas.

    Args:
        frames (List[np.ndarray]): List of video frames (images).
        transformations (List[np.ndarray]): List of 3x3 transformation matrices for each frame.
        reference_index (int): The index of the reference frame (typically the middle frame).
        start_col (int): The column index to start creating panoramas.
        end_col (int): The column index to end creating panoramas.
        min_translation_threshold (int): Minimum translation difference in pixels to include a strip.
        output_dir (str): Directory where the panorama images will be saved.
        input_type (str): Mode of panorama creation - "moving" for dynamic offsets, "static" for static alignment.

    Returns:
        List[np.ndarray]: List of generated panorama images.
    """

    if reference_index is None:
        reference_index = len(frames) // 2  # Set the reference frame as the middle frame if not specified

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the global canvas size and offsets
    (canvas_width, canvas_height), min_coords = calculate_global_canvas_size_and_offset(frames, transformations)
    panoramas = []
    canvas_width = int(canvas_width * 1.2)
    canvas_height = int(canvas_height * 1.2)
    # Loop over the specified column range to create panoramas
    if input_type == "moving":
        for col_offset in range(start_col, end_col + 1):
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)  # Black canvas
            accumulated_x_pos = CANVAS_X_START  # Track where the strip should be placed on the canvas
            y_offset = CANVAS_Y_START
            accumulated_prev_x_trans = 0  # Track accumulated X translations to calculate dynamic strip width
            accumulated_prev_y_trans = 0
            print(f"Creating panorama starting at column: {col_offset} for each frame.")
            for i, frame in enumerate(frames):
                height, width = frame.shape[:2]

                # Calculate the total X translation for the current frame
                transformed_origin = transformations[i] @ np.array([0, 0, 1])
                transformed_origin /= transformed_origin[2]  # Normalize homogeneous coordinate
                current_translation_x = transformed_origin[0]
                current_translation_y = transformed_origin[1]

                # Reset translation when reaching the reference index
                if i == reference_index:
                    accumulated_prev_x_trans = 0

                # Calculate strip width based on X translation
                strip_width = round(np.abs(current_translation_x - accumulated_prev_x_trans))
                current_translation_y = round(current_translation_y - accumulated_prev_y_trans)
                # Skip frame if the translation is below the threshold
                if strip_width < min_translation_threshold:
                    continue

                # Start taking the strip from the specified column
                col_start = col_offset

                # Ensure col_start and strip width do not exceed frame dimensions
                col_end = min(col_start + strip_width, width)

                if col_end <= col_start:
                    continue  # Skip if the strip cannot be extracted

                # Extract the vertical strip from the frame
                strip = frame[:, col_start:col_end]

                # Calculate where to place the strip in the canvas based on accumulated position
                x_pos = accumulated_x_pos
                accumulated_x_pos += (col_end - col_start)

                x_end = min(x_pos + (col_end - col_start), canvas_width)
                if x_end <= x_pos:
                    continue  # Skip if the strip does not fit on the canvas

                strip_trimmed = strip[:, :x_end - x_pos]  # Trim the strip if necessary
                if input_type == "moving":
                    canvas[y_offset + current_translation_y:current_translation_y + y_offset + height,
                    col_offset + x_pos:col_offset + x_end] = strip_trimmed
                elif input_type == "static":
                    canvas[y_offset + current_translation_y:current_translation_y + y_offset + height,
                    x_pos:x_end] = strip_trimmed

                # Update previous translation for the next iteration
                accumulated_prev_x_trans = current_translation_x
                accumulated_prev_y_trans = transformed_origin[1]
            panoramas.append(canvas)

            # Save each panorama image and display it for playback-like visualization
            panorama_filename = os.path.join(output_dir, f"panorama_col_{col_offset}.png")
            cv2.imwrite(panorama_filename, canvas)
    elif input_type == "static":
        for col_offset in range(end_col, start_col - 1, -1):
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)  # Black canvas
            accumulated_x_pos = CANVAS_X_START  # Track where the strip should be placed on the canvas
            y_offset = CANVAS_Y_START
            accumulated_prev_x_trans = 0  # Track accumulated X translations to calculate dynamic strip width
            accumulated_prev_y_trans = 0
            print(f"Creating panorama starting at column: {col_offset} for each frame.")
            for i, frame in enumerate(frames):
                height, width = frame.shape[:2]

                # Calculate the total X translation for the current frame
                transformed_origin = transformations[i] @ np.array([0, 0, 1])
                transformed_origin /= transformed_origin[2]  # Normalize homogeneous coordinate
                current_translation_x = transformed_origin[0]
                current_translation_y = transformed_origin[1]

                # Reset translation when reaching the reference index
                if i == reference_index:
                    accumulated_prev_x_trans = 0

                # Calculate strip width based on X translation
                strip_width = int(np.abs(current_translation_x - accumulated_prev_x_trans))
                current_translation_y = int(current_translation_y - accumulated_prev_y_trans)
                # Skip frame if the translation is below the threshold
                if strip_width < min_translation_threshold:
                    continue

                # Start taking the strip from the specified column
                col_start = col_offset

                # Ensure col_start and strip width do not exceed frame dimensions
                col_end = min(col_start + strip_width, width)

                if col_end <= col_start:
                    continue  # Skip if the strip cannot be extracted

                # Extract the vertical strip from the frame
                strip = frame[:, col_start:col_end]

                # Calculate where to place the strip in the canvas based on accumulated position
                x_pos = accumulated_x_pos
                accumulated_x_pos += (col_end - col_start)

                x_end = min(x_pos + (col_end - col_start), canvas_width)
                if x_end <= x_pos:
                    continue  # Skip if the strip does not fit on the canvas

                strip_trimmed = strip[:, :x_end - x_pos]  # Trim the strip if necessary
                if input_type == "moving":
                    canvas[y_offset + current_translation_y:current_translation_y + y_offset + height,
                    col_offset + x_pos:col_offset + x_end] = strip_trimmed
                elif input_type == "static":
                    canvas[y_offset + current_translation_y:current_translation_y + y_offset + height,
                    x_pos:x_end] = strip_trimmed

                # Update previous translation for the next iteration
                accumulated_prev_x_trans = current_translation_x
                accumulated_prev_y_trans = transformed_origin[1]
            panoramas.append(canvas)

            # Save each panorama image and display it for playback-like visualization
            panorama_filename = os.path.join(output_dir, f"panorama_col_{col_offset}.png")
            cv2.imwrite(panorama_filename, canvas)
    return panoramas


def save_video(frames, output_path, fps=30, resize_factor=1.0):
    """
    Save a sequence of frames as an MP4 video.

    Args:
        frames (List[np.ndarray]): List of frames to be saved as a video.
        output_path (str): Path to save the output video file.
        fps (int): Frame rate of the video (frames per second).
        resize_factor (float): Factor by which to resize frames before saving.

    Returns:
        None
    """
    if resize_factor != 1.0:
        frames = [cv2.resize(frame, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA) for frame in
                  frames]

    height, width = frames[0].shape[:2]

    # Use 'mp4v' codec for MP4 compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()

    print(f"Stabilized video saved as: {output_path}")


def resize_frames(frames, scale_factor=0.5):
    """
        Resize a list of frames by a specified scale factor.

        Args:
            frames (List[np.ndarray]): List of video frames (images).
            scale_factor (float): Scaling factor for resizing frames.

        Returns:
            List[np.ndarray]: List of resized frames.
        """
    resized_frames = [cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor) for frame in frames]
    return resized_frames


def create_video_from_frames(frames, output_path, fps=30):
    """
    Create a video from a list of frames that plays in sequence.

    Args:
        frames (List[np.ndarray]): List of frames (images) to be saved as a video.
        output_path (str): Path to save the output video file.
        fps (int): Frames per second for the video playback.

    Returns:
        None
    """
    height, width, _ = frames[0].shape

    # Define the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"Video saved at {output_path}")


def preprocess_video(input_path, output_path, target_width=None, target_height=None, target_fps=None):
    """
    Preprocess a video by adjusting its resolution and frame rate.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the output (preprocessed) video.
        target_width (int): Desired width of the output video. If None, keeps the original width.
        target_height (int): Desired height of the output video. If None, keeps the original height.
        target_fps (float): Desired frames per second for the output video. If None, keeps the original FPS.

    Returns:
        None
    """
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    # Get original video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    # Set target resolution and FPS
    width = target_width if target_width else original_width
    height = target_height if target_height else original_height
    fps = target_fps if target_fps else original_fps

    # Define the codec and create the VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # No more frames

        # Resize the frame if necessary
        if (width, height) != (original_width, original_height):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        # Write the processed frame
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"Preprocessed video saved at {output_path} with {frame_count} frames at {fps:.2f} FPS.")


def main(input_video_path, output_video_path, input_type, rotation):
    #preprocess_video(input_video_path, "inputs/mine_bad.mp4", 640, 360, 10)
    frames = load_video(input_video_path)
    height, width = frames[0].shape[:2]
    clean_forward_transformations = calculate_forward_transformations(frames.copy(), True)
    cumulative_transformations = calculate_cumulative_transformations(clean_forward_transformations.copy(), 0, False,
                                                                      False)
    panoramas = create_mosaics_video(frames, cumulative_transformations, reference_index=0,
                                     start_col=COL_START, end_col=width - COL_END,
                                     min_translation_threshold=MIN_TRANSLATION_THRESHOLD,
                                     input_type=input_type)
    panoramas = resize_frames(panoramas, scale_factor=SCALE)
    create_video_from_frames(panoramas, output_video_path, FPS)


if __name__ == "__main__":
    main("viewpoint_input.mp4", "viewpoint_result.mp4", "moving", False)

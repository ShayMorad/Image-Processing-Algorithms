import cv2
import numpy as np
import math
import os

# Constants and Hyperparameters
DEBUG = True  # Set to False to disable debug outputs
SIFT_FEATURES = 500
RESIZE_FACTOR = 0.5  # For downscaling input videos if needed


def load_video(video_path):
    """Load video frames as a list of images."""
    cap = cv2.VideoCapture(video_path)  # Open the video file
    frames = []  # List to store the frames

    # # Create output directory if it doesn't exist
    # if not os.path.exists("frames_check"):
    #     os.makedirs("frames_check")
    #
    # frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()  # Read the frame
        if not ret:
            break  # If no more frames, stop the loop
        frames.append(frame)  # Append the frame to the list
        # # Save the frame as an image
        # frame_filename = os.path.join("frames_check", f"frame_{frame_count:04d}.png")
        # cv2.imwrite(frame_filename, frame)
        #
        # frame_count += 1  # Increment frame count
    cap.release()  # Release the video capture object
    print("num of frames: ", len(frames))
    return frames  # Return the list of frames


def detect_and_match_features(frame1, frame2):
    """Detect features using SIFT and match them between two frames."""
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(frame1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(frame2, None)

    # Match features using brute-force matcher (no FLANN)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    return keypoints1, keypoints2, matches


def calculate_transformation_matrix(frame1, frame2):
    """Calculate transformation matrix (rotation, translation, etc.) between two frames."""
    print("calculating transformation matrix")

    # Detect and match features
    keypoints1, keypoints2, matches = detect_and_match_features(frame1, frame2)
    match_img = draw_matches(frame1, frame2, keypoints1, keypoints2, matches)
    cv2.imshow("Matches", match_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(matches) < 2:
        raise Exception("Not enough matches to calculate transformation.")

    # Extract two matched points from the matches
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches[:2]])  # First two points from frame1
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches[:2]])  # Corresponding points from frame2

    # Calculate the angle of the lines formed by the points in both frames
    # Line 1: (x1, y1) -> (x2, y2)
    x1, y1 = src_pts[0]
    x2, y2 = src_pts[1]
    dx1, dy1 = x2 - x1, y2 - y1
    angle1 = math.atan2(dy1, dx1)  # Angle of the line in frame1

    # Line 2: (x1', y1') -> (x2', y2')
    x1_prime, y1_prime = dst_pts[0]
    x2_prime, y2_prime = dst_pts[1]
    dx2, dy2 = x2_prime - x1_prime, y2_prime - y1_prime
    angle2 = math.atan2(dy2, dx2)  # Angle of the line in frame2

    # Calculate the rotation as the difference between the two angles
    rotation_angle = angle2 - angle1

    # Rotation matrix (2D)
    rotation_matrix = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle)],
        [np.sin(rotation_angle), np.cos(rotation_angle)]
    ])

    # Apply rotation to the source points (frame1)
    rotated_src_pts = np.dot(rotation_matrix, np.array([src_pts[0][0], src_pts[0][1]]))

    # Calculate the translation based on only one pair of points (the first pair)
    # Translation is the difference between the rotated source point and the corresponding destination point
    translation_x = dst_pts[0][0] - src_pts[0][0]
    translation_y = dst_pts[0][1] - src_pts[0][1]

    # Combine rotation and translation into a transformation matrix (2x3)
    transformation_matrix_2x3 = np.hstack([rotation_matrix, np.array([[translation_x], [translation_y]])])

    print(transformation_matrix_2x3)
    print("finished calculating transformation matrix")
    return transformation_matrix_2x3  # The transformation matrix

def stabilize_video(frames):
    """Stabilize video by removing shaky movements and rotations."""
    print("Stabilizing video")

    stabilized_frames = [frames[0]]  # Start with the first frame as the reference
    previous_transformation_matrix = np.eye(2, 3)  # Identity matrix for initial frame
    previous_angle = 0  # To keep track of the previous rotation angle

    for i in range(1, len(frames)):
        frame1 = stabilized_frames[i - 1]  # Previous frame (stabilized)
        frame2 = frames[i]  # Current frame

        # Calculate the transformation matrix between consecutive frames
        transformation_matrix = calculate_transformation_matrix(frame1, frame2)

        # Decompose the transformation matrix into rotation and translation
        rotation_matrix = transformation_matrix[:, :2]
        translation_x = transformation_matrix[0, 2]
        translation_y = transformation_matrix[1, 2]

        # Limit the rotation to prevent excessive rotation from accumulating
        angle_change = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0]) - previous_angle
        angle_change = max(-5, min(5, angle_change))  # Limit to 5 degrees of rotation

        # Update the previous angle
        previous_angle += angle_change

        # Apply limited rotation (to avoid large jumps)
        rotation_matrix = np.array([
            [np.cos(previous_angle), -np.sin(previous_angle)],
            [np.sin(previous_angle), np.cos(previous_angle)]
        ])

        # Limit the translation to avoid large shifts (especially vertical)
        translation_x = max(-50, min(50, translation_x))  # Limit horizontal translation
        translation_y = 0  # Remove vertical shake (no y translation)

        # Construct the new transformation matrix with limited translation and adjusted rotation
        stable_transformation_matrix = np.hstack([rotation_matrix, np.array([[translation_x], [translation_y]])])

        # Apply the stabilization transformation to the current frame
        stabilized_frame = cv2.warpAffine(frame2, stable_transformation_matrix, (frame2.shape[1], frame2.shape[0]))

        stabilized_frames.append(stabilized_frame)  # Append the stabilized frame

    print("Finished stabilizing video")
    return stabilized_frames  # Return stabilized frames


def align_frames(frames):
    """Align consecutive frames using transformation matrices."""
    print("aligning frames")
    aligned_frames = [frames[0]]  # First frame is the reference
    for i in range(1, len(frames)):
        frame1 = aligned_frames[i - 1]  # Previous aligned frame
        frame2 = frames[i]  # Current frame

        # Calculate transformation matrix
        transformation_matrix = calculate_transformation_matrix(frame1, frame2)

        # Apply transformation to the current frame
        aligned_frame = cv2.warpAffine(frame2, transformation_matrix, (frame2.shape[1], frame2.shape[0]))

        aligned_frames.append(aligned_frame)  # Add the aligned frame to the list
    print("finished aligning frames")
    return aligned_frames  # Return the list of aligned frames


def overlay_frames(frame1, frame2, alpha=0.5):
    """Overlay two frames with some transparency."""
    return cv2.addWeighted(frame1, alpha, frame2, 1 - alpha, 0)


def draw_matches(frame1, frame2, keypoints1, keypoints2, matches):
    """Draw the matches between keypoints of two frames."""
    return cv2.drawMatches(frame1, keypoints1, frame2, keypoints2, matches, None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


def calculate_frame_difference(frame1, frame2):
    """Calculate the difference between two frames."""
    return cv2.absdiff(frame1, frame2)



def main():
    # Load input video
    input_video_path = "inputs/boat.mp4"
    frames = load_video(input_video_path)

    # Stabilize video frames
    stabilized_frames = stabilize_video(frames)

    # Write the stabilized frames to a new video file
    height, width, layers = stabilized_frames[0].shape
    output_video_path = "outputs/stabilized_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Specify video codec
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))

    for frame in stabilized_frames:
        out.write(frame)  # Write each frame to the output video

    out.release()  # Release the VideoWriter
    print("Stabilized video saved to:", output_video_path)


if __name__ == "__main__":
    main()

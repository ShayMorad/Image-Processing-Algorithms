import cv2
import os

def save_video_frames(video_path, output_dir):
    """
    Load a video, extract the first, middle, and last frames, and save them as images.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where the frames will be saved.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_frame_index = total_frames // 2

    frame_indices = [0, mid_frame_index, total_frames - 1]
    frame_names = ["first_frame.png", "middle_frame.png", "last_frame.png"]

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count in frame_indices:
            frame_name = frame_names[frame_indices.index(frame_count)]
            frame_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(frame_path, frame)
            print(f"Saved {frame_name} at {frame_path}")

        frame_count += 1

    cap.release()
    print("Frames saved successfully!")

def main():
    video_path = "viewpoint_result.mp4"  # Replace with the actual path to your video
    output_dir = "viewpoint_frames"
    save_video_frames(video_path, output_dir)
    video_path = "dynamic_result.mp4"  # Replace with the actual path to your video
    output_dir = "dynamic_frames"
    save_video_frames(video_path, output_dir)
    video_path = "good_result.mp4"  # Replace with the actual path to your video
    output_dir = "good_frames"
    save_video_frames(video_path, output_dir)
    video_path = "bad_result.mp4"  # Replace with the actual path to your video
    output_dir = "bad_frames"
    save_video_frames(video_path, output_dir)

if __name__ == "__main__":
    main()

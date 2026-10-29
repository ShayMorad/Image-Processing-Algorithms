import numpy as np
import mediapy as mp

GRAYSCALE_LEVELS = 256
GRAYSCALE_MIN_LEVEL = 0
GRAYSCALE_MAX_LEVEL = 255


def main(video_path, video_type):
    """
    Main entry point for ex1
    :param video_path: path to video file
    :param video_type: category of the video (either 1 or 2)
    :return: a tuple of integers representing the frame number for which the scene cut was detected (i.e. the last frame index of the first scene and the first frame index of the second scene)
    """

    # create a list to store the cumulative histograms of each frame
    frames_cumsum_histograms = []

    # import requested video by video_path argument and convert it to grayscale, so we can use an algorithm similar to
    # what was learned in class.
    imported_video = mp.read_video(video_path, output_format="gray")

    for i, frame in enumerate(imported_video):
        # calculate the histogram for each frame using numpy and 256 grayscale levels as learnt in class
        histogram, bins = np.histogram(frame, bins=GRAYSCALE_LEVELS, range=(GRAYSCALE_MIN_LEVEL, GRAYSCALE_MAX_LEVEL))
        # calculate the cumulative histogram for each frame for better comparison accuracy as learnt in class
        cumulativ_histogram = np.cumsum(histogram)

        # add each cumulative histogram of each frame to our list of frames histograms so we can do comparisons on it
        # and find our wanted frame-cut
        frames_cumsum_histograms.append(cumulativ_histogram)

    # create a list to store the difference between each 2 consecutive frames in their cumulative histograms
    frames_differences = []

    for i in range(len(frames_cumsum_histograms) - 1):
        # calculate histograms difference as shown in class
        cumsum_histograms_diff = np.abs(frames_cumsum_histograms[i + 1] - frames_cumsum_histograms[i])

        # store difference in the designated list
        frames_differences.append(cumsum_histograms_diff)

    # create a single value list of the differences in histograms between frames, as the previous list is a more complex
    # one, where each object is a 256-element array representing the pixel-wise difference.

    summed_frames_differences = []
    for diff in frames_differences:
        summed_delta = np.sum(diff)
        summed_frames_differences.append(summed_delta)

    # pull the start index of the first frame between the 2 consecutive frames that had the highest cumsum histograms
    # difference between them
    first_frame_idx = np.argmax(summed_frames_differences)
    second_frame_idx = first_frame_idx + 1

    return (first_frame_idx, second_frame_idx)

    # # for testing and plotting purposes
    # return ((first_frame_idx, second_frame_idx), frames_cumsum_histograms, frames_delta, summed_frames_delta)


# if __name__ == '__main__':
#     video_path = "path/to/input_video.mp4"
#
#     try:
#         frames = mp.read_video(video_path, output_format="gray")
#         print(f"Successfully loaded {len(frames)} frames.")
#         print(f"Frame shape: {frames[0].shape if frames else 'No frames loaded.'}")
#     except Exception as e:
#         print(f"Error: {e}")

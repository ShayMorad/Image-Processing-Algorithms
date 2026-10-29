import ex1
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
from pathlib import Path



def process_videos():
    cat1_histograms, cat2_histograms = [], []
    cat1_deltas, cat2_deltas = [], []

    # Iterating over the files in the current directory
    for file in os.listdir("."):
        path = Path(file)
        file_name, file_extension = path.stem, path.suffix
        if "mp4" not in file_extension:
            continue
        print(f"Processing {path.name} with extension {file_extension}")

        # Identify the category from the file name
        category = get_category(file_name)
        if category is None:
            print(f"Skipping {file}: No recognized category")
            continue

        try:
            # Extract data from the video using the ex1 module
            (first_frame, second_frame), cumsum_histograms, deltas, summed_deltas = ex1.main(str(path), category)

            # Append data to respective category lists
            if category == "1":
                cat1_histograms.append(cumsum_histograms)
                cat1_deltas.append(summed_deltas)
            elif category == "2":
                cat2_histograms.append(cumsum_histograms)
                cat2_deltas.append(summed_deltas)

            print(f"Scene cut detected between frames {first_frame} and {second_frame} for {path.name}")
        except Exception as e:
            print(f"Error processing {file}: {e}")

    # Check for sufficient data
    if len(cat1_histograms) < 1 or len(cat2_histograms) < 1:
        print("Insufficient data for generating plots.")
        return

    # Plot histograms for each category
    try:
        generate_histogram_plot(1, 1, cat1_histograms[0], cut_frame_index=99)
        generate_histogram_plot(1, 2, cat1_histograms[1], cut_frame_index=149)
        generate_histogram_plot(2, 1, cat2_histograms[0], cut_frame_index=174)
        generate_histogram_plot(2, 2, cat2_histograms[1], cut_frame_index=74)
    except IndexError as e:
        print(f"Error generating histograms: {e}")

    # Generate delta plots for each category
    try:
        generate_delta_plot(1, cat1_deltas)
        generate_delta_plot(2, cat2_deltas)
    except Exception as e:
        print(f"Error generating delta plots: {e}")


def get_category(file_name):
    """Determine the category from the file name."""
    if "category1" in file_name:
        return "1"
    elif "category2" in file_name:
        return "2"
    return None


def generate_histogram_plot(category_num, video_num, histograms, cut_frame_index):
    """
    Generate a simplified histogram plot showing cumulative pixel trends across all frames.
    """
    frame_indices = range(len(histograms))
    total_cumulative_counts = [sum(histogram) for histogram in histograms]  # Aggregate per frame

    # Create a DataFrame for plotting
    data = pd.DataFrame({
        "Frame Number": frame_indices,
        "Total Cumulative Count": total_cumulative_counts
    })

    # Plot the total cumulative pixel count across frames
    fig = px.line(
        data,
        x="Frame Number",
        y="Total Cumulative Count",
        title=f"Total Cum-Sum Count by Gray Intensity Across Frames (Category {category_num}, Video {video_num})",
        labels={"Frame Number": "Frame", "Total Cumulative Count": "Cumulative Pixel Count"}
    )

    # Highlight the scene cut frame
    fig.add_vline(
        x=cut_frame_index,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Scene Cut at Frame {cut_frame_index}",
        annotation_position="top right"
    )

    # Customize the layout
    fig.update_layout(
        title_x=0.5,
        plot_bgcolor="white",
        xaxis=dict(title="Frame Index", tickmode="linear", tick0=0, dtick=10),
        yaxis=dict(title="Total Cum-Sum by Intensity", showticklabels=False)
    )

    # Save the plot to a file
    fig.write_image(f"category{category_num}_video{video_num}_simplified_histogram.png")




def generate_delta_plot(category_num, deltas):
    """Generate a delta plot for the differences between cumulative histograms."""
    fig = go.Figure()

    # Plot data for both videos with distinct colors
    fig.add_trace(go.Scatter(
        x=[i for i in range(len(deltas[0]))],
        y=deltas[0],
        name="Video 1 (Delta)",
        line=dict(color="rgba(255, 99, 132, 0.8)")
    ))

    fig.add_trace(go.Scatter(
        x=[i for i in range(len(deltas[1]))],
        y=deltas[1],
        name="Video 2 (Delta)",
        line=dict(color="rgba(54, 162, 235, 0.8)")
    ))

    # Configure the plot layout
    fig.update_layout(
        title=f"Cumulative Delta Between Frames (Category {category_num})",
        xaxis_title="Frame Index",
        yaxis_title="Cumulative Delta Value",
        legend_title="Video Number",
        plot_bgcolor="white",
        xaxis=dict(
            tickvals=[i for i in range(0, len(deltas[0]), 50)],
            ticktext=[str(i) for i in range(0, len(deltas[0]), 50)],
            tickangle=45,
        ),
        yaxis=dict(title="Cumulative Delta Value", range=[0, max(max(deltas[0]), max(deltas[1])) * 1.2]),
    )

    fig.write_image(f"category{category_num}_cumulative_histogram_delta.png")


if __name__ == '__main__':
    process_videos()

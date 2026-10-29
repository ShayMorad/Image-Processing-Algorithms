import os

import numpy as np
import scipy.signal
import scipy.fft
import scipy.io.wavfile
import matplotlib.pyplot as plt

# Constants
SCIPY_FLOAT = 32768.0
WATERMARK_GOOD_FREQ = 21000  # High frequency for good watermark (near human hearing limit)
WATERMARK_BAD_FREQ = 500  # Frequency for bad watermark (audible)
GOOD_WATERMARK_AMPLITUDE = 0.015  # Amplitude for good watermark
BAD_WATERMARK_AMPLITUDE = 0.1  # Amplitude for bad watermark
TIME_LIMIT = 4  # Time limit for analysis in seconds
FILTERED_FREQ_RANGE = (19500, 20200)  # Frequency range to analyze for watermark detection
SINE_FREQ_RANGE = (15000, 21000)  # Frequency range to analyze for sine wave watermark
SINE_WATERMARK_LOW_BOUND = 0.400
SINE_WATERMARK_MED_BOUND = 0.570
AMPLITUDE_REMARK = " #Amplitude seems different for each file so we don't include it"


def task1_watermarking(audio_path):
    # Read original audio file
    sample_rate, audio_data = scipy.io.wavfile.read(audio_path)
    print(f"Audio File Sample rate: {sample_rate} Hz")

    # Ensure audio data is in float32
    audio_data = audio_data.astype(np.float32) / SCIPY_FLOAT

    def add_good_watermark(data):
        print("Creating a good watermark")
        watermark_freq = WATERMARK_GOOD_FREQ  # High frequency near human hearing limit
        t = np.arange(len(data)) / sample_rate  # Generate time array for each sample in the audio data
        watermark = GOOD_WATERMARK_AMPLITUDE * np.sin(2 * np.pi * watermark_freq * t)  # Very low amplitude
        return data + watermark  # Add the watermark to the original audio data

    def add_bad_watermark(data):
        print("Creating a bad watermark")
        watermark_freq = WATERMARK_BAD_FREQ  # Clearly audible frequency
        t = np.arange(len(data)) / sample_rate  # Generate time array for each sample in the audio data
        watermark = BAD_WATERMARK_AMPLITUDE * np.sin(2 * np.pi * watermark_freq * t)  # Higher amplitude
        return data + watermark  # Add the watermark to the original audio data

    # Create and save watermarked versions
    good_watermarked = add_good_watermark(audio_data)
    good_watermarked_path = './task1_good_watermark.wav'
    bad_watermarked = add_bad_watermark(audio_data)
    bad_watermarked_path = './task1_bad_watermark.wav'

    # Clip the watermarked audio data to ensure that the values are within the valid range of [-1.0, 1.0].
    # This prevents clipping or distortion in the audio by limiting the values to the maximum and minimum amplitudes.
    good_watermarked = np.clip(good_watermarked, -1.0, 1.0)
    bad_watermarked = np.clip(bad_watermarked, -1.0, 1.0)

    # Save the watermarked audio data to WAV files using scipy.io.wavfile.write
    # The 'write' function writes the audio data to a file with the specified sample rate.
    # Convert the watermarked audio to 32-bit floating-point format
    print("Saving good watermarked audio file")
    scipy.io.wavfile.write(good_watermarked_path, sample_rate, good_watermarked.astype(np.float32))
    print("Saving bad watermarked audio file")
    scipy.io.wavfile.write(bad_watermarked_path, sample_rate, bad_watermarked.astype(np.float32))

    return good_watermarked_path, bad_watermarked_path


def task2_spectrograms(audio_files_paths):
    # Create a spectrogram for each audio file to manually look for the watermarks
    for i, file in enumerate(audio_files_paths):
        output_image_path = f'./task2_file{i}_spectrogram_.png'
        plot_spectrogram(file, output_image_path, True)


def task3_speedup_method(audio_file_path):
    # Step 1: Load the audio file
    sample_rate, audio_data = scipy.io.wavfile.read(audio_file_path)

    # Step 2: Perform STFT (Short-Time Fourier Transform) to analyze frequency content
    f, t, Zxx = scipy.signal.stft(audio_data, fs=sample_rate, nperseg=2048)

    # Step 3: Plot the Spectrogram to visualize frequency content
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    plt.title(f"Spectrogram of {audio_file_path}")
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar(label='Amplitude')
    plt.show()

    # Step 4: Analyze waveform
    plt.figure(figsize=(10, 6))
    plt.plot(audio_data[:10000])  # Show first 10,000 samples for visual analysis
    plt.title(f"Waveform of {audio_file_path}")
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.show()


def plot_spectrogram(audio_path, output_image_path, should_log_compress):
    sample_rate, audio_data = scipy.io.wavfile.read(audio_path)
    print(f"Plotting Spectrogram of: {audio_path}")
    frequencies, time, Zxx = scipy.signal.stft(audio_data, sample_rate, nperseg=1024)

    plt.figure(figsize=(10, 6))
    if (should_log_compress):
        plt.pcolormesh(time, frequencies, np.log10(np.abs(Zxx) + 0.1), shading='auto', cmap='jet')
    else:
        plt.pcolormesh(time, frequencies, np.abs(Zxx), shading='auto', cmap='jet')
    plt.title(f'Spectrogram of {os.path.basename(audio_path)}')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.colorbar(label='Amplitude')
    # plt.ylim(0, 20000) # Useful for task 3 and other tasks to show a full graph of frequencies from 0 to 20k
    plt.savefig(output_image_path, format='png', dpi=300)
    print(f"Spectrogram saved to: {output_image_path}")
    plt.close()


def plot_filtered_spectrogram(audio_file_path, time_limit=TIME_LIMIT, freq_range=FILTERED_FREQ_RANGE, nperseg=1024):
    # Step 1: Load the audio file
    sample_rate, audio_data = scipy.io.wavfile.read(audio_file_path)

    # Step 2: Limit to the first 5 seconds (or specified time)
    max_samples = int(time_limit * sample_rate)
    audio_data = audio_data[:max_samples]

    # Step 3: Perform STFT on the audio data
    f, t, Zxx = scipy.signal.stft(audio_data, fs=sample_rate, nperseg=nperseg)

    # Step 4: Cut frequencies between 18kHz and 20.2kHz
    freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
    Zxx_filtered = Zxx[freq_mask, :]
    f_filtered = f[freq_mask]

    # Step 5: Filter to keep only the frequencies with the highest amplitude at each time slice
    # Find the maximum amplitude for each time slice (column) in the filtered spectrogram
    max_amplitude_per_time = np.max(np.abs(Zxx_filtered), axis=0)
    max_amplitude_mask = np.abs(Zxx_filtered) > 0.9 * max_amplitude_per_time  # Keep the highest amplitudes

    # Filter the spectrogram based on the mask
    Zxx_filtered_high = Zxx_filtered * max_amplitude_mask

    # Step 6: Plot the spectrogram of the filtered data
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f_filtered, np.abs(Zxx_filtered_high), shading='auto', cmap='jet')
    plt.title(f"Filtered Spectrogram (Freq Range: {freq_range[0]} - {freq_range[1]} Hz, Time: 0-{time_limit} s)")
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar(label='Amplitude')
    plt.show()


def detect_watermark_sine_wave(audio_file_path, freq_range=SINE_FREQ_RANGE, time_limit=5, nperseg=2048,
                               amplitude_threshold=0.05):
    print(f"Processing file: {audio_file_path}")
    # Step 1: Load the audio file
    sample_rate, audio_data = scipy.io.wavfile.read(audio_file_path)

    # Step 2: Truncate the audio to the first `time_limit` seconds
    max_samples = int(time_limit * sample_rate)
    audio_data = audio_data[:max_samples]

    # Step 3: Perform STFT
    f, t, Zxx = scipy.signal.stft(audio_data, fs=sample_rate, nperseg=nperseg)

    # Step 4: Clip to frequency range
    freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
    Zxx_filtered = Zxx[freq_mask, :]
    f_filtered = f[freq_mask]

    # Step 5: Apply amplitude threshold
    amplitude = np.abs(Zxx_filtered)
    amplitude[amplitude < amplitude_threshold] = 0

    # Step 6: Find dominant frequency for each time slice
    dominant_frequencies = []
    for time_idx in range(amplitude.shape[1]):
        if np.any(amplitude[:, time_idx] > 0):  # Check if any frequency is above the threshold
            max_freq_idx = np.argmax(amplitude[:, time_idx])
            dominant_frequencies.append(f_filtered[max_freq_idx])
        else:
            dominant_frequencies.append(0)  # No dominant frequency
    dominant_frequencies = np.array(dominant_frequencies)

    # Step 7: Detect first peak before 1 second
    time_mask_before_1sec = t < 1  # Find indices before 1 second
    first_peak_idx = np.argmax(dominant_frequencies[time_mask_before_1sec])
    first_peak_time = t[first_peak_idx]

    # Step 8: Calculate the time window for the second peak
    time_window = 1 + 4 * (first_peak_time)  # Start from 1 second and add 1.5 * the first peak time

    # Ensure the second peak search time is within the bounds of the total time
    if time_window > t[-1]:
        time_window = t[-1]  # Adjust to not exceed the last time point

    # Step 9: Detect second peak in the defined time window
    time_mask_after_1sec = (t > 1) & (t < time_window)  # Time mask after 1 second and within the time window
    second_peak_idx = np.argmax(dominant_frequencies[time_mask_after_1sec]) + np.argmax(time_mask_after_1sec)
    second_peak_time = t[second_peak_idx]

    # Calculate period and frequency
    period = second_peak_time - first_peak_time
    sine_wave_frequency = 1 / period

    # Categorize the sine wave frequency
    category = None
    real_function = None
    if sine_wave_frequency < SINE_WATERMARK_LOW_BOUND:
        category = 1
        real_function = "Approximately f ~= sin(2π⋅0.350)"
    elif SINE_WATERMARK_LOW_BOUND <= sine_wave_frequency <= SINE_WATERMARK_MED_BOUND:
        category = 2
        real_function = "Approximately f ~= sin(2π⋅0.500)"
    else:
        category = 3
        real_function = "Approximately f ~= sin(2π⋅0.650)"

    return category, sine_wave_frequency, real_function


def task2_watermark_classification(audio_file_paths):
    categories = {
        "Category 1": [[], None],  # First element is the list of files, second is the function
        "Category 2": [[], None],
        "Category 3": [[], None]
    }

    for file_path in audio_file_paths:
        category, sine_wave_frequency, real_function = detect_watermark_sine_wave(file_path)

        # Add the file to the appropriate category based on the sine wave frequency
        if category == 1:
            categories["Category 1"][0].append(file_path)  # Append file to category 1
            categories["Category 1"][1] = real_function + AMPLITUDE_REMARK  # Make sure real function is set
        elif category == 2:
            categories["Category 2"][0].append(file_path)  # Append file to category 2
            categories["Category 2"][1] = real_function + AMPLITUDE_REMARK  # Make sure real function is set
        elif category == 3:
            categories["Category 3"][0].append(file_path)  # Append file to category 3
            categories["Category 3"][1] = real_function + AMPLITUDE_REMARK  # Make sure real function is set

    return categories


def main():
    # Run tasks in sequence
    # Task 1
    print("Task 1")
    good_path, bad_path = task1_watermarking('Task 1/task1.wav')
    plot_spectrogram(good_path, './good_watermark_spectogarm.png', True)
    plot_spectrogram(bad_path, './bad_watermark_spectogarm.png', True)
    print()

    # Task 2
    print("Task 2")
    audio_files_paths = [
        'Task 2/0_watermarked.wav',
        'Task 2/1_watermarked.wav',
        'Task 2/2_watermarked.wav',
        'Task 2/3_watermarked.wav',
        'Task 2/4_watermarked.wav',
        'Task 2/5_watermarked.wav',
        'Task 2/6_watermarked.wav',
        'Task 2/7_watermarked.wav',
        'Task 2/8_watermarked.wav'
    ]

    # Create spectograms to see visually what to look for and how to hard-code the algorithm to sort
    task2_spectrograms(audio_files_paths)

    # # Create spectrograms of only 19.5kHz - 20.2kHz frequencies to see the spikes of the sine waves watermarks
    # for path in audio_files_paths:
    #     plot_filtered_spectrogram(path, time_limit=4, freq_range=(20000, 20200))

    # Classify each file to its respectable category (1 of 3) with its simple function as written in the exercise
    categories = task2_watermark_classification(audio_files_paths)
    print("Files Categories: ")
    for category_name, category_data in categories.items():
        print(f"{category_name}:")
        # Print the files in the category
        print(f"  Files: {category_data[0]}")
        # Print the sine wave function for that category
        print(f"  Real function: {category_data[1]}")
        print()  # Blank line between categories for readability

    # # Task 3
    # print("Task 3")
    # plot_spectrogram('Task 3/task3_watermarked_method1.wav', './task3_method1_spectrogram.png', True)
    # plot_spectrogram('Task 3/task3_watermarked_method2.wav', './task3_method2_spectrogram.png', True)


if __name__ == "__main__":
    main()

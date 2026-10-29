import numpy as np
import cv2

N = 512
PYRAMID_LEVEL = 10
MIN_INTENSITY = 0
MAX_INTENSITY = 255


def construct_gaus_pyr(image, levels):
    """Build a Gaussian pyramid for the given image."""
    gaussian_pyramid = [np.float32(image.copy())]
    # Start from level 1 because level 0 (original image) is already in the list
    for i in range(1, levels):
        # Create gaussian of next level by taking the previous level and applying cv2.pyrDown
        next_level_gaussian_image = np.float32(cv2.pyrDown(gaussian_pyramid[i - 1]))
        gaussian_pyramid.append(next_level_gaussian_image)
    return gaussian_pyramid


def construct_lap_pyr(gaussian_pyramid):
    """Build a Laplacian pyramid from the given Gaussian pyramid."""
    laplacian_pyramid = []
    for i in range(len(gaussian_pyramid) - 1):
        # Expand g(n+1)
        expand_Gn1 = cv2.pyrUp(gaussian_pyramid[i + 1])
        # Resize the expanded g(n+1) to match the dimensions of g(n)
        resize_expanded_Gn1 = cv2.resize(expand_Gn1, (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
        Gn = gaussian_pyramid[i]
        # Get the laplacian by subtracting from g(n) the changed expanded and resized g(n+1)
        laplacian = Gn - resize_expanded_Gn1
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1])  # Last level is the Gaussian pyramid's smallest image
    return laplacian_pyramid


def sum_constructed_blended_pyr(laplacian_pyramid):
    """Reconstruct an image from its Laplacian pyramid."""
    reconstructed_image = laplacian_pyramid[-1]
    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        # Expand
        reconstructed_image = cv2.pyrUp(reconstructed_image)
        # Resize
        reconstructed_image = cv2.resize(reconstructed_image,
                                         (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0]))
        reconstructed_image = reconstructed_image + laplacian_pyramid[i]
    return reconstructed_image


def construct_blended_lap_pyr(laplacian_pyramid1, laplacian_pyramid2, gaussian_mask_pyramid):
    """Blend two Laplacian pyramids using a Gaussian mask pyramid."""
    blended_pyramid = []
    for img1_lap, img2_lap, mask_gaus in zip(laplacian_pyramid1, laplacian_pyramid2, gaussian_mask_pyramid):
        # If mask = 1, it means the color was white which means we want to take from image 1
        # If mask = 0, it means the color was black which means we want to take from image 2
        blended = img1_lap * mask_gaus + img2_lap * (1 - mask_gaus)
        blended_pyramid.append(blended)
    return blended_pyramid


def blend_images(image1_path, image2_path, mask_path):
    """Blend two images using pyramids."""
    # Load images and mask
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    mask = cv2.imread(mask_path) / 255
    mask = np.round(mask).astype(np.uint8)

    # Resize images and mask to match dimensions
    image1 = cv2.resize(image1, (N, N))
    image2 = cv2.resize(image2, (N, N))
    mask = cv2.resize(mask, (N, N))

    # Calculate pyramid levels as shown in lecture
    pyramid_levels = min(PYRAMID_LEVEL, int(np.floor(np.log2(N))))

    # Build Gaussian pyramids
    gp_image1 = construct_gaus_pyr(image1, pyramid_levels)
    gp_image2 = construct_gaus_pyr(image2, pyramid_levels)
    gp_mask = construct_gaus_pyr(mask, pyramid_levels)

    # # Normalize mask to [0, 1] for blending as shown in lecture
    # gp_mask = [m / 255.0 for m in gp_mask]

    # Build Laplacian pyramids
    lp_image1 = construct_lap_pyr(gp_image1)
    lp_image2 = construct_lap_pyr(gp_image2)

    # Blend the Laplacian pyramids
    blended_pyramid = construct_blended_lap_pyr(lp_image1, lp_image2, gp_mask)
    # Sum the blended image pyramid and clip to RGB values
    blended_image = np.clip(sum_constructed_blended_pyr(blended_pyramid), MIN_INTENSITY, MAX_INTENSITY).astype(np.uint8)

    # cv2.imshow("Blended Image", blended_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite("blended_image.jpg", blended_image)
    return blended_image


def compute_fourier(image_path):
    """Calculate Fourier Transform of an image."""
    # Load image in grayscale as in lecture it was shown on gray images
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (N, N))
    fft = np.fft.fft2(image)  # Compute the Fourier Transform
    fft_centered = np.fft.fftshift(fft)  # Center by the zero frequency
    return fft_centered


def create_frequency_domain_gaussian(shape, spread=32, high_pass=False):
    """Create a Gaussian filter in the frequency domain."""
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    x = np.linspace(-center_col, center_col, cols)
    y = np.linspace(-center_row, center_row, rows)
    X, Y = np.meshgrid(x, y)
    distance_squared = X ** 2 + Y ** 2
    gaussian = np.exp(-distance_squared / (2 * (spread ** 2)))

    if high_pass:
        gaussian = 1 - gaussian
    return gaussian


def create_hybrid(image1_path, image2_path):
    """Create a hybrid image from two images."""
    image1_fourier_spectrom = compute_fourier(image1_path)
    image2_fourier_spectrom = compute_fourier(image2_path)
    target_shape = image1_fourier_spectrom.shape
    gauss_filter_lowpass = create_frequency_domain_gaussian(target_shape)

    # gaussian_normalized = cv2.normalize(gauss_filter_lowpass, None, MIN_INTENSITY, MAX_INTENSITY,
    #                                     cv2.NORM_MINMAX).astype(np.uint8)
    # cv2.imwrite("gaussian_filter_lowpass.jpg", gaussian_normalized)

    image1_filtered = image1_fourier_spectrom * gauss_filter_lowpass

    image2_filtered = image2_fourier_spectrom * (1 - gauss_filter_lowpass)

    hybrid_spectrom = image1_filtered + image2_filtered
    hybrid_image = np.fft.ifft2(np.fft.ifftshift(hybrid_spectrom))
    hybrid_image_real = np.real(hybrid_image)
    hybrid_image_normalized = cv2.normalize(hybrid_image_real, None, MIN_INTENSITY, MAX_INTENSITY, cv2.NORM_MINMAX)
    hybrid_image_normalized = hybrid_image_normalized.astype(np.uint8)

    # cv2.imshow("Hybrid Image", hybrid_image_normalized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite("hybrid_image.jpg", hybrid_image_normalized)
    return hybrid_image_normalized


def save_pyramid_levels(pyramid, output_prefix, is_laplacian=False):
    for i in range(min(3, len(pyramid))):  # Save up to 3 levels
        level = pyramid[i]
        if is_laplacian:
            # Normalize Laplacian levels to [0, 255] for visualization
            level = cv2.normalize(level, None, MIN_INTENSITY, MAX_INTENSITY, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            level = level.astype(np.uint8)
        output_path = f"{output_prefix}_level_{i}.jpg"
        cv2.imwrite(output_path, level)


def visualize_pyramids(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (N, N))  # Resize to match the expected input size

    # Build Gaussian pyramid
    gaussian_pyramid = construct_gaus_pyr(image, PYRAMID_LEVEL)
    save_pyramid_levels(gaussian_pyramid, "gaussian_pyramid")

    # Build Laplacian pyramid
    laplacian_pyramid = construct_lap_pyr(gaussian_pyramid)
    save_pyramid_levels(laplacian_pyramid, "laplacian_pyramid", is_laplacian=True)


if __name__ == "__main__":
    # Task 1.1
    # Perform blending
    blend_images("cat.jpg", "lion.jpg", "mask.jpg")
    # visualize_pyramids("input_images/cat.jpg")

    # Task 1.2
    # Create a hybrid image
    create_hybrid("cute_cat.jpg", "cute_dog.jpg")

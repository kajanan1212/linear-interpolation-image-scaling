import cv2
import numpy as np


def load_image(file_name: str):
    image = cv2.imread(file_name)
    return image


def write_image(file_name: str, image):
    cv2.imwrite(file_name, image)


def rgb_to_gray_scale(image):
    RGB = 0.21, 0.72, 0.07
    weights = np.array(RGB).reshape(3, 1)

    gray_scale_image = np.dot(image, weights).astype(np.uint8)

    return gray_scale_image


def resample_gs_image(times, gs_image):
    gs_image = gs_image.reshape(gs_image.shape[:2])
    row, col = gs_image.shape
    s_row, s_col = round(row * times), round(col * times)

    c, r = np.meshgrid(np.arange(s_col), np.arange(s_row))

    o_r = r / times
    o_c = c / times

    x1 = np.floor(o_r).astype(int)
    y1 = np.floor(o_c).astype(int)

    x2 = np.minimum(x1 + 1, row - 1)
    y2 = np.minimum(y1 + 1, col - 1)

    dx = o_r - x1
    dy = o_c - y1

    s_px = (1 - dx) * (1 - dy) * gs_image[x1, y1] \
        + dx * (1 - dy) * gs_image[x2, y1] \
        + (1 - dx) * dy * gs_image[x1, y2] \
        + dx * dy * gs_image[x2, y2]

    s_px = np.clip(s_px, 0, 255).astype(np.uint8)

    resampled_gs_image = s_px.reshape((s_row, s_col, 1)).astype(np.uint8)

    return resampled_gs_image


def SSE(image_1, image_2):
    return np.sum(np.square(image_1 - image_2))


def MSE(image_1, image_2):
    return np.mean(np.square(image_1 - image_2))


if __name__ == "__main__":
    # 1.Load an image from the disk
    image = load_image("images/01-img.jpg")
    print(f"Original image's shape: {image.shape}")

    # 2.Convert the image to gray-scale (8bpp format)
    gs_image = rgb_to_gray_scale(image)
    print(f"Gray scale image's shape: {gs_image.shape}")
    write_image("images/02-gs-img.jpg", gs_image)

    # 3.Re-sample the image such that the size is 0.7 times it original dimensions 
    # using linear interpolation method and save the image.
    TIMES = 0.7

    down_sampled_gs_image = resample_gs_image(TIMES, gs_image)
    print(f"Down sampled gray scale image's shape: {down_sampled_gs_image.shape}")
    write_image("images/03-ds-img.jpg", down_sampled_gs_image)

    # 4.Re-sample the image created in (step 3) back to its original size and save the image.
    up_sampled_gs_image = resample_gs_image(1 / TIMES, down_sampled_gs_image)
    print(f"Up sampled gray scale image's shape: {up_sampled_gs_image.shape}")
    write_image("images/04-us-img.jpg", up_sampled_gs_image)

    # 5.Compute the sum of the average of the squared difference between pixels 
    # in the original image (in step 2) and the re-samples image in (step 4).
    sse = SSE(gs_image, up_sampled_gs_image)
    mse = MSE(gs_image, up_sampled_gs_image)
    print(f"Sum of Square Error of reconstruction: {sse}")
    print(f"Mean Square Error of reconstruction: {mse}")

import cv2 as cv
import matplotlib.pyplot as plt

def print_hist(img, channels_list: list[str], name: str, color: str = "blue"):
    img_split = cv.split(img)
    assert (len(channels_list) == len(img_split))

    # Compute and plot histograms for Lab color space
    fig, axes = plt.subplots(len(channels_list), 1, figsize=(10, 8))
    fig.suptitle(f'Lab Channel Histograms for {name}')

    for i, (channel, title) in enumerate(zip(img_split, channels_list)):
        hist_lab = cv.calcHist([channel], [0], None, [100], [0, 256])
        axes[i].bar(range(len(hist_lab)), hist_lab.flatten(), color=color)
        axes[i].set_title(title)

    plt.tight_layout()
    plt.show()
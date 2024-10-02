import cv2 as cv
import matplotlib.pyplot as plt


def plot_hist_from_img(img, channels_list: list[str], name: str, color: str = "blue"):
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


def plot_hist_from_list(histograms, index, color_space='Lab'):
    if index >= len(histograms) or histograms[index] is None:
        print(f"Index {index} is out of range or has no histogram data.")
        return

    # Extract the desired histogram based on the color space
    histogram = histograms[index]
    if color_space == 'Lab':
        # Lab histograms are the first three parts of the concatenated histogram
        L_hist = histogram[:256]
        a_hist = histogram[256:512]
        b_hist = histogram[512:768]

        # Plot each channel
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.bar(range(256), L_hist, color='black')
        plt.title('L Channel Histogram')
        plt.xlabel('Intensity Value')
        plt.ylabel('Number of pixels')

        plt.subplot(1, 3, 2)
        plt.bar(range(256), a_hist, color='red')
        plt.title('a Channel Histogram')
        plt.xlabel('Intensity Value')

        plt.subplot(1, 3, 3)
        plt.bar(range(256), b_hist, color='blue')
        plt.title('b Channel Histogram')
        plt.xlabel('Intensity Value')

    elif color_space == 'HSV':
        # HSV histograms follow Lab histograms, assuming the size of each bin
        H_hist = histogram[768:768 + 180]
        S_hist = histogram[768 + 180:768 + 180 + 256]
        V_hist = histogram[768 + 180 + 256:768 + 180 + 256 + 256]

        # Plot each channel
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.bar(range(180), H_hist, color='orange')
        plt.title('Hue Channel Histogram')
        plt.xlabel('Hue Value')
        plt.ylabel('Frequency')

        plt.subplot(1, 3, 2)
        plt.bar(range(256), S_hist, color='green')
        plt.title('Saturation Channel Histogram')
        plt.xlabel('Intensity Value')

        plt.subplot(1, 3, 3)
        plt.bar(range(256), V_hist, color='purple')
        plt.title('Value Channel Histogram')
        plt.xlabel('Intensity Value')

    else:
        print("Invalid color space. Please choose 'Lab' or 'HSV'.")
        return

    # Display the plots
    plt.tight_layout()
    plt.show()


def plot_hist_task1(original_image, image_colorspace, color_space='Lab'):
      from compute_descriptors import compute_histogram
      plt.figure(figsize= (20,10))

      #Plot the original image
      plt.subplot(2,2,1)
      plt.imshow(cv.cvtColor(original_image, cv.COLOR_BGR2RGB))
      plt.title('Original Image')
      plt.axis('off')

      # Plot the image in the specified color space
      plt.subplot(2, 2, 2)
      plt.imshow(image_colorspace)
      plt.title(f'{color_space} Image')
      plt.axis('off')


      if color_space == 'Lab':
        # Lab histograms are the first three parts of the concatenated histogram
        L_hist = compute_histogram(image_colorspace, [0], [256], [0, 256])
        a_hist = compute_histogram(image_colorspace, [1], [256], [0, 256])
        b_hist = compute_histogram(image_colorspace, [2], [256], [0, 256])

        # Plot each channel
        plt.subplot(2, 3, 4)
        plt.bar(range(256), L_hist, color='black')
        plt.title('L Channel Histogram')
        plt.xlabel('Intensity Value')
        plt.ylabel('Number of pixels')

        plt.subplot(2, 3, 5)
        plt.bar(range(256), a_hist, color='red')
        plt.title('a Channel Histogram')
        plt.xlabel('Intensity Value')
        

        plt.subplot(2, 3, 6)
        plt.bar(range(256), b_hist, color='blue')
        plt.title('b Channel Histogram')
        plt.xlabel('Intensity Value')
    

      elif color_space == 'HSV':
        # HSV histograms follow Lab histograms, assuming the size of each bin
        H_hist = compute_histogram(image_colorspace, [0], [180], [0, 180])
        S_hist = compute_histogram(image_colorspace, [1], [256], [0, 256])
        V_hist = compute_histogram(image_colorspace, [2], [256], [0, 256])
       
        plt.subplot(2, 3, 4)
        plt.bar(range(180), H_hist, color='orange')
        plt.title('Hue Channel Histogram')
        plt.xlabel('Hue Value')
        plt.ylabel('Number of pixels')

        plt.subplot(2, 3, 5)
        plt.bar(range(256), S_hist, color='green')
        plt.title('Saturation Channel Histogram')
        plt.xlabel('Intensity Value')
        

        plt.subplot(2, 3, 6)
        plt.bar(range(256), V_hist, color='purple')
        plt.title('Value Channel Histogram')
        plt.xlabel('Intensity Value')
       

      else:
        print("Invalid color space. Please choose 'Lab' or 'HSV'.")
        return

    # Display the plots
      plt.tight_layout(pad= 3.0)
      plt.show()

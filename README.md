<p align="center">
<h3 align="center">Module C1 Project</h3>

  <p align="center">
    Project for the Module C1 in Master's in Computer Vision
<br>
    <a href="https://github.com/EymoLabs/eymo-cloud-rs/issues/new?template=bug.md">Report bug</a>
    ·
    <a href="https://github.com/EymoLabs/eymo-cloud-rs/issues/new?template=feature.md&labels=feature">Request feature</a>
  </p>
</p>


## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
- [Project Structure](#project-structure)
- [WEEK 1: Tasks](#week-1-tasks)
  - [Task 1: Museum and query image descriptors (BBDD & QSD1)](#task-1-museum-and-query-image-descriptors-bbdd--qsd1)
  - [Task 2: Selection and implementation of similarity measures to compare images](#task-2-selection-and-implementation-of-similarity-measures-to-compare-images)
  - [Task 3: Implement retrieval system (retrieve top K results)](#task-3-implement-retrieval-system-retrieve-top-k-results)
    - [Parameters](#parameters)
    - [Process Description](#process-description)
    - [Best Result Methods](#best-result-methods)
  - [Task 4: Processing the QST1 Testing Dataset](#task-4-processing-the-qst1-testing-dataset)
- [WEEK 2: Tasks](#week-2-tasks)
  - [Task 1: Implement 3D/2D block and hierarchical histograms)](#task-1-implement-3d2d-block-and-hierarchical-histograms)
  - [Task 2: Test query system using query set QSD1-W2 development and evaluate retrieval results](#task-2-test-query-system-using-query-set-qsd1-w2-development-and-evaluate-retrieval-results)
  - [Task 3: Remove background using the background color for the QSD2-W2 dataset](#task-3-remove-background-using-the-background-color-for-the-qsd2-w2-dataset)
  - [Task 4: Background removal evaluation](#task-4-background-removal-evaluation)
  - [Task 5: Remove background + Retrieval system (QSD2-W2)](#task-5-remove-background--retrieval-system-qsd2-w2)
  - [Task 6: Processing the QST1 and QST2 Testing Dataset](#task-6-processing-the-qst1-and-qst2-testing-dataset)

- [Team Members](#team-members)
- [License](#license)


## Introduction

This project is developed as part of the Master's program in Computer Vision in Barcelona, specifically for the course **C1: Introduction to Human and Computer Vision** during the first academic semester. 

The aim of the project is to implement computer vision techniques to retrieve query images from a museum database. By utilizing methods such as feature extraction and histogram comparison, the system is designed to accurately identify and retrieve artworks based on visual content, thereby enhancing the accessibility and exploration of museum collections.

## Installation

This section will guide you through the installation process of the project and its testing.

### Prerequisites
The following prerequisites must be followed:
- Python >= v3.8

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MCV-2024-C1-Project/Team4.git
   ```

2. **Navigate to the corresponding week's folder:**
   
   For example, to enter the folder for week 1:
   ```bash
   cd week1
   ```
   
4. **Create a virtual environment:**
   ```bash
   python -m venv env
   ```

5. **Activate the virtual environment:**
    - On Windows:
      ```bash
      .\env\Scripts\activate
      ```
    - On MacOS/Linux:
      ```bash
      source env/bin/activate
      ```

6. **Install the dependencies:**
  ```bash
  pip install -r requirements.txt
  ```

   
## Project Structure

Within the downloaded repository, you'll find the following directories and files, logically grouping common assets. The data folders need to be downloaded and decompressed from the provided links:

- **BBDD:** [Download here](https://e-aules.uab.cat/2024-25/pluginfile.php/398404/mod_page/content/186/BBDD.zip)
- **qsd1_w1:** [Download here](https://e-aules.uab.cat/2024-25/pluginfile.php/398404/mod_page/content/186/qsd1_w1.zip?time=1602013828018)
- **qsd2_w1:** [Download here](https://e-aules.uab.cat/2024-25/pluginfile.php/398404/mod_page/content/188/qsd2_w2.zip?time=1728249194853)
- **qst1_w1:** [Download here](https://drive.google.com/file/d/1GbI0ik3IeUNN51dJBewpeWL_Dm_uNhNT/view?usp=drive_link)
- **qst1_w2:** [Download here](https://drive.google.com/file/d/1TLCfm4uYXf40PKUxRW4r_8vqn4yEKatl/view?usp=drive_link)
- **qst2_w2:** [Download here](https://drive.google.com/file/d/1t_9DZVH7uxykdDF6c9M3Cim6RpMlFubw/view?usp=drive_link)

Once downloaded and extracted, the project structure will look like this:

    Team4/
    ├── data/
    │   ├── BBDD/
    │   ├── qsd1_w1/
    │   ├── qsd2_w1/
    │   ├── qst1_w1/
    │   ├── qst1_w2/
    │   └── qst2_w2/
    ├── week1/
    │   └── ...
    ├── week2/
    │   └── ...

<h2 align="center">WEEK 1: Tasks</h2>

### Project Structure

    week1/
    ├── evaluation/
    │   ├── bbox_iou.py
    │   └── evaluation_funcs.py
    ├── src/
    │   ├── average_precision.py
    │   ├── compute_db_descriptors.py
    │   ├── compute_descriptors.py
    │   ├── compute_img_descriptors.py
    │   ├── compute_similarities.py
    │   ├── main.py
    │   ├── metrics.py
    │   └── utils.py
    ├── utils/
    │   ├── plot_results.py
    │   └── print_dict.py
    
### Task 1: Museum and query image descriptors (BBDD & QSD1)

- **Index the Database (BBDD):** Generate descriptors offline and saves them in a `.pkl` file.
  ```bash
  python .\compute_db_descriptors.py
  ```
- **Compute image descriptors (QSD1):**
  
  Methods:
   - **Color space CieLab:** Histograms for the L, a, and b channels.
     
     Example command:
     ```bash
     python .\compute_img_descriptors.py data\qsd1_w1 00001.jpg hist_lab 
     ```
     ![Figure_1](https://github.com/user-attachments/assets/b661403c-8a9b-4afb-bae6-4519f5e15ec5)

   - **Color space HSV:** Histograms for the Hue, Saturation, and Value channels.
     
     Example command:
     ```bash
     python .\compute_img_descriptors.py data\qsd1_w1 00001.jpg hist_hsv
     ```
     ![Figure_2](https://github.com/user-attachments/assets/e6f45adb-9c68-478c-bc40-266303ba2558)

### Task 2: Selection and implementation of similarity measures to compare images

The measures used are implemented using the OpenCV library with the function ```cv::compareHist```, that compares two dense or two sparse histograms using the specified method, or have been manually defined.

- **Color space CieLab:** The optimal similarity measure is the Lorentzian distance:
  
  <img src="https://github.com/user-attachments/assets/40107771-8524-47d8-92d6-3708a4a571a6" alt="image" width="300"/>


  `measure = Lorentzian`

- **Color space HSV:** The optimal similarity measure is the Canberra distance:

  <img src="https://github.com/user-attachments/assets/f210d0cb-031a-4cb3-97a1-d80614962dd5" alt="image" width="300"/>



  `measure = Canberra`

### Task 3: Implement retrieval system (retrieve top K results)

For this task, the `main` function is used.

#### Parameters:

The following parameters need to be passed via the command line when running the script:

- `color_space`: The color space used for computing the descriptors. Options: `Lab`, `HSV`.
- `similarity_measure`: The similarity measure used for comparing the images. Options:
  - `HISTCMP_CORREL`
  - `HISTCMP_CHISQR`
  - `HISTCMP_INTERSECT`
  - `HISTCMP_BHATTACHARYYA`
  - `HISTCMP_HELLINGER`
  - `HISTCMP_CHISQR_ALT`
  - `HISTCMP_KL_DIV`
  - `Manhattan`
  - `Lorentzian`
  - `Canberra`
- `k_value`: The number of top results to retrieve. Example: `1`, `5`.
- `query_path`: The path to the query dataset.
- `is_test`: A flag to indicate if the model is in testing mode. Options: `True` (without ground truth) or `False` (with ground truth).

#### Process Description:

This function first computes the image descriptors (CieLab Histograms or HSV histograms) for all images in QST1 and saves them in a `.pkl` file, based on the specified color space argument.

Next, it reads the `.pkl` files containing the computed image descriptors for both the query dataset (QST1) and the museum dataset (BBDD, computed offline).

For each image in the query set (QST1), it calculates the similarity measure against all museum images (BBDD) and stores the top K indices of the most similar museum images for each query image in a `.pkl` file (saved in the path specified by the `query_path` parameter).

Finally, the evaluation of the system is conducted using **mAP@K (mean Average Precision at K)**, which involves calculating the Average Precision for each value of `k` from `1` to `K` (AP@K) and then taking the mean across all queries (mAP@K).

This metric indicates how effectively the system returns the results for each case.


#### Best Result Methods

The best results are obtained using the following methods:

- **Method 1: Lab - Lorentzian Distance**

  Terminal commands for `k=1` and `k=5`:

  ```bash
  python .\main.py Lab Lorentzian 1 data\qsd1_w1 False
  python .\main.py Lab Lorentzian 5 data\qsd1_w1 False
  ```

- **Method 2: HSV - Canberra Distance**

  Terminal commands for `k=1` and `k=5`:
  
  ```bash
  python .\main.py HSV Canberra 1 data\qsd1_w1 False
  python .\main.py HSV Canberra 5 data\qsd1_w1 False
  ```

#### Results

| Method                          | mAP@1          | mAP@5           |
|---------------------------------|----------------|------------------|
| **Lab - Lorentzian Distance**    | 0.533          | 0.582 |
| **HSV - Canberra Distance** | 0.700         | 0.734 |

### Task 4: Processing the QST1 Testing Dataset

To apply the algorithm to the QST1 testing dataset, you need to set the `is_test` flag to `True` in the program's configuration. When `is_test` is enabled, the program will perform image comparisons based on your chosen parameters and save the retrieval results in a pickle file. 

The terminal commands for testing dataset with our two methods are:

- **Method 1: Lab - Lorentzian Distance**
```bash
python .\main.py Lab Lorentzian 10 data\qst1_w1 True
```
- **Method 2: HSV - Canberra Distance**
```bash
python .\main.py HSV Canberra 10 data\qst1_w1 True
```

Since the testing dataset does not have ground truth labels, the Mean Average Precision (mAP@k) cannot be calculated for this task.

<h2 align="center">WEEK 2: Tasks</h2>

### Project Structure

    week2/
    ├── evaluation/
    │   ├── bbox_iou.py
    │   └── evaluation_funcs.py
    ├── src/
    │   ├── average_precision.py
    │   ├── background_removal.py
    │   ├── compute_db_descriptors.py
    │   ├── compute_similarities.py
    │   ├── histograms.py
    │   ├── main.py
    │   └── metrics.py
    ├── utils/
    │   ├── plot_results.py
    │   └── print_dict.py

### Task 1: Implement 3D/2D block and hierarchical histograms
- **Block 3D histograms scheme:**
  ![image](https://github.com/user-attachments/assets/8b992878-43a0-4bc3-a7e1-6a00ac8cb558)

  Before using the flatten() function, we get a 3D histogram. The computed histogram for image 00001.jpg in the HSV color space when it is not divided into blocks is shown below:
  ![image](https://github.com/user-attachments/assets/71417b4a-8a82-4b2a-a19b-ffce44a32c0f)
  Since only 2 bins/channel are used, the range of values for each channel is divided into 2 levels. Therefore, pixels values can fall into 8 possible colors. Size and color are used in this scatter to represent the amount of pixels of each color that are present in the image. By using the flatten() function a 8-element vector is obtained conatining the number of pixels for each color (the flattened histogram is shown in the algorithm scheme).


  
- **Hierarchical 3D histograms scheme:**
  ![image](https://github.com/user-attachments/assets/cdc3d024-7d63-481d-8ea4-0b8ead6d2649)


### Task 2: Test query system using query set QSD1-W2 development and evaluate retrieval results
- **Index the Database (BBDD):** Generate descriptors offline and save them in a `.pkl` file.
  ```bash
  # Block 3D histograms
  python compute_db_descriptors.py --color_space=HSV --num_blocks=256 --num_bins=4 

  # Hierarchical 3D histograms
  python compute_db_descriptors.py --color_space=HSV --num_levels=5 --num_bins=4 --is_pyramid=True   
  ```
  
- **Compute image descriptors (QSD1) and evaluate retrieval results:**
  ```bash
  # Block 3D histograms
  python main.py ./data/qsd1_w1 --color_space=HSV --num_blocks=256 --num_bins=4 --similarity_measure=HISTCMP_CHISQR_ALT --k_value=1
  
  # Hierarchical 3D histograms
  python main.py ./data/qsd1_w1 --color_space=HSV --num_levels=5 --num_bins=4 --similarity_measure=HISTCMP_CHISQR_ALT --k_value=1 --is_pyramid=True
  ```

#### Results

| Method                          | mAP@1          | 
|---------------------------------|----------------|
| **Block 3D histograms**         | 0.8            |
| **Hierarchical 3D histograms**  | 0.8            |

### Task 3: Remove background using the background color for the QSD2-W2 dataset

This algorithm removes the background from images by leveraging the S (saturation) and V (value) channels in the HSV color space. By isolating the foreground based on these channels, we achieve a clear separation of the main subject from the background.

#### Algorithm Breakdown:

1. **Convert image to HSV color space:**
   1. This is because the S and V channels are useful in isolating color and brightness, making it easier to separate the foreground from the background.

2. **Extract border pixels for S and V channels:** Border pixels (*10 pixels*) in the S and V channels are extracted from the top, bottom, left, and right edges of the image. These are used to determine the predominant background colors:
   1. **S Border Pixels**: The maximum value among these pixels is used to define the S threshold for background separation.
   2. **V Border Pixels**: The minimum value among these pixels is used to define the V threshold.

3. **Threshold adjustment:** Based on the extracted border pixels, thresholds are adjusted:
   1. If the maximum S border value exceeds 70% of *max(S)=255*, it uses the 97th percentile of the S border pixels.
   2. If the minimum V border value is below 30% of *max(V)=255*, it uses the 3rd percentile of the V border pixels.

4. **Apply thresholds to create foreground mask:** The adjusted thresholds are applied to the S and V channels of the entire image:
   1. Pixels with values lower than the S threshold and higher than the V threshold are set to 0, marking the background.
   2. Pixels meeting the criteria remain as foreground pixels, marked with a value of 1.

5. **Refine mask using border pixels:** The mask’s outer edges are cleared (*set to 0*) to remove any artifacts near the image borders.

6. **Fill surrounded pixels:** The function ```fill_surrounded_pixels()``` is used to fill in any isolated pixels completely surrounded by foreground pixels, refining the mask.

7. **Morphological operations:** A morphological open operation removes noise using a small *5x5* kernel. Then, a morphological close operation smooths the mask and merges any remaining small holes using a larger *50x50* kernel. The kernel sizes is a hyperparameter to be adjusted based on the image size.

8. **Combine foreground and background:**
   1. The foreground is isolated by applying the mask to the original image.
   2. The background mask is inverted and subtracted from the original image to create a clean final output with only the foreground.

> **Note:**  
> The **threshold adjustment** in 3rd point is designed to prevent issues with objects that are positioned near the borders of the image. By using the 97th percentile for the S channel and the 3rd percentile for the V channel, we ensure that border-adjacent foreground elements are not mistakenly classified as background, maintaining accuracy even when objects are close to the edges.

#### Usage:
```bash
# Without score computation
python background_removal.py data/qsd2_w1
# With score computation
python background_removal.py data/qsd2_w1 --score=True

## If using python3 and the script is not executable, use:
python3 background_removal.py data/qsd2_w1
python3 background_removal.py data/qsd2_w1 --score=True
```

<table>
  <tr>
    <td style="text-align: center;">
      <img src="https://github.com/user-attachments/assets/735b5fd6-9301-463f-a07d-7366ac0aced3" alt="Imagen 1" width="200"/>
      <p>Example Image</p>
    </td>
    <td style="text-align: center;">
      <img src="https://github.com/user-attachments/assets/6e11dc2a-6a6c-4eaf-9fd3-8ccc8bad5cdc" alt="Imagen 2" width="200"/>
      <p>Predicted Mask</p>
    </td>
    <td style="text-align: center;">
      <img src="https://github.com/user-attachments/assets/40aec7ee-cd25-4dd2-b498-cc121b52e298" alt="Imagen 3" width="215"/>
      <p>Cropped Image</p>
    </td>
  </tr>
</table>


### Task 4: Background removal evaluation 
The algorithm evaluates the accuracy of background removal using the **global F1 score, precision, and recall metrics**. This process compares the algorithm-generated masks with the provided ground truth masks to assess how accurately the background was removed.

#### Evaluation process

1. **Loading ground truth and predicted masks**  
   - The `load_masks()` function reads all image masks from a specified folder. It distinguishes between the original ground truth masks (saved as `.png` files) and the predicted masks (saved as `_mask.png` files).
   - These masks are aligned by filename, ensuring each predicted mask matches its corresponding ground truth mask.

2. **Calculating F1 score**  
   The `global_f1_score()` function iterates through each pair of predicted and ground truth masks and calculates the number of:
   - **True Positives (TP):** Pixels correctly identified as foreground (both predicted and ground truth are 255).
   - **False Positives (FP):** Pixels incorrectly identified as foreground (predicted as 255, but ground truth is 0).
   - **False Negatives (FN):** Pixels incorrectly identified as background (predicted as 0, but ground truth is 255).

3. **Global precision and recall calculation**  
   - **Precision:** The ratio of true positives to all predicted positives (`TP / (TP + FP)`). Precision shows the accuracy of detected foreground pixels.
   - **Recall:** The ratio of true positives to all actual positives (`TP / (TP + FN)`). Recall measures the algorithm's ability to capture all relevant foreground pixels.

4. **Global F1 score calculation:** The F1 score is the harmonic mean of precision and recall, representing the balance between these two metrics.

This output reflects the algorithm's overall accuracy in foreground-background separation across the entire dataset. To evaluate the algorithm use the commands from Task 3 with the `--score=True` flag.

#### Our results

| Metric        | Value |
|---------------|-------|
| Global F1 Score | 0.97  |
| Global Precision | 0.95  |
| Global Recall    | 0.98  |

### Task 5: Remove background + Retrieval system (QSD2-W2)
The final task combines the background removal algorithm with the retrieval system to enhance the accuracy of image retrieval. By removing the background from query images, the system can focus on the main subject, improving the quality of image comparisons and retrieval results.

#### Follow the steps below to run the combined system:
1. **Run the background removal algorithm:**  
   Execute the background removal script with the specified dataset path. This will generate the processed images with the background removed in the folder ```/masked``` inside the dataset directory.
   ```bash
   python background_removal.py data/qsd2_w1
   
   # Use python3 if necessary
   python3 background_removal.py data/qsd2_w1
   ```
2. **Move Ground Truth file:**  
   Move the `gt_corresps.pkl` file from the QSD2-W2 dataset to the `/masked` folder. This file contains the ground truth correspondences for the query images.

3. **Run the retrieval system:**  
   Execute the retrieval system script with the specified dataset path. This will perform image comparisons using the processed images and generate the retrieval results.
   
   The following example uses HSV color space, 256 blocks, 2 bins, Canberra distance, and retrieves the top k=1 result:
   ```bash
   python main.py ./data/qsd2_w1/masked --color_space=HSV --num_levels=5 --num_bins=4 --similarity_measure=HISTCMP_CHISQR_ALT --k_value=1 --is_pyramid=True
   
   # Use python3 if necessary
   python3 main.py ./data/qsd2_w1/masked --color_space=HSV --num_levels=5 --num_bins=4 --similarity_measure=HISTCMP_CHISQR_ALT --k_value=1 --is_pyramid=True
   ```
4. **Evaluate the retrieval results:**  
    The system will generate retrieval results for the processed images as well as evaluate the system using the ground truth correspondences. The evaluation metrics include mAP@k.


### Task 6: Processing the QST1 and QST2 Testing Dataset

#### QST-1 dataset
To apply the algorithm to the QST1 testing dataset, you need to set the `is_test` flag to `True` in the program's configuration. When `is_test` is enabled, the program will perform image comparisons based on your chosen parameters and save the retrieval results in a pickle file. 

The best results for the QSD1 were obtained by using the following parameters (therefore, we are using the same parameters for the QST1 dataset):
```bash
python main.py ./data/qst1_w2 --color_space=HSV --num_levels=5 --num_bins=4 --similarity_measure=HISTCMP_CHISQR_ALT --k_value=10 --is_test=True --is_pyramid=True
```

#### QST-2 dataset
> **Note:**  
> We use the same parameters from QST1 dataset to retrieve similarities in QST2.

For the QST-2 dataset we follow the steps below to run the background removal, as well as the retrieval system:
```bash
# Remove background
python background_removal.py data/qst2_w1

# Retrieve similarities from BBDD
python main.py ./data/qst2_w1/masked --color_space=HSV --num_levels=5 --num_bins=4 --similarity_measure=HISTCMP_CHISQR_ALT --k_value=10 --is_test=True --is_pyramid=True
```

<h2 align="center">WEEK 3: Tasks</h2>

### Project Structure --> arreglar al final

    week3/
    ├── evaluation/
    │   ├── bbox_iou.py
    │   └── evaluation_funcs.py
    ├── src/
    │   ├── average_precision.py
    │   ├── compute_db_descriptors.py
    │   ├── compute_descriptors.py
    │   ├── compute_img_descriptors.py
    │   ├── compute_similarities.py
    │   ├── main.py
    │   ├── metrics.py
    │   └── utils.py
    ├── utils/
    │   ├── plot_results.py
    │   └── print_dict.py
    
### Task 1: Filter noise with linear or non-linear filters

#### Linear Filters

- **Gaussian Filter:**

The Gaussian filter is a linear filtering technique commonly used in image processing to reduce noise and smooth images. It works by convolving the image with a Gaussian function, which gives higher weight to pixels closer to the center of the kernel and lower weight to those further away. This results in a weighted average of the neighboring pixels, effectively blurring the image while preserving its edges better than uniform averaging methods.

*Example of execution:*
```bash
python .\noise_filtering.py ./data/qsd1_w3 --filter gaussian
```

This script first filters the image using the Gaussian filter (selected in this case) and then compares the filtered image to the original to detect whether the image had noise. 

The comparison is made using the SSIM (Structural Similarity Index) metric. 

If the SSIM value is greater than 0.56, the image is considered to be free of noise, and the original image is saved in a new folder named `images_without_noise`. If the SSIM value is less than or equal to 0.56, the image is considered to have noise, and the filtered image is saved instead. 

Additionally, the script prints a list of images that were identified as noisy.

```bash
Images with noise:
00005.jpg
00006.jpg
00016.jpg
00018.jpg
00020.jpg
00023.jpg
00029.jpg
```

*Gaussian Filter Testing:*

The script compares filtered images with their original noise-free versions (`non_augmented` folder) using the SSIM metric to assess the quality of noise filtering.

```bash
python .\filter_test.py ./data/qsd1_w3
```

**Note:** Not all noise-free images have an SSIM score of 1 because some images in the `qsd1_w3` dataset were altered either by noise or by color changes. Images with an SSIM value of 1 are unaltered. In this section, we focus on noise removal, specifically targeting the images listed as noisy above.

```bash
SSIM Results:
00000.jpg -> SSIM: 0.9704
00001.jpg -> SSIM: 1.0000
00002.jpg -> SSIM: 0.9947
00003.jpg -> SSIM: 0.9948
00004.jpg -> SSIM: 1.0000
00005.jpg -> SSIM: 0.8296
00006.jpg -> SSIM: 0.5234
00007.jpg -> SSIM: 0.9910
00008.jpg -> SSIM: 1.0000
00009.jpg -> SSIM: 0.9907
00010.jpg -> SSIM: 1.0000
00011.jpg -> SSIM: 1.0000
00012.jpg -> SSIM: 0.9865
00013.jpg -> SSIM: 0.9925
00014.jpg -> SSIM: 0.9912
00015.jpg -> SSIM: 0.9931
00016.jpg -> SSIM: 0.8201
00017.jpg -> SSIM: 0.9848
00018.jpg -> SSIM: 0.8959
00019.jpg -> SSIM: 0.9867
00020.jpg -> SSIM: 0.7833
00021.jpg -> SSIM: 1.0000
00022.jpg -> SSIM: 1.0000
00023.jpg -> SSIM: 0.7898
00024.jpg -> SSIM: 1.0000
00025.jpg -> SSIM: 0.9892
00026.jpg -> SSIM: 0.8266
00027.jpg -> SSIM: 1.0000
00028.jpg -> SSIM: 0.9967
00029.jpg -> SSIM: 0.8810
```

#### Non-linear Filters

- **Median Filter:**

The median filter is a non-linear digital filtering technique, often used to remove noise from images. It works by replacing each pixel's value with the median value of the intensities in the neighborhood of that pixel. This is particularly effective for reducing salt-and-pepper noise.

*Example of execution:*
```bash
python .\noise_filtering.py ./data/qsd1_w3 --filter median
```

*Median Filter Testing:*
```bash
python .\filter_test.py ./data/qsd1_w3
```

**Note:** Not all noise-free images have an SSIM score of 1 because some images in the `qsd1_w3` dataset were altered either by noise or by color changes. Images with an SSIM value of 1 are unaltered. In this section, we focus on noise removal, specifically targeting the images listed as noisy above.

```bash
SSIM Results:
00000.jpg -> SSIM: 0.9704
00001.jpg -> SSIM: 1.0000
00002.jpg -> SSIM: 0.9947
00003.jpg -> SSIM: 0.9948
00004.jpg -> SSIM: 1.0000
00005.jpg -> SSIM: 0.8380
00006.jpg -> SSIM: 0.4671
00007.jpg -> SSIM: 0.9910
00008.jpg -> SSIM: 1.0000
00009.jpg -> SSIM: 0.9907
00010.jpg -> SSIM: 1.0000
00011.jpg -> SSIM: 1.0000
00012.jpg -> SSIM: 0.9865
00013.jpg -> SSIM: 0.9925
00014.jpg -> SSIM: 0.9912
00015.jpg -> SSIM: 0.9931
00016.jpg -> SSIM: 0.8519
00017.jpg -> SSIM: 0.9848
00018.jpg -> SSIM: 0.9406
00019.jpg -> SSIM: 0.9867
00020.jpg -> SSIM: 0.8145
00021.jpg -> SSIM: 1.0000
00022.jpg -> SSIM: 1.0000
00023.jpg -> SSIM: 0.8893
00024.jpg -> SSIM: 1.0000
00025.jpg -> SSIM: 0.9892
00026.jpg -> SSIM: 0.8266
00027.jpg -> SSIM: 1.0000
00028.jpg -> SSIM: 0.9967
00029.jpg -> SSIM: 0.9102
```

- **Non-Local Means (NLM) Filter:**

The Non-Local Means filter is another effective non-linear filtering technique that reduces noise while preserving details. It works by averaging the pixels based on their similarity to the target pixel, considering pixels from the entire image rather than just the local neighborhood.

*Example of execution:*
```bash
python .\noise_filtering.py ./data/qsd1_w3 --filter nlm
```

*Non-Local Means Filter Testing:*
```bash
python .\filter_test.py ./data/qsd1_w3
```

**Note:** Not all noise-free images have an SSIM score of 1 because some images in the `qsd1_w3` dataset were altered either by noise or by color changes. Images with an SSIM value of 1 are unaltered. In this section, we focus on noise removal, specifically targeting the images listed as noisy above.

```bash
SSIM Results:
00000.jpg -> SSIM: 0.9704
00001.jpg -> SSIM: 1.0000
00002.jpg -> SSIM: 0.9947
00003.jpg -> SSIM: 0.9948
00004.jpg -> SSIM: 1.0000
00005.jpg -> SSIM: 0.7050
00006.jpg -> SSIM: 0.5303
00007.jpg -> SSIM: 0.9910
00008.jpg -> SSIM: 1.0000
00009.jpg -> SSIM: 0.9907
00010.jpg -> SSIM: 1.0000
00011.jpg -> SSIM: 1.0000
00012.jpg -> SSIM: 0.9865
00013.jpg -> SSIM: 0.9925
00014.jpg -> SSIM: 0.9912
00015.jpg -> SSIM: 0.9931
00016.jpg -> SSIM: 0.7707
00017.jpg -> SSIM: 0.9848
00018.jpg -> SSIM: 0.9298
00019.jpg -> SSIM: 0.9867
00020.jpg -> SSIM: 0.7680
00021.jpg -> SSIM: 1.0000
00022.jpg -> SSIM: 1.0000
00023.jpg -> SSIM: 0.7864
00024.jpg -> SSIM: 1.0000
00025.jpg -> SSIM: 0.9892
00026.jpg -> SSIM: 0.8266
00027.jpg -> SSIM: 1.0000
00028.jpg -> SSIM: 0.9967
00029.jpg -> SSIM: 0.8366
```

#### Examples

The table below compares the original noisy images with those filtered using different noise reduction techniques: Gaussian, Median, and Non-Local Means. 

For each method, the SSIM score is calculated to evaluate how well the filter preserves the structural similarity of the image when compared with the ground truth. 

Higher SSIM values indicate better noise reduction while maintaining image details.

The Median filter generally performs best, but certain images, like **00006.jpg**, exhibit unique characteristics where the texture resembles noise, affecting the SSIM score.

| Original Image        | Gaussian Filtered Image    | Median Filtered Image    | Non-Local Means Filtered Image    |
|-----------------------|----------------------------|---------------------------|------------------------------------|
| ![image](https://github.com/user-attachments/assets/b3726eff-57e6-47d0-ad20-7dcb308bb4da) | ![image](https://github.com/user-attachments/assets/5e25824d-ba7a-4e68-800d-9a1d4e2480d0) |  ![image](https://github.com/user-attachments/assets/d328af13-a54d-4554-907b-508266cfbb9b) | ![image](https://github.com/user-attachments/assets/56d1c77c-ac2a-49d8-8dd1-0948e619202b) |
| 00023.jpg -> SSIM: | 0.7898 |  **0.8893** | 0.7864 |
| ![image](https://github.com/user-attachments/assets/4cf13edc-e2e3-4f8f-8367-96a0e0141231) | ![image](https://github.com/user-attachments/assets/12b4503a-a80e-44eb-b70d-afc03f949ca6) | ![image](https://github.com/user-attachments/assets/4cbd9a80-79dc-41f4-8d09-7abf78f4340f) | ![image](https://github.com/user-attachments/assets/7ae5c131-e32c-4c21-9fc3-25e24acedeba) |
| 00018.jpg -> SSIM: | 0.8959 |  **0.9406** | 0.9298 |
| ![image](https://github.com/user-attachments/assets/52b878bc-4f5e-43ab-9b6d-488a905c703b)  | ![image](https://github.com/user-attachments/assets/8f0d10fb-aa24-4ed2-9ac6-eb862dd62f9a) | ![image](https://github.com/user-attachments/assets/3e180932-cec5-495c-bb0e-596849afe944) | ![image](https://github.com/user-attachments/assets/07a52a5e-1f3a-40a0-98f0-f56ce6b2344d) |
| 00006.jpg -> SSIM: | 0.5234 |  0.4671 | **0.5303** |

### Task 2: Implement texture descriptors and evaluate query system (QSD1-W3) using only texture descriptors

#### Evaluate query system (QSD1-W3) with noise removal using the best color descriptors from previous week:
  
```bash
# Block 3D color histograms
(env) PS C:\Users\34663\GitHub\Team4\week2\src> python main.py ./data/qsd1_w3/images_without_noise --color_space=HSV --num_blocks=256 --num_bins=4 --similarity_measure=HISTCMP_CHISQR_ALT --k_value=1
Processing images: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:00<00:00, 63.68image/s]
Processing images: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:00<00:00, 67.64image/s]
mAP@1 for HSV: 0.7333333333333333

# Hierarchical 3D color histograms
(env) PS C:\Users\34663\GitHub\Team4\week2\src> py main.py ./data/qsd1_w3/images_without_noise --color_space=HSV --num_levels=5 --num_bins=4 --similarity_measure=HISTCMP_CHISQR_ALT --k_value=1 --is_pyramid=True
Processing images: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:00<00:00, 47.10image/s]
Processing images: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:00<00:00, 49.12image/s]
mAP@1 for HSV: 0.7333333333333333
```

#### Evaluate query system (QSD1-W3) with noise removal using texture descriptors:

- **LBP (Local Binary Pattern) descriptor:**

```bash
python .\compute_db_descriptors.py --descriptor_type=LBP --num_blocks=256  --num_bins=16
python main.py ./data/qsd1_w3/images_without_noise  --descriptor_type=LBP --num_blocks=256 --num_bins=16 --similarity_measure=Manhattan --k_value=1
# Result
mAP@1 for LBP: 0.9666666666666667
```

- **DCT descriptor with all coefficients:**

```bash
python .\compute_db_descriptors.py --descriptor_type=DCT --num_blocks=64  --num_bins=32
python main.py ./data/qsd1_w3/images_without_noise  --descriptor_type=DCT --num_blocks=64 --num_bins=32 --similarity_measure=HISTCMP_CHISQR_ALT --k_value=1
# Result
mAP@1 for DCT: 0.9
```

- **DCT descriptor with N coefficients (zig-zag scan):**

```bash
python .\compute_db_descriptors.py --descriptor_type=DCT --num_blocks=128  --N=100
python main.py ./data/qsd1_w3/images_without_noise  --descriptor_type=DCT --num_blocks=128 --N=100 --similarity_measure=Lorentzian --k_value=1
# Result
mAP@1 for DCT: 0.9666666666666667
```

- **wavelet-based descriptor:**

```bash
python .\compute_db_descriptors.py --descriptor_type=wavelet --num_levels=1  --wavelet_type=db1
python main.py ./data/qsd1_w3/images_without_noise  --descriptor_type=wavelet --wavelet_type=db1 --num_levels=1 --similarity_measure=Ssim --k_value=1 
# Result
mAP@1 for wavelet: 0.6
```

- **Evaluate query system (QSD1-W3) without noise removal using texture descriptors:**

- **LBP (Local Binary Pattern) descriptor:**

```bash
python .\compute_db_descriptors.py --descriptor_type=LBP --num_blocks=256  --num_bins=16
python main.py ./data/qsd1_w3  --descriptor_type=LBP --num_blocks=256 --num_bins=16 --similarity_measure=Manhattan --k_value=1
# Result
mAP@1 for LBP: 0.7666666666666667
```

- **DCT descriptor with all coefficients:**

```bash
python .\compute_db_descriptors.py --descriptor_type=DCT --num_blocks=64  --num_bins=32
python main.py ./data/qsd1_w3  --descriptor_type=DCT --num_blocks=64 --num_bins=32 --similarity_measure=HISTCMP_CHISQR_ALT --k_value=1
# Result
mAP@1 for DCT: 0.8666666666666667
```

- **DCT descriptor with N coefficients (zig-zag scan):**

```bash
python .\compute_db_descriptors.py --descriptor_type=DCT --num_blocks=128  --N=100
python main.py ./data/qsd1_w3  --descriptor_type=DCT --num_blocks=128 --N=100 --similarity_measure=Lorentzian --k_value=1
# Result
mAP@1 for DCT: 0.9666666666666667
```

- **wavelet-based descriptor:**

```bash
python .\compute_db_descriptors.py --descriptor_type=wavelet --num_levels=1  --wavelet_type=db1
python main.py ./data/qsd1_w3  --descriptor_type=wavelet --wavelet_type=db1 --num_levels=1 --similarity_measure=Ssim --k_value=1 
# Result
mAP@1 for wavelet: 0.6333333333333333
```


### Task 3: : Detect all the paintings (max 2 per image) + Remove background
The background removal algorithm has been improved by relying on a combination of computer vision techniques, including GrabCut for background segmentation, contour detection, polygon approximation, and perspective transformation. 

The algorithm focuses on identifying and isolating objects of interest (e.g., artworks) and transforming them to a canonical view.

#### Algorithm Breakdown

The algorithm consists of several steps:

1. **Background detection using GrabCut algorithm**  
   Using the GrabCut algorithm, we create a mask that distinguishes between the foreground (artwork) and the background. GrabCut uses an iterative optimization process to classify pixels as either foreground or background based on a provided initial rectangle that encloses the object of interest.

   ![Equation](https://latex.codecogs.com/png.latex?E%28x%2C%20z%29%20%3D%20U%28x%2C%20z%29%20%2B%20V%28x%29)

   where: ![Equation](https://latex.codecogs.com/png.latex?U%28x%2C%20z%29) is the data term, representing the likelihood of pixel ![Equation](https://latex.codecogs.com/png.latex?x) being foreground or background given its label ![Equation](https://latex.codecogs.com/png.latex?z), and ![Equation](https://latex.codecogs.com/png.latex?V%28x%29) is the smoothness term, encouraging neighboring pixels to have similar labels.

2. **Morphology + Fill surrounded pixels**  
   After GrabCut, morphological operations (opening and closing) are applied to clean up noise and fill holes in the foreground mask. Additionally, the `fill_surrounded_pixels` function ensures that any pixels surrounded by foreground in all directions are included in the mask.

3. **Artwork detection via contour extraction**  
   Contours, which represent the boundaries of objects in the mask, are extracted. The two largest contours are assumed to represent the primary objects (artworks) in the image. If only one object is present, only one contour is returned.

4. **Contour approximation and corner detection**  
   The contours are approximated into polygons using Douglas-Peucker's algorithm, which simplifies the contours into fewer points.

   The goal is to approximate each contour to a quadrilateral (four corners), representing the artwork's bounding box. The simplification is controlled by the parameter ![Equation](https://latex.codecogs.com/png.latex?%5Cepsilon), which is a fraction of the contour's perimeter:

   ![Equation](https://latex.codecogs.com/png.latex?%5Cepsilon%20%3D%20%5Cbeta%20%5Ctimes%20l)

   where ![Equation](https://latex.codecogs.com/png.latex?l) is the perimeter of the contour, and ![Equation](https://latex.codecogs.com/png.latex?%5Cbeta) is a small constant (e.g., 0.02). The algorithm keeps increasing ![Equation](https://latex.codecogs.com/png.latex?%5Cbeta) until the contour has exactly four points.

5. **Perspective Transformation**  
   Once the corner points of the artwork are identified, a perspective transformation is applied to align the object into a rectangular view, correcting any perspective distortions.
   The transformation matrix ![Equation](https://latex.codecogs.com/png.latex?M) is calculated using the four corner points from the artwork (source points) and mapping them to the desired output rectangle (destination points). The transformation is applied using the equation:

   ![Equation](https://latex.codecogs.com/png.latex?%5Cmathbf%7Bp%27%7D%20%3D%20M%20%5Ccdot%20%5Cmathbf%7Bp%7D)

   where ![Equation](https://latex.codecogs.com/png.latex?%5Cmathbf%7Bp%7D) is the original point in the source image, ![Equation](https://latex.codecogs.com/png.latex?M) is the transformation matrix, and ![Equation](https://latex.codecogs.com/png.latex?%5Cmathbf%7Bp%27%7D) is the point in the transformed image.

#### Usage:
```bash
# Without score computation
python background_removal.py data/qsd2_w3
# With score computation
python background_removal.py data/qsd2_w3 --score=True

## If using python3 and the script is not executable, use:
python3 background_removal.py data/qsd2_w3
python3 background_removal.py data/qsd2_w3 --score=True
```

<table>
  <tr>
    <td style="text-align: center;">
      <img src="https://github.com/user-attachments/assets/6666b87c-ab11-4e25-b5b2-5a8c2d1d8400" alt="Imagen 1" width="400"/>
      <p>Example Image</p>
    </td>
    <td style="text-align: center;">
      <img src="https://github.com/user-attachments/assets/e451e8ce-844c-48ba-a4ae-74ec295aad14" alt="Imagen 2" width="400"/>
      <p>Predicted Mask</p>
    </td>
    <td style="text-align: center;">
      <img src="https://github.com/user-attachments/assets/ef1e8773-70cf-488d-8968-19f063a6f4a6" alt="Imagen 3" width="170"/>
      <p>Cropped Image 1</p>
    </td>
    <td style="text-align: center;">
      <img src="https://github.com/user-attachments/assets/7d8a1708-729f-46c4-8ec1-c4d167628d68" alt="Imagen 3" width="170"/>
      <p>Cropped Image 2</p>
    </td>
  </tr>
</table>

#### Our results

| Metric        | Value |
|---------------|-------|
| Global F1 Score | 0.99  |
| Global Precision | 0.99  |
| Global Recall    | 0.99  |

### Task 5: Remove background + Retrieval system (QSD2-W3)

#### Without Noise Removal + With Background Removal

- **LBP (Local Binary Pattern) descriptor:**

```bash
# Background Removal Offline
python background_removal.py data/qsd2_w3
# Retrieval system (QSD2-W3)
python .\compute_db_descriptors.py --descriptor_type=LBP --num_blocks=256  --num_bins=16
python main.py ./data/qsd2_w3/masked  --descriptor_type=LBP --num_blocks=256 --num_bins=16 --similarity_measure=Manhattan --k_value=1
# Result
mAP@1 for LBP: 0.5333333333333333
```

- **DCT descriptor with all coefficients:**

```bash
# Background Removal Offline
python background_removal.py data/qsd2_w3
# Retrieval system (QSD2-W3)
python .\compute_db_descriptors.py --descriptor_type=DCT --num_blocks=64  --num_bins=32
python main.py ./data/qsd2_w3/masked  --descriptor_type=DCT --num_blocks=64 --num_bins=32 --similarity_measure=HISTCMP_CHISQR_ALT --k_value=1
# Result
mAP@1 for DCT: 0.6333333333333333
```

- **DCT descriptor with N coefficients (zig-zag scan):**

```bash
# Background Removal Offline
python background_removal.py data/qsd2_w3
# Retrieval system (QSD2-W3)
python .\compute_db_descriptors.py --descriptor_type=DCT --num_blocks=128  --N=100
python main.py ./data/qsd2_w3/masked  --descriptor_type=DCT --num_blocks=128 --N=100 --similarity_measure=Lorentzian --k_value=1
# Result
mAP@1 for DCT: 0.75
```

- **wavelet-based descriptor:**

```bash
# Background Removal Offline
python background_removal.py data/qsd2_w3
# Retrieval system (QSD2-W3)
python .\compute_db_descriptors.py --descriptor_type=wavelet --num_levels=1  --wavelet_type=db3
python main.py ./data/qsd2_w3/masked  --descriptor_type=wavelet --wavelet_type=db3 --num_levels=1 --similarity_measure=Ssim --k_value=1 
# Result
mAP@1 for wavelet: 0.6
```

#### With Noise Removal + With Background Removal

- **LBP (Local Binary Pattern) descriptor:**

```bash
# Median filter for noise removal Offline
python .\noise_filtering.py ./data/qsd2_w3 --filter median
# Background Removal Offline
python background_removal.py data/qsd2_w3/images_without_noise
# Retrieval system (QSD2-W3)
python .\compute_db_descriptors.py --descriptor_type=LBP --num_blocks=256  --num_bins=16
python main.py ./data/qsd2_w3/images_without_noise/masked  --descriptor_type=LBP --num_blocks=256 --num_bins=16 --similarity_measure=Manhattan --k_value=1
# Result
mAP@1 for LBP: 0.6333333333333333
```

- **DCT descriptor with all coefficients:**

```bash
# Median filter for noise removal Offline
python .\noise_filtering.py ./data/qsd2_w3 --filter median
# Background Removal Offline
python background_removal.py data/qsd2_w3/images_without_noise
# Retrieval system (QSD2-W3)
python .\compute_db_descriptors.py --descriptor_type=DCT --num_blocks=64  --num_bins=32
python main.py ./data/qsd2_w3/images_without_noise/masked  --descriptor_type=DCT --num_blocks=64 --num_bins=32 --similarity_measure=HISTCMP_CHISQR_ALT --k_value=1
# Result
mAP@1 for DCT: 0.65
```

- **DCT descriptor with N coefficients (zig-zag scan):**

```bash
# Median filter for noise removal Offline
python .\noise_filtering.py ./data/qsd2_w3 --filter median
# Background Removal Offline
python background_removal.py data/qsd2_w3/images_without_noise
# Retrieval system (QSD2-W3)
python .\compute_db_descriptors.py --descriptor_type=DCT --num_blocks=128  --N=100
python main.py ./data/qsd2_w3/images_without_noise/masked  --descriptor_type=DCT --num_blocks=128 --N=100 --similarity_measure=Lorentzian --k_value=1
# Result
mAP@1 for DCT: 0.7666666666666667
```

- **wavelet-based descriptor:**

```bash
# Median filter for noise removal Offline
python .\noise_filtering.py ./data/qsd2_w3 --filter median
# Background Removal Offline
python background_removal.py data/qsd2_w3/images_without_noise
# Retrieval system (QSD2-W3)
python .\compute_db_descriptors.py --descriptor_type=wavelet --num_levels=1  --wavelet_type=db3
python main.py ./data/qsd2_w3/images_without_noise/masked  --descriptor_type=wavelet --wavelet_type=db3 --num_levels=1 --similarity_measure=Ssim --k_value=1 
# Result
mAP@1 for wavelet: 0.5333333333333333
```

#### With Background Removal + With Noise Removal

- **LBP (Local Binary Pattern) descriptor:**

```bash
# Background Removal Offline
python background_removal.py data/qsd2_w3
# Median filter for noise removal Offline
python .\noise_filtering.py ./data/qsd2_w3/masked --filter median
# Retrieval system (QSD2-W3)
python .\compute_db_descriptors.py --descriptor_type=LBP --num_blocks=256  --num_bins=16
python main.py ./data/qsd2_w3/masked/images_without_noise  --descriptor_type=LBP --num_blocks=256 --num_bins=16 --similarity_measure=Manhattan --k_value=1
# Result
mAP@1 for LBP: 0.5666666666666667
```

- **DCT descriptor with all coefficients:**

```bash
# Background Removal Offline
python background_removal.py data/qsd2_w3
# Median filter for noise removal Offline
python .\noise_filtering.py ./data/qsd2_w3/masked --filter median
# Retrieval system (QSD2-W3)
python .\compute_db_descriptors.py --descriptor_type=DCT --num_blocks=64  --num_bins=32
python main.py ./data/qsd2_w3/masked/images_without_noise  --descriptor_type=DCT --num_blocks=64 --num_bins=32 --similarity_measure=HISTCMP_CHISQR_ALT --k_value=1
# Result
mAP@1 for DCT: 0.6166666666666667
```

- **DCT descriptor with N coefficients (zig-zag scan):**

```bash
# Background Removal Offline
python background_removal.py data/qsd2_w3
# Median filter for noise removal Offline
python .\noise_filtering.py ./data/qsd2_w3/masked --filter median
# Retrieval system (QSD2-W3)
python .\compute_db_descriptors.py --descriptor_type=DCT --num_blocks=128  --N=100
python main.py ./data/qsd2_w3/masked/images_without_noise  --descriptor_type=DCT --num_blocks=128 --N=100 --similarity_measure=Lorentzian --k_value=1
# Result
mAP@1 for DCT: 0.7666666666666667
```

- **wavelet-based descriptor:**

```bash
# Background Removal Offline
python background_removal.py data/qsd2_w3
# Median filter for noise removal Offline
python .\noise_filtering.py ./data/qsd2_w3/masked --filter median
# Retrieval system (QSD2-W3)
python .\compute_db_descriptors.py --descriptor_type=wavelet --num_levels=1  --wavelet_type=db3
python main.py ./data/qsd2_w3/masked/images_without_noise  --descriptor_type=wavelet --wavelet_type=db3 --num_levels=1 --similarity_measure=Ssim --k_value=1 
# Result
mAP@1 for wavelet: 0.6
```

<h2 align="center">WEEK 4: Tasks</h2>

### Project Structure

    week4/
    ├── evaluation/
    │   ├── bbox_iou.py
    │   └── evaluation_funcs.py
    ├── src/
    │   ├── average_precision.py
    │   ├── compute_db_descriptors.py
    │   ├── compute_descriptors.py
    │   ├── compute_img_descriptors.py
    │   ├── compute_similarities.py
    │   ├── main.py
    │   ├── metrics.py
    │   └── utils.py
    ├── utils/
    │   ├── plot_results.py
    │   └── print_dict.py
    
### Task 1: Keypoint Detection and Descriptor Computation

The script `keypoint_detection.py` provides functions to detect keypoints and generate their descriptors using three methods: SIFT, ORB, and DAISY. Each method offers different advantages in terms of computation cost, robustness, and invariance to scale, rotation, and illumination changes.

- **SIFT (Scale-Invariant Feature Transform):**

SIFT is a method for detecting and describing keypoints that is invariant to scale, rotation, and illumination, making it ideal for robust feature matching across images. It identifies keypoints by detecting extrema in scale-space, refining their location, assigning orientation, and generating a distinctive descriptor for each.

```bash
def sift(img):
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # Initiate SIFT detector
    sift = cv.SIFT_create(nfeatures=400)
    # find the keypoints and compute the descriptors with SIFT
    kp, des = sift.detectAndCompute(gray,None)
    return kp, des
```

- **ORB (Oriented FAST and Rotated BRIEF):**

ORB is a fast and efficient alternative to SIFT, combining the FAST keypoint detector and a modified BRIEF descriptor. It is designed to be rotation-invariant and has a low computation cost, which makes it suitable for real-time applications, especially on devices with limited resources. Unlike SIFT and SURF, ORB is patent-free.

```bash
def orb(img):
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # Initiate ORB detector
    orb = cv.ORB_create()
    # find the keypoints with ORB
    kp = orb.detect(gray,None)
    # compute the descriptors with ORB
    kp, des = orb.compute(gray, kp)
    return kp, des
```

- **DAISY:**

DAISY is a descriptor that leverages gradient orientation histograms and is structured for fast, dense feature extraction. It’s often used in applications where dense representations of local features are needed, such as bag-of-features models in image recognition tasks. DAISY is robust to minor geometric and photometric transformations, making it a versatile choice.

```bash
def daisy_descriptor(img):
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # Compute the DAISY descriptors
    descs, descs_img = daisy(gray, step=180, radius=58, rings=2, histograms=6, orientations=8, visualize=True)
    # Reshape descriptors to a 2D array and convert to float32 for compatibility
    return descs.reshape(-1, descs.shape[2]).astype(np.float32)
```

### Task 2: Find tentative matches based on similarity of local appearance and verify matches



## Team Members

This project was developed by the following team members:

- **[Judit Salavedra](https://github.com/juditsalavedra)**
- **[Judith Caldés](https://github.com/judithcaldes)**
- **[Carme Corbi](https://github.com/carmecorbi)**
- **[Yeray Cordero](https://github.com/yeray142)**

## License
The MIT License (MIT). Please see [LICENSE File](LICENSE) for more information.


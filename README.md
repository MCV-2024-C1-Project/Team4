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


## Table of contents

- [Table of contents](#table-of-contents)
- [Introduction](#introduction)
- [Installation](#installation)
    - [Prerequisites](#prerequisites)
- [What's included](#whats-included)
- [License](#license)


## Introduction

To be done

## Installation

This section will guide you through the installation process of the project and its testing.

### Prerequisites
The following prerequisites must be followed:
- Python >= v3.8

## Tasks

### Task 1:  Museum and query image descriptors (BBDD & QSD1)

- **Index the Database (BBDD):** Generate descriptors offline.
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

### Task 2:  Selection and implementation of similarity measures to compare images

The measures used are implemented using the OpenCV library with the function cv::compareHist, that compares two dense or two sparse histograms using the specified method.

- **Color space CieLab:** The optimal similarity measure is the Hellinger/Bhattacharyya distance:
  
  <img src="https://github.com/user-attachments/assets/bc26e2ec-7512-4c4a-ba61-73d87d73fc17" alt="image" width="300"/>

  `measure = cv.HISTCMP_HELLINGER`

- **Color space HSV:** The optimal similarity measure is the Alternative Chi-Square distance:

  <img src="https://github.com/user-attachments/assets/44885a21-38c6-4eff-86b9-f216f6ed36fb" alt="image" width="300"/>

  `measure = cv.HISTCMP_CHISQR_ALT`

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
- `k_value`: The number of top results to retrieve. Example: `1`, `5`.
- `query_path`: The path to the query dataset.
- `is_test`: A flag to indicate if the model is in testing mode. Options: `True` (without ground truth) or `False` (with ground truth).

#### Process Description:

This function first computes the image descriptors for all images in QST1 and saves them in a `.pkl` file, based on the specified color space argument.

Next, it reads the `.pkl` files containing the computed image descriptors for both the query dataset (QST1) and the museum dataset (BBDD, computed offline).

For each image in the query set (QST1), it calculates the similarity measure against all museum images (BBDD).

Finally, the evaluation of the system is conducted using **mAP@K (mean Average Precision at K)**, which involves calculating the Average Precision for each value of `k` from `1` to `K` (AP@K) and then taking the mean across all queries (mAP@K).

This metric indicates how effectively the system returns the results for each case.


#### Best Result Methods

The best results are obtained using the following methods:

- **Method 1: Lab - Hellinger Distance**

  Terminal commands for `k=1` and `k=5`:

  ```bash
  python .\main.py Lab HISTCMP_HELLINGER 1 data\qsd1_w1 False
  python .\main.py Lab HISTCMP_HELLINGER 5 data\qsd1_w1 False
  ```

- **Method 2: HSV - Alternative Chi Square**

  Terminal commands for `k=1` and `k=5`:
  
  ```bash
  python .\main.py HSV HISTCMP_CHISQR_ALT 1 data\qsd1_w1 False
  python .\main.py HSV HISTCMP_CHISQR_ALT 5 data\qsd1_w1 False
  ```

#### Results

| Method                          | mAP@1          | mAP@5           |
|---------------------------------|----------------|------------------|
| **Lab - Hellinger Distance**    | 0.43333333333333335 | 0.4972222222222223 |
| **HSV - Alternative Chi Square** | 0.4666666666666667 | 0.5372222222222223 |


     
## What's included

Within the download you'll find the following directories and files, logically grouping common assets. You'll see something like this:

    ```text
    MCV-2024-C1-Project/
    ├── evaluation/
    │   ├── bbox_iou.py
    │   └── evaluation_funcs.py
    ├── src/
    │   ├── average_precision.py
    │   └── ...
    ├── utils/
    │   ├── plot_results.py
    │   └── print_dict.py
    ```

## License
The MIT License (MIT). Please see [LICENSE File](LICENSE) for more information.


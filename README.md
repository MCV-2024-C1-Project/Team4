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
   
2. **Create a virtual environment:**
   ```bash
   python -m venv env
   ```

3. **Activate the virtual environment:**
    - On Windows:
      ```bash
      .\env\Scripts\activate
      ```
    - On MacOS/Linux:
      ```bash
      source env/bin/activate
      ```

4. **Install the dependencies:**
  ```bash
  pip install -r requirements.txt
  ```

   
## Project Structure

Within the downloaded repository, you'll find the following directories and files, logically grouping common assets. The data folders need to be downloaded and decompressed from the provided links:

- **BBDD:** [Download here](https://e-aules.uab.cat/2024-25/pluginfile.php/398404/mod_page/content/186/BBDD.zip)
- **qsd1_w1:** [Download here](https://e-aules.uab.cat/2024-25/pluginfile.php/398404/mod_page/content/186/qsd1_w1.zip?time=1602013828018)
- **qst1_w1:** [Download here](https://e-aules.uab.cat/2024-25/mod/resource/view.php?id=176342)

Once downloaded and extracted, the project structure will look like this:

    Team4/
    ├── data/
    │   ├── BBDD/
    │   ├── qsd1_w1/
    │   └── qst1_w1/
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

<h2 align="center">WEEK 1: Tasks</h2>

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

### Task 1: Implement 3D/2D block and hierarchical histograms
Explicar com s'implementen els histogrames i un exemple d'algun plot.

  
  

### Task 2: Test query system using query set QSD1-W2 development and evaluate retrieval results
- **Index the Database (BBDD):** Generate descriptors offline and saves them in a `.pkl` file.
  ```bash
  # Block 3D histograms
  python compute_db_descriptors.py HSV 32 16 False
  python compute_db_descriptors.py Lab 32 2 False

  # Hierarchical 3D histograms
  python compute_db_descriptors.py HSV 32 16 True
  python compute_db_descriptors.py Lab 32 2 True
  ```
```bash
# Block 3D histograms
python main.py Lab 32 2 False Lorentzian 1 data\qsd1_w1 False
# Hierarchical 3D histograms
python main.py Lab 4 2 True Lorentzian 1 data\qsd1_w1 False
```

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

#### Examples:
PUT EXAMPLES HERE

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

#### Example Output

```plaintext
Global F1 Score: 0.96
Global Precision: 0.94
Global Recall: 0.98
```

This output reflects the algorithm's overall accuracy in foreground-background separation across the entire dataset. To evaluate the algorithm use the commands from Task 3 with the `--score=True` flag.

### Task 5: Background removal evaluation (QSD2-W2)  

### Task 6:Processing the QST1 and QST2 Testing Dataset

#### Our results

| Metric        | Value |
|---------------|-------|
| Global F1 Score | 0.97  |
| Global Precision | 0.95  |
| Global Recall    | 0.98  |

## Team Members

This project was developed by the following team members:

- **[Judit Salavedra](https://github.com/juditsalavedra)**
- **[Judith Caldés](https://github.com/judithcaldes)**
- **[Carme Corbi](https://github.com/carmecorbi)**
- **[Yeray Cordero](https://github.com/yeray142)**

## License
The MIT License (MIT). Please see [LICENSE File](LICENSE) for more information.


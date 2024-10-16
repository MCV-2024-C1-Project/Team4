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

## Team Members

This project was developed by the following team members:

- **[Judit Salavedra](https://github.com/juditsalavedra)**
- **[Judith Caldés](https://github.com/judithcaldes)**
- **[Carme Corbi](https://github.com/carmecorbi)**
- **[Yeray Cordero](https://github.com/yeray142)**

## License
The MIT License (MIT). Please see [LICENSE File](LICENSE) for more information.


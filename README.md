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


## Team Members

This project was developed by the following team members:

- **[Judit Salavedra](https://github.com/juditsalavedra)**
- **[Judith Caldés](https://github.com/judithcaldes)**
- **[Carme Corbi](https://github.com/carmecorbi)**
- **[Yeray Cordero](https://github.com/yeray142)**

## License
The MIT License (MIT). Please see [LICENSE File](LICENSE) for more information.


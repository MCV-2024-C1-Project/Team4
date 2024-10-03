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
To be done

### Steps
To be done

#### Task 1:  Museum and query image descriptors (BBDD & QSD1)

- **Index the Database (BBDD):** Generate descriptors offline.
  ```bash
  python .\compute_db_descriptors.py

- **Compute image descriptors (QSD1):**
  
  Methods:
   - **Color space CieLab:** Histograms for the L, a, and b channels.
     
     Example command:
     ```bash
     python .\compute_img_descriptors.py 00001.jpg hist_lab
     ```
     ![Figure_1](https://github.com/user-attachments/assets/b661403c-8a9b-4afb-bae6-4519f5e15ec5)

   - **Color space HSV:** Histograms for the Hue, Saturation, and Value channels.
     
     Example command:
     ```bash
     python .\compute_img_descriptors.py 00001.jpg hist_hsv
     ```
     ![Figure_2](https://github.com/user-attachments/assets/e6f45adb-9c68-478c-bc40-266303ba2558)

#### Task 2:  Selection and implementation of similarity measures to compare images

The measures used are implemented using the OpenCV library with the function cv::compareHist, that compares two dense or two sparse histograms using the specified method.

- **Color space CieLab:** The optimal similarity measure is the Hellinger/Bhattacharyya distance:
  
  <img src="https://github.com/user-attachments/assets/bc26e2ec-7512-4c4a-ba61-73d87d73fc17" alt="image" width="300"/>

  `method = cv.HISTCMP_HELLINGER`

- **Color space HSV:** The optimal similarity measure is the Alternative Chi-Square distance:

  <img src="https://github.com/user-attachments/assets/44885a21-38c6-4eff-86b9-f216f6ed36fb" alt="image" width="300"/>

  `method = cv.HISTCMP_CHISQR_ALT`
  
     
## What's included

Within the download you'll find the following directories and files, logically grouping common assets. You'll see something like this:

    ```text
    eymo-cloud-rs/
    ├── evaluation/
    │   ├── bbox_iou.py
    │   └── evaluation_funcs.py
    ├── evaluation/
    │   ├── plot_results.py
    │   └── print_dict.py
    ```

## License
The MIT License (MIT). Please see [LICENSE File](LICENSE) for more information.


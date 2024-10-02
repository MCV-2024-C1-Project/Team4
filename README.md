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

- **Index the Database:** Generate descriptors offline.
  ```bash
  python .\compute_db_descriptors.py

- **Compute image descriptors (QSD1):**
  
  Methods:
   - **Color space CieLab:** Histograms for the L, a, and b channels.
     
     Example command:
     ```bash
     python .\compute_img_descriptors.py 00001.jpg hist_lab
     
   - **Color space HSV histogram:** Histograms for the Hue, Saturation, and Value channels.
     
     Example command:
     ```bash
     python .\compute_img_descriptors.py 00001.jpg hist_hsv
     
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


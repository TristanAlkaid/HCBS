
# Hierarchical Chat-Based Strategies with MLLMs for Spatio-Temporal Action Detection
[![Paper](https://img.shields.io/badge/IPM-Paper-<COLOR>.svg)](https://www.sciencedirect.com/journal/information-processing-and-management) 
[![Dataset](https://img.shields.io/badge/Dataset-Football_Description-red)](Data/)

This project is the official implementation of the paper _"Hierarchical Chat-Based Strategies with MLLMs For Spatio-Temporal Action Detection"_.

## 🏈 Key Features
- **Hierarchical Chat-Based Strategy (HCBS)**
  
  A progressive dialogue protocol guiding MLLMs to generate structured action descriptions:
  
    • Entity Localization: Identify potential actors

    • Trajectory Prediction: Analyze motion patterns

    • Micro-action Parsing: Capture subtle motion details

- **Football Description Dataset**
  
  Contains 712 football match clip descriptions  focusing on:
  
    • 83% small-target motions

    • 65% multi-participator overlapping scenarios



## 🛠️ Setup
### Code Integration
1. Clone base framework:
```bash
git clone https://github.com/MCG-NJU/MOC-Detector.git
```

2. Overwrite core modules:
```bash
cp -r Project/src/ MOC-Detector/src/
```

## 🚀 Quick Start
Please the instruction of https://github.com/MCG-NJU/MOC-Detector.


## 📊 Benchmark Results
Performance on Multisports dataset:
|         Method          | FrameAP@0.5 (%) | VideoAP@0.5 (%) |
| :---------------------: | :-------------: | :-------------: |
|       ROAD (ICCV)       |      3.90       |      0.00       |
|     SlowFast (ICCV)     |      0.00       |      0.00       |
|   MOC-Detector (ECCV)   |      6.40       |      0.00       |
| MOC+ConvFormer (MICCAI) |      5.10       |      0.00       |
| MOC+DilateFormer (TMM)  |      5.97       |      0.04       |
|  Ours with VideoLLaMA2  |      7.21       |      0.00       |
|  Ours with LLaVA-NeXT   |      7.46       |      0.06       |
|    Ours with LongVA     |      7.30       |      0.02       |
|   **Ours with LLaVA**   |    **8.23**     |    **0.11**    |


Performance on J-HMDB dataset:
| Method                          | FrameAP@0.5 (%) | VideoAP@0.5 (%) |
|:-------------------------------:|-----------------|-----------------|
| ROAD (ICCV)                     | 71.10           | 72.00           |
| MOC-Detector (ECCV)             | 81.06           | 77.20           |
| Tad-TR (TIP)                    | 68.70           | 78.90           |
| HIT (WACV)                      | 88.10           | 83.80           |
| MOC+ConvFormer (MICCAI)         | 79.65           | 67.01           |
| MOC+DilateFormer (TMM)          | 10.25           | 5.85            |
| Ours with VideoLLaMA2           | 98.91           | 100.00          |
| **Ours with LLaVA-NeXT**       | **99.32**       | **100.00**      |
| Ours with LongVA                | 98.82           | 100.00          |
| Ours with LLaVA                 | 98.44           | 100.00          |

Performance on UCF101-24 dataset:
| Method                          | FrameAP@0.5 (%) | VideoAP@0.5 (%) |
|---------------------------------|-----------------|-----------------|
| ROAD (ICCV)                     | 43.30           | 46.30           |
| MOC-Detector (ECCV)             | 98.49           | 53.80           |
| YOWOv2 (ARXIV)                  | 87.00           | 52.80           |
| HIT (WACV)                      | 84.80           | 74.30           |
| MOC+ConvFormer (MICCAI)         | 97.00           | 88.49           |
| MOC+DilateFormer (TMM)          | 92.16           | 86.85           |
| Ours with VideoLLaMA2           | 98.39           | 91.69           |
| Ours with LLaVA-NeXT            | 97.94           | 89.36           |
| Ours with LongVA                | 97.82           | 90.80           |
| **Ours with LLaVA**             | **98.88**       | **92.20**      |

Performance using Prompts Pooling:
| MLLMs        | FrameAP@0.2 (%) | FrameAP@0.5 (%) | FrameAP@0.75 (%) | VideoAP@0.2 (%) | VideoAP@0.5 (%) | VideoAP@0.75 (%) |
|-------------|-------|-------|-------|--------|--------|-------|
| LLaVA-NeXT  | 99.99 | 99.58 | 73.38 | 100.00 | 100.00 | 86.79 |
| LLaVA       | 99.56 | 98.02 | 61.69 | 100.00 | 100.00 | 79.79 |
| VideoLLaMA2 | 99.58 | 98.98 | 60.42 | 100.00 | 100.00 | 75.49 |
| LongVA      | 99.72 | 97.66 | 64.52 | 100.00 | 100.00 | 67.01 |

## 📜 Citation
If you use this work, please cite:
```bibtex
@article{HCBS,
  title={Hierarchical Chat-Based Strategies with MLLMs For Spatio-Temporal Action Detection},
  author={Xuyang Zhou, Ye Wang, Fei Tao, Hong Yu, Qun Liu},
  journal={Information Processing & Management},
  year={2025}
}
```

## License
This project is released under the MIT License. See [LICENSE](LICENSE) for details.

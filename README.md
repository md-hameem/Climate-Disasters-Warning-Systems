# Climate Disaster Warning System

This repository contains code and models for detecting and analyzing climate disasters, including fire, flood, and sea-level rise events. The project leverages deep learning and data analysis techniques to process satellite and ground data for early warning and research purposes.

## Project Structure

```
model/
  Fire/
    fire_detection_resnet50_V1.h5
    Fire_Detection.ipynb
    D-Fire/
    .ipynb_checkpoints/
  Flood/
    flood_detection.ipynb
    optimizer_vit.pth
    resnet_confusion_matrix.csv
    resnet_hard_predictions.csv
    resnet_metrics.pkl
    resnet_model_checkpoint.pth
    resnet_probability_predictions.csv
    resnet_test_metrics_summary.csv
    resnet_test_summary_metrics.csv
    vit_model.pth
    .ipynb_checkpoints/
  Sea-Level Rise/
    CSR_GRACE_GRACE-FO_RL06_Mascons_all-corrections_v02.nc
    SLR_GRACE.ipynb
    Data/
    .ipynb_checkpoints/
.gitignore
```

## Components

### Fire Detection

- [`model/Fire/Fire_Detection.ipynb`](model/Fire/Fire_Detection.ipynb): Jupyter notebook for training and evaluating a fire detection model using ResNet50 and Keras. Includes data preprocessing, augmentation, model training, evaluation, and saving.
- `fire_detection_resnet50_V1.h5`: Trained Keras model for fire detection.
- `D-Fire/`: Contains training and test images and labels for fire detection.

### Flood Detection

- [`model/Flood/flood_detection.ipynb`](model/Flood/flood_detection.ipynb): Jupyter notebook for flood detection using deep learning models (ResNet, ViT).
- Model checkpoints and metrics: Includes files such as `optimizer_vit.pth`, `resnet_model_checkpoint.pth`, and various CSV/PKL files for evaluation and predictions.

### Sea-Level Rise Analysis

- [`model/Sea-Level Rise/SLR_GRACE.ipynb`](model/Sea-Level Rise/SLR_GRACE.ipynb): Jupyter notebook for analyzing sea-level rise using GRACE satellite data (NetCDF format).
- `CSR_GRACE_GRACE-FO_RL06_Mascons_all-corrections_v02.nc`: NetCDF data file with satellite measurements.
- `Data/`: Additional data resources for sea-level rise analysis.

## Getting Started

1. **Clone the repository**
   ```sh
   git clone <your-repo-url>
   cd <repo-folder>
   ```

2. **Install dependencies**
   - Python 3.x
   - Jupyter Notebook
   - TensorFlow, Keras, OpenCV, NumPy, Matplotlib, netCDF4, etc.

   You can install dependencies using:
   ```sh
   pip install -r requirements.txt
   ```

3. **Run Notebooks**
   - Open the desired notebook in Jupyter and follow the instructions in each section.

## Data

- Fire and flood datasets are organized in subfolders with images and label files.
- Sea-level rise data is provided in NetCDF format.

## Notes

- Model files (`.h5`, `.pth`) are large and may not be included in the repository due to `.gitignore` rules.
- Check `.gitignore` for excluded files and folders.

## License

Specify your license here.

---

**Contact:**  
For questions or contributions, please open an issue or submit a pull request.
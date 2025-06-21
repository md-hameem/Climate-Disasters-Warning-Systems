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
LICENSE
README.md
requirements.txt
```

## Models Used

This project utilizes state-of-the-art deep learning models for climate disaster detection and analysis:

- **Fire Detection:**  
  Uses a ResNet50-based convolutional neural network implemented in Keras. The model is trained to classify images as fire or non-fire, leveraging transfer learning for improved accuracy and faster convergence.  
  - *Why ResNet50?* ResNet50 is a powerful image classification model known for its deep architecture and skip connections, which help prevent vanishing gradients and improve performance on complex image tasks.

- **Flood Detection:**  
  Employs both ResNet and Vision Transformer (ViT) architectures for image-based flood detection. The ResNet model provides robust baseline performance, while the ViT explores transformer-based approaches for potentially better generalization on diverse flood imagery.  
  - *Why ResNet and ViT?* ResNet is a proven standard for image tasks, while ViT leverages self-attention mechanisms, which can capture global context in images, potentially improving detection in challenging scenarios.

- **Sea-Level Rise Analysis:**  
  Analyzes satellite data (GRACE) in NetCDF format using data analysis and visualization techniques in Jupyter notebooks.  
  - *Why GRACE data?* GRACE satellite data provides precise measurements of Earth's gravity field, which can be used to infer changes in sea level and mass distribution, making it ideal for scientific analysis of sea-level rise.

## Download Datasets and Trained Models

- **Fire and Flood Datasets:**  
  Please download the datasets from their respective sources or request access if required. Example sources include [Kaggle Fire Dataset](https://www.kaggle.com/datasets/phylake1337/fire-dataset) and [Kaggle Flood Dataset](https://www.kaggle.com/datasets/ratthachat/flood-image-dataset).  
  Place the datasets in the corresponding subfolders under `model/Fire/D-Fire/` and `model/Flood/`.

- **Sea-Level Rise Data:**  
  GRACE satellite data can be downloaded from the [NASA GRACE Data Portal](https://podaac.jpl.nasa.gov/GRACE).

- **Trained Models:**  
  Pretrained model files (e.g., `.h5`, `.pth`) may be too large for direct inclusion in this repository. You can download [TRAINED MODEL](https://drive.google.com/drive/folders/1uGfHQNVUJ4oqNp0-tL8cfA4jEjHz_Lxc?usp=sharing) from the following links or request access:
  - Fire Detection ResNet50 & Flood Detection ResNet and ViT & SLR Detection; Place the downloaded models in their respective folders as shown in the project structure.

## Components

### Fire Detection

- [`model/Fire/Fire_Detection.ipynb`](model/Fire/Fire_Detection.ipynb): Jupyter notebook for training and evaluating a fire detection model using ResNet50 and Keras. Includes data preprocessing, augmentation, model training, evaluation, and saving.
- `fire_detection_resnet50_V1.h5`: Trained Keras model for fire detection.
- `D-Fire/`: Contains training and test images and labels for fire detection.

### Flood Detection

- [`model/Flood/flood_detection.ipynb`](model/Flood/flood_detection.ipynb): Jupyter notebook for flood detection using deep learning models (ResNet, ViT). Includes model training, evaluation, and comparison.
- Model checkpoints and metrics: Includes files such as `optimizer_vit.pth`, `resnet_model_checkpoint.pth`, and various CSV/PKL files for evaluation and predictions.

### Sea-Level Rise Analysis

- [`model/Sea-Level Rise/SLR_GRACE.ipynb`](model/Sea-Level Rise/SLR_GRACE.ipynb): Jupyter notebook for analyzing sea-level rise using GRACE satellite data (NetCDF format).
- `CSR_GRACE_GRACE-FO_RL06_Mascons_all-corrections_v02.nc`: NetCDF data file with satellite measurements.
- `Data/`: Additional data resources for sea-level rise analysis.

## Getting Started

1. **Clone the repository**
   ```sh
   git clone https://github.com/md-hameem/Climate-Disasters-Warning-Systems.git
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

MIT License. See [LICENSE](LICENSE) for details.

---

**Contact:**  
For questions or contributions, please open an issue or submit a pull or Mail me at: hamimmd555@gmail.com
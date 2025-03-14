# README

## Environment
---
- **Platform**: Google Colab  
- **Hardware**: T4 GPU & A100 GPU  
- **Python Version**: 3.10.1  

---

## How to Inference
---
Since the environment is set up in Google Colab, the inference steps are straightforward:

1. Create an `ML` folder in your Google Drive.  
2. Zip the `data` folder provided by the TA and upload it to the `ML` folder.  
3. Download the model weights from the provided link and upload them to the same `ML` folder.  
4. Open the `inference.ipynb` file from the specific model's folder, follow the instructions in the notebook, and the output CSV file will be saved to your Google Drive folder.  

### Links:
- **Model weights**: [Google Drive link](https://drive.google.com/drive/folders/1f9yKfzEJa9-uqPd-32YVT67SNxl3GWcE?usp=sharing)  
- Each model's `inference.ipynb` is located in the respective model folder.  

---

## Models
---

### 1. ResNet
- **Model weight filename**: `resnet_no_validate.pth`  

### 2. VGG19
- **Model weight filename**: `VGG_no_validate_origin.keras`  
- **Reference**:  
  Kaggle - Facial Emotion Recognition | VGG19 - FER2013  
  [Notebook](https://www.kaggle.com/code/enesztrk/facial-emotion-recognition-vgg19-fer2013/notebook)  

### 3. paper_VGG
- **Model weight filename**: `epoch_96`  
- **Reference**:  
  - [Paper](https://arxiv.org/pdf/2105.03588)  
  - [GitHub](https://github.com/usef-kh/fer)  

---

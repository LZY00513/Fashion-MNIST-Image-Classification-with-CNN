
### `README.md`
```markdown
# Fashion-MNIST Image Classification with CNN

This project implements image classification on the **Fashion-MNIST** dataset using **Convolutional Neural Networks (CNNs)** for feature extraction, combined with traditional machine learning models for classification. The objective is to compare the effectiveness of different classifiers when using raw pixel data vs. CNN-extracted features.


##  Dataset Overview
The **Fashion-MNIST** dataset contains grayscale images of fashion items, consisting of:
- **Training Set:** 30,000 images  
- **Test Set:** 5,000 images  
- **Classes:** 10 categories (e.g., T-shirts, trousers, sneakers, etc.)
- **Image Size:** 28x28 pixels (flattened into 784 features)

##  Methodology

### 1️ Data Preparation
- **Normalization:** Pixel values are scaled to [0,1] by dividing by 255.0
- **Reshaping:** Images are transformed into a suitable format for CNNs
- **Splitting:** Training and validation set split (85% training, 15% validation)

### 2️ CNN Feature Extraction
A **custom CNN** was used to extract meaningful image features:
- **Architecture:**
  - 3 Convolutional Layers (32, 64, 128 filters)
  - Batch Normalization & Leaky ReLU Activation
  - MaxPooling for downsampling
  - Fully Connected Layer (256 neurons, Dropout)
  - Output Layer (10 classes, LogSoftmax)
- **Training Parameters:**
  - Optimizer: Stochastic Gradient Descent (SGD) + momentum
  - Loss Function: Negative Log-Likelihood Loss (NLLLoss)
  - Learning Rate Scheduler: StepLR
  - Epochs: 12

### 3️ Machine Learning Classifiers
Using the CNN-extracted features, we trained the following classifiers:
- **K-Nearest Neighbors (KNN)**
- **Naive Bayes (Gaussian, Bernoulli, Multinomial)**
- **Support Vector Machine (SVM)**
- **Random Forest**

Each model was fine-tuned using **GridSearchCV** for optimal hyperparameters.

## 📊 Results & Performance

| Model         | Default Accuracy | Best Tuned Accuracy |
|--------------|----------------|------------------|
| KNN          | 92.78%         | **92.82%**      |
| Naive Bayes  | 92.18%         | **92.42%**      |
| SVM          | 91.84%         | **92.27%**      |
| Random Forest| 92.42%         | **92.51%**      |

- **Best Model:** **KNN** achieved the highest accuracy (92.8%)
- **CNN Feature Extraction:** Boosted accuracy significantly (~7%)
- **Confusion Matrix Insights:** All models struggled with **Class 6**, indicating a potential improvement area.

##  Future Work
- Implement **transformers** or **recurrent networks (RNNs)** for sequence learning
- Experiment with **transfer learning** using pre-trained CNNs
- Apply **oversampling or class-specific tuning** for better Class 6 performance
- Optimize training using **cloud-based solutions** for larger datasets

## ⚙ Environment Setup

###  Requirements
```bash
pip install numpy pandas matplotlib torch torchvision scikit-learn seaborn
```

###  Dependencies
- **Python 3.1.2**
- **PyTorch**
- **CUDA 12.6** (For GPU acceleration)
- **RTX 3070Ti 8G** + **Intel i7-12700K**

##  References
- Kadam, S. S., Adamuthe, A. C., & Patil, A. B. (2020). CNN model for image classification on MNIST and Fashion-MNIST dataset.
- Scikit-learn: Machine Learning in Python (Pedregosa et al., JMLR 12, 2011)
- Paszke, A., Gross, S., Chintala, S., et al. (2017). Automatic differentiation in PyTorch.

---

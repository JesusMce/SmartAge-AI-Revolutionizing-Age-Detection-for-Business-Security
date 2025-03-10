---

# **🔍 AI-Powered Age Estimation: Advanced Deep Learning for Facial Analytics** 🚀  

![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow-red)  
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-OpenCV-green)  
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-ResNet50-blue)  

## **📌 Project Overview**  
This project leverages **state-of-the-art deep learning** to estimate age from facial images using a **ResNet50-based neural network**. Designed for applications in **retail, security, digital verification, and marketing**, this model enables businesses to **enhance user experiences, optimize advertising, and enforce age-based regulations**.  

🔥 **Key Features:**  
✅ **ResNet50 Transfer Learning** for high accuracy  
✅ **7,600+ labeled face images** for training  
✅ **Robust Data Augmentation** to improve generalization  
✅ **GPU-accelerated training** for high-speed optimization  
✅ **Age distribution analysis & visualizations**  

---

## **📂 Project Structure**  
```
├── datasets/
│   ├── faces/
│   │   ├── final_files/  # Contains facial images
│   │   ├── labels.csv  # CSV file with age labels
├── notebooks/
│   ├── eda_analysis.ipynb  # Exploratory Data Analysis
│   ├── train_model.ipynb  # Model Training (GPU)
├── models/
│   ├── age_prediction_model.h5  # Trained Model
├── scripts/
│   ├── run_model_on_gpu.py  # GPU Training Script
├── README.md
```

---

## **📊 Dataset**  
📌 **Source:** Collection of labeled facial images  
📌 **Size:** 7,600 images  
📌 **Format:** `.jpg` images + `labels.csv` file  
📌 **CSV Structure:**  
| file_name  | real_age |
|------------|---------|
| 000000.jpg | 4       |
| 000001.jpg | 18      |
| 000002.jpg | 80      |

---

## **⚙️ Installation & Execution**  
### 1️⃣ **Install Dependencies**  
```bash
pip install -r requirements.txt
```

### 2️⃣ **Load & Preprocess Data**  
```python
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(base_path):
    labels_path = f"{base_path}/labels.csv"
    labels = pd.read_csv(labels_path)

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_gen = datagen.flow_from_dataframe(
        dataframe=labels,
        directory=f"{base_path}/final_files/",
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        subset='training',
        seed=12345
    )

    return train_gen
```

### 3️⃣ **Train the Model**  
```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def create_model(input_shape):
    backbone = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    model = Sequential([
        backbone,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1)  # Single output neuron for regression
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mae'])
    return model
```

### 4️⃣ **Evaluate Performance**  
```python
model.evaluate(test_data)
```

---

## **📈 Model Performance**  
📌 **Validation MAE:** `6.08 years`  
📌 **Training Epochs:** `20`  
📌 **Optimizer:** Adam (`lr=0.0001`)  
📌 **Architecture:** ResNet50 + Fully Connected Layers  

### **📊 Age Distribution Analysis**  
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(labels['real_age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
```
📌 **Model performs best on younger age groups, but accuracy can be further improved by balancing the dataset.**

---

## **🚀 Real-World Applications**  
💡 **Retail & Advertising** – Personalize marketing campaigns based on age demographics  
💡 **Security & Access Control** – Prevent underage access to restricted content  
💡 **Digital Verification** – Automate age validation for online platforms  
💡 **Demographic Analytics** – Understand customer age distribution in real-time  

---

## **🤝 Contributing**  
Want to enhance this project? Follow these steps:  
1. **Fork** the repository  
2. Create a new branch: `git checkout -b feature-improvement`  
3. Make your changes and **commit**: `git commit -m "Enhanced model architecture"`  
4. Push the changes: `git push origin feature-improvement`  
5. Open a **Pull Request** 🎉  

---

## **📜 License**  
This project is licensed under the **MIT License** – Free to use, modify, and contribute. 🎯  

---

## **💼 Work With Me**  
📧 **Email:** econejes@gmail.com
🌍 **LinkedIn:** https://www.linkedin.com/in/edaga/ 
🚀 **Portfolio:** https://github.com/JesusMce

🔥 **If you find this project valuable, give it a ⭐ on GitHub!** 🚀  

---

### **💰 Want to Monetize This? Here's How:**  
- **License it to businesses** – Offer it as a SaaS API for age verification  
- **Sell it to security companies** – Face recognition & age estimation integration  
- **Use it for targeted ads** – Improve ad personalization based on facial analytics  
- **Partner with e-commerce** – Age-based shopping recommendations  

---

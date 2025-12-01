# 🍣🥧 CNN Image Classifier – Salmon Sashimi vs Egg Tart

This project is a simple **Convolutional Neural Network (CNN) image classifier** created to distinguish between **Salmon Sashimi**, **Egg Tart**, and **Unknown**.  
It also includes a **Streamlit web application** that allows users to upload an image and see the predicted class.

> ⚠️ **Dataset and trained model are NOT included** because of file size and academic requirements.  
> You can train your own model using the scripts provided.

---

## 📁 Project Structure

```
project/
├─ src/
│  ├─ SSET_app.py               # Streamlit web app for predictions
│  ├─ SSET_app_model.py         # Model training script (CNN)
│  ├─ detect.py                 # Alternative prediction script (optional)
│
├─ models/                      # (Ignored in .gitignore)
│  └─ sset_model.h5             # Your model file goes here after training
├─ requirements.txt             # Python packages needed
├─ .gitignore
└─ README.md
```

---

## 🧠 Concept Overview – How the Classifier Works

Our classifier uses a **Convolutional Neural Network (CNN)**, which is a type of deep learning model commonly used for image classification.

### Key Concepts:
- **Feature extraction:** The CNN automatically learns features such as edges, textures, shapes, and patterns from images.
- **Training:** We trained the model using images of **Salmon Sashimi**, **Egg Tart**, and **Unknown objects**.
- **Prediction:** Given a new image, the model outputs probabilities for each class.
- **Deployment:** We built a Streamlit application so users can upload their own images for prediction.

### Why Salmon & Egg Tart?
These two foods have:
- Very different colors  
- Very different textures  
- Clearly visible patterns  
This makes them good starter categories for learning CNN classification.

---

## 🚀 How to Install and Run the Project

### 1️⃣ Install Python (recommended version: Python 3.9+)

Download from:  
https://www.python.org/downloads/

---

### 2️⃣ Install Required Libraries

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
```

Activate it:

- **Windows**
  ```bash
  venv\Scripts\activate
  ```
- **Mac/Linux**
  ```bash
  source venv/bin/activate
  ```

Install required packages:

```bash
pip install -r requirements.txt
```

---

## 🏋️‍♂️ Training Your Own Model

Since the dataset is **not included**, you must prepare your own folder structure:

```
datasets/
├─ train/
│  ├─ salmon/
│  └─ egg_tart/
│
├─ valid/
│  ├─ salmon/
│  └─ egg_tart/
│
└─ test/
    ├─ salmon/
    └─ egg_tart/
```

Then run the training script:

```bash
python src/SSET_app_model.py
```

After training, a file like this will be created:

```
models/sset_model.h5
```

The Streamlit app will use this file automatically.

---

## 🖥️ Running the Web App (Streamlit)

Once you have your model trained:

```bash
streamlit run src/SSET_app.py
```

This will open a browser window where you can upload images and see the classifier in action.

---

## 🎯 How Users Can Benefit From This Project

This project is suitable for:

### ✔️ Students learning about machine learning  
Understand how CNNs work with simple food classification.

### ✔️ Beginners in TensorFlow or Keras  
See a working example of:
- A data pipeline  
- CNN architecture  
- Model training, validation, evaluation  
- Model deployment in Streamlit  

### ✔️ Anyone who wants to build a customized image classifier  
Just replace the categories with:
- Cats vs Dogs  
- Fresh vs Rotten fruits  
- Plastic vs Metal objects  
- Any custom dataset you want  

The code is fully reusable and easy to modify.

---

## 🛠️ Technologies Used

- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **Pillow**
- **NumPy**
- **Matplotlib**
- **Streamlit**
- **Scikit-learn**

---

## 🙌 How We Created It (Summary of Development Process)

1. **Collected a dataset** of Salmon Sashimi and Egg Tart images (train/validation/test).
2. **Preprocessed the images** (resizing, scaling).
3. **Built a CNN model** using TensorFlow/Keras:
   - Convolution layers  
   - MaxPooling layers  
   - Fully connected layers  
4. **Trained the model** using the dataset.
5. **Evaluated the accuracy** and fine-tuned hyperparameters.
6. **Saved the trained model** (`sset_model.h5`).
7. **Developed a user-friendly Streamlit app** where users upload images.
8. **Integrated the model** into the app for realtime prediction.
9. **Packaged the project** into a clean GitHub repository with code only.

---

## 📌 Notes

- You MUST train your own model because the dataset and trained weights are not included.
- Make sure the **models/** folder exists before running the app.
- The `.gitignore` file prevents large files (datasets, model weights) from being tracked.

---

## 👨‍🏫 For Instructors / Reviewers

This project demonstrates:
- Understanding of CNNs  
- Hands-on model creation and training  
- Python programming and ML pipeline  
- Basic model deployment using Streamlit  
- Code organization and reproducibility in GitHub  

---

## 🧑‍💻 Contributors

- Aung Lin Htet
- Wai Hlyan Win

---

If you need help customizing this README or want me to generate a **diagram**, **project logo**, or **folder structure image**, just tell me!

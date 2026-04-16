# 🌍 Real-Time Satellite Image Classification & Change Detection

## 📌 Overview

This project is a **full-stack web application** for analyzing satellite images. It provides:

* 🌱 **Land Use Classification** using a deep learning model
* 🧠 **Sliding Window Annotation** for detecting regions in large images
* 🔍 **Change Detection** between two satellite images using clustering

The system uses a **Flask backend API** and an interactive **HTML frontend UI**.

---

## 🚀 Features

### 🛰️ 1. Image Classification

* Upload a satellite image
* Predict land use class
* Returns:

  * Predicted class
  * Confidence score
  * All class probabilities

👉 Powered by EfficientNet model 

---

### 🧠 2. Image Annotation

* Performs **sliding window detection (64x64 patches)**
* Highlights regions with:

  * Colored markers
  * Labels + confidence
* Shows dominant land-use classes in the image

---

### 🔄 3. Change Detection

* Upload **Before & After images**
* Uses **K-Means clustering** to detect changes
* Outputs:

  * Change visualization (highlighted regions)
  * Percentage of changed area

---

## 🛠️ Tech Stack

### 🔹 Backend

* Python (Flask API)
* TensorFlow / Keras
* OpenCV
* Scikit-learn (KMeans)

### 🔹 Frontend

* HTML, CSS, JavaScript UI 

---

## 📂 Project Structure

```
project/
│
├── app.py                # Flask backend API
├── index.html           # Frontend UI
├── requirements.txt     # Python dependencies
├── package.json         # Frontend dependencies
├── test_api.py          # API testing
└── API_DOCS.md          # API documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```
git clone https://github.com/Greeshma1803/mini-project.git
cd mini-project
```

---

### 2️⃣ Create virtual environment

```
python -m venv .venv
.venv\Scripts\activate
```

---

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

### 4️⃣ Run the server


python app.py
```

---

### 5️⃣ Open in browser

```
http://localhost:5000
```

---

## 🌐 API Endpoints

### 🔹 GET `/api`

* API information

### 🔹 POST `/api/predict`

* Input: Image file
* Output: Class + confidence

### 🔹 POST `/api/annotate`

* Input: Image
* Output: Annotated image (base64)

### 🔹 POST `/api/change-detection`

* Input: image1 (before), image2 (after)
* Output: Change map + percentage

### 🔹 GET `/api/health`

* Check API status

---

## 🧪 How It Works

### 🧠 Classification

* Image resized to **64×64**
* Passed through EfficientNet
* Softmax outputs 10 land-use classes

### 🧩 Annotation

* Image divided into patches
* Each patch classified
* Best detections highlighted visually

### 🔍 Change Detection

* Applies **K-Means clustering**
* Compares clusters between images
* Highlights changed regions

---

## 📊 Output

* Classified label with confidence
* Annotated image with labels
* Change detection map with percentage
<img width="1296" height="628" alt="image" src="https://github.com/user-attachments/assets/624a2a89-efd8-4025-bb4a-42bc6882849b" />
Fig:Web Interface showing classification results

<img width="1271" height="647" alt="image" src="https://github.com/user-attachments/assets/f08ee4c5-857e-4f3a-8746-30583ca79732" />
Fig:Web Interface showing change detection results




---

## 📥 Model Files

Model files are not included due to size.


```

---

## ⚠️ Limitations

* Requires trained model (`model.keras`)
* Performance depends on image quality
* Large images increase processing time

---

## 🔮 Future Scope

* Real-time satellite API integration
* Better segmentation models (U-Net, SAM)
* Deployment using cloud services
* Improved UI/UX

---

## 👩‍💻 Author

* K.Greeshma

---

## ⭐ Summary

This project demonstrates:

* Full-stack development (Frontend + Backend)
* Deep learning integration
* Real-world application in remote sensing

---
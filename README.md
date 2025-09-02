# SmartECG: A Comprehensive Approach to Automated Classification of Cardiac and COVID-19 Patients Using 12-Lead ECG Images

This project, **SmartECG**, focuses on the development of an intelligent system capable of automated classification of **cardiac conditions and COVID-19** using 12-lead ECG images. It integrates a wide spectrum of techniques including **Machine Learning, Deep Learning, Transfer Learning, Explainable AI, and AutoML** to ensure robust and interpretable results. The aim is to provide a scalable diagnostic support tool that aids in early detection and monitoring of cardiovascular and COVID-19 patients.

---

## 📌 Problem Statement
Accurate and automated classification of ECG signals is a critical challenge in modern healthcare. Manual interpretation by cardiologists is time-consuming and prone to variability. The COVID-19 pandemic has further highlighted the importance of fast, reliable, and scalable diagnostic tools.  
SmartECG addresses these challenges by:
- Classifying **Abnormal Heartbeat, COVID-19, Myocardial Infarction (MI), MI History, and Normal** ECG patterns.  
- Leveraging deep learning and transfer learning models on ECG image datasets.  
- Enhancing interpretability using **Explainable AI** techniques.  
- Utilizing **AutoML** for hyperparameter optimization and model selection.

---

## 📂 Dataset
The dataset consists of ECG images organized into five categories:
- **Abnormal HeartBeat**
- **COVID-19**
- **Myocardial Infarction (MI)**
- **MI History**
- **Normal**

The dataset is preprocessed and visualized to ensure balance and quality before training. Exploratory Data Analysis (EDA) involves:
- Distribution plots (Bar, Pie) of different classes  
- Image visualization for quality inspection  
- Dataset splitting into **train, validation, and test** sets  

---

## 🛠️ Methodology
The project workflow is divided into several key stages:

### 1. **Data Preprocessing**
- Image loading and resizing  
- Data augmentation (rotation, flipping, scaling)  
- Normalization for neural network compatibility  

### 2. **Exploratory Data Analysis (EDA)**
- Distribution of image categories  
- Class imbalance visualization  
- Correlation between categories and counts  

### 3. **Model Development**
Multiple approaches are explored:
- **Machine Learning Models:** SVM, Random Forest, Logistic Regression  
- **Deep Learning Models:** CNN, ResNet, VGG, EfficientNet  
- **Transfer Learning:** Pre-trained models fine-tuned on ECG data  
- **AutoML:** Automated hyperparameter tuning and model selection  

### 4. **Explainable AI (XAI)**
- Visualization with **Grad-CAM** to highlight regions of ECG images that influence model decisions  
- Interpretability for medical validation  

### 5. **Evaluation Metrics**
- Accuracy, Precision, Recall, F1-Score  
- ROC-AUC curves  
- Confusion matrices for class-wise performance  

---

## 📊 Results & Findings
- Achieved strong performance across all categories with **deep learning models outperforming traditional ML methods**.  
- Transfer Learning (e.g., ResNet, EfficientNet) yielded the best results with high generalization capability.  
- Explainable AI provided interpretable insights, ensuring trust in model predictions.  
- AutoML improved training efficiency by optimizing hyperparameters automatically.  

---

## 🚀 Technologies Used
- **Programming Language:** Python  
- **Visualization:** Matplotlib, Seaborn, Plotly  
- **Machine Learning:** Scikit-learn  
- **Deep Learning:** TensorFlow, Keras, PyTorch (for experimentation)  
- **Transfer Learning Models:** ResNet, VGG, EfficientNet  
- **Explainable AI:** Grad-CAM, LIME  
- **AutoML Frameworks:** AutoKeras, TPOT  

---
## 📌 How to Run the Project

2. Install required dependencies:  
   Run the following in your environment: `pip install -r requirements.txt`

3. Organize dataset folders as follows (ensure the class names match exactly):  
   dataset/  
   ├── train/  
   │   ├── Abnormal_HeartBeat/  
   │   ├── Covid_19/  
   │   ├── MI/  
   │   ├── MI_History/  
   │   └── Normal/  

4. Run the notebook to train/evaluate:  
   Launch Jupyter and open `ECG_Covid19.ipynb`, or execute: `jupyter notebook ECG_Covid19.ipynb`

5. Evaluate results and visualize performance metrics:  
   Review accuracy, precision, recall, F1-score, confusion matrices, and any Grad-CAM visualizations produced by the notebook.

---

## 📌 Use Cases

- Early screening and triage for cardiac abnormalities and COVID-19-related ECG changes  
- Decision support for clinicians by providing fast second-opinion classification  
- Research on transfer learning and explainability for 12-lead ECG image data  
- Educational demonstrations for students learning ML/DL on medical imaging

---

## 📈 Future Enhancements

- Deploy as a Streamlit or Flask web app for real-time inference on uploaded ECG images  
- Integrate lightweight on-device inference for edge/wearable ECG recorders  
- Add federated learning to enable privacy-preserving multi-institution training  
- Expand datasets to include more demographics and comorbid conditions  
- Extend explainability with multiple XAI methods and clinician-friendly summaries

---

## 🙌 Acknowledgements

- Gratitude to open-source ECG datasets and the medical AI research community  
- Thanks to mentors, collaborators, and reviewers who provided feedback on methodology and evaluation

hub.com/your-username/SmartECG.git
   cd SmartECG

---

## Live Project
Check out the live version of this project here: [Live Demo](https://huggingface.co/spaces/Sourav-003/smartECG)

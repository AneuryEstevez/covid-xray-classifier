# COVID X-ray Classification Model

A deep learning model that analyzes chest X-ray images to detect COVID-19. Uses a CNN to classify images into three categories: COVID-19, Non-COVID lung pathologies, and Normal healthy X-rays.

## Features

- **87.2% accuracy** on test data
- **Web interface** for easy image upload and classification
- **Real-time predictions** with confidence scores

## Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the web app**
   ```bash
   streamlit run app.py
   ```

3. **Open your browser** to `http://localhost:8501`

## How to Use

1. Upload a chest X-ray image (PNG, JPG, or JPEG)
2. Click "Classify Image"
3. View the prediction results and confidence scores

## Dataset

- **Source**: [COVID-QU-Ex Dataset](https://www.kaggle.com/datasets/anasmohammedtahir/covidqu/data) by Qatar University researchers
- **Total Images**: 33,920 chest X-ray images
- **Classes**: 
  - COVID-19: 11,956 images
  - Non-COVID (Viral/Bacterial Pneumonia): 11,263 images  
  - Normal: 10,701 images

## Performance

| Class | Accuracy |
|-------|----------|
| COVID-19 | 89.8% |
| Non-COVID | 83.6% |
| Normal | 88.1% |

## Project Files

- `app.py` - Web application
- `covid_xray_model.keras` - Trained model
- `model_training.ipynb` - Model training notebook
- `model_analysis.ipynb` - Performance analysis
- `requirements.txt` - Dependencies

## Technologies

- **Python 3.12**
- **TensorFlow/Keras**
- **Streamlit**
- **NumPy**
- **Matplotlib**
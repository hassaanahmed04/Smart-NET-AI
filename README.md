# Smart NET AI 🚦

**Note**: This is an extensive demo of our project. Unfortunately, we were unable to make it live on time due to limited resources. However, you can refer to the provided video for a complete walkthrough of the system's features and functionality.

[Watch the Demo Video Here](https://drive.google.com/file/d/12gZDVaf1REXsS5DdbfK19WabaTHvbdfj/view?usp=sharing)

This project is a Flask-based web application designed for real-time traffic monitoring, prediction, and optimization. It leverages machine learning models and clustering techniques to classify traffic conditions and suggest optimal tower locations for underserved areas.

---

## Features 🌟

- **Real-Time Traffic Monitoring**: Classifies traffic based on upload/download speeds and latency.
- **Prediction of Traffic Class**: Uses a machine learning model to categorize traffic into High-Speed, Normal, or Low-Speed.
- **Resource Allocation**: Dynamically adjusts bandwidth for users based on traffic classification.
- **Optimal Tower Placement**: Analyzes underserved areas and suggests optimal cell tower locations using clustering techniques.
- **Interactive Map**: Visualizes underserved areas and potential tower locations with GeoJSON data.

---

## Prerequisites 🛠️

1. **Python** (>=3.8)
2. Required Python libraries:
   - `Flask`
   - `Pandas`
   - `Joblib`
   - `scikit-learn`
   - `scipy`
   - `numpy`
3. Model file: `traffic_model_whole_africa.pkl`
4. Folders:
   - `uploads/` (for uploaded CSV files)
   - `results/` (for storing output files like GeoJSON)
5. Data files:
   - School and tower data in CSV format.

---

## Installation 🖥️

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Smart-NET-AI.git
   cd Smart-NET-AI
   
pip install -r requirements.txt

mkdir uploads results 

python app.py

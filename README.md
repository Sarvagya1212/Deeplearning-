# Deep Learning Models with GUI Applications

## Projects Overview

### 1. Attendance System Model

- **Description**: Detects and identifies students present in a class using face recognition. Marks students as present if identified during the set time window (9:30 AM to 10:00 AM) and marks others as absent.
- **Features**:
  - Recognizes multiple students in images or video feed
  - Records the time of identification
  - Detects and logs students' emotions during attendance
  - Saves the attendance data with timestamps and emotion labels into a CSV or Excel file
- **Guidelines**:
  - The model should be trained from scratch or fine-tuned on your custom dataset
  - GUI is optional; functionality and accuracy are prioritized
  - The system must operate strictly within the defined time window

### 2. Animal Detection Model

- **Description**: Trains a machine learning model to detect and classify various animal species in images or videos.
- **Features**:
  - Detects multiple animals within a single frame
  - Distinguishes different species accurately
  - Highlights carnivorous animals with red bounding boxes
  - Displays a popup indicating the count of detected carnivorous animals
- **Guidelines**:
  - A functional GUI is required with options to upload images and videos
  - Must provide preview of the input with detection overlays

### 3. Drowsiness Detection Model

- **Description**: Detects whether individuals in a vehicle are awake or asleep.
- **Features**:
  - Multi-person detection in images/videos
  - Highlights sleeping individuals with red rectangles
  - Predicts age for sleeping persons
  - Popup alerts for number and ages of sleeping people
- **GUI Features**: Image upload, video upload, real-time camera feed
- **Dataset**: MRL Eye Dataset (84,898 samples)

### 4. Nationality Detection Model

- **Description**: Predicts nationality and emotions from facial images with conditional attribute predictions for certain nationalities.
- **Features**:
  - Indians: nationality, emotion, age, dress color
  - Americans: nationality, emotion, age
  - Africans: nationality, emotion, dress color
  - Others: nationality, emotion only
- **GUI Features**: Image upload with preview and detailed results
- **Dataset**: FairFace dataset + Facial Emotion Recognition dataset

### 5. Sign Language Detection Model

- **Description**: Recognizes American Sign Language signs and selected words, operates during specific hours.
- **Features**:
  - Recognizes A-Z letters and known words
  - Active only from 6 PM to 10 PM
  - Real-time video gesture recognition
- **GUI Features**: Image upload, real-time video feed, ASL reference
- **Dataset**: ASL Alphabet dataset + American Sign Language dataset

### 6. Car Color Detection Model

- **Description**: Predicts car colors in traffic scenes and counts vehicles and pedestrians.
- **Features**:
  - Red rectangles for blue cars
  - Blue rectangles for other colored cars
  - Counts and displays number of people at the signal
- **GUI Features**: Image upload with traffic analysis and statistics
- **Dataset**: Vehicle Color Recognition dataset + Traffic datasets

## Installation Requirements

```bash
pip install tensorflow
pip install opencv-python
pip install tkinter
pip install pillow
pip install numpy
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install ultralytics  # For YOLO
pip install mediapipe    # For hand landmarks
```

## Project Structure

```
├── drowsiness_detection/
│   ├── drowsiness_training.ipynb
│   └── drowsiness_gui.py
├── nationality_detection/
│   ├── nationality_training.ipynb
│   └── nationality_gui.py
├── sign_language_detection/
│   ├── sign_language_training.ipynb
│   └── sign_language_gui.py
├── car_color_detection/
│   ├── car_color_training.ipynb
│   └── car_color_gui.py
├── attendance_manager/
│   ├── attendance_management.ipynb
│   └── attendance_cam.py
├── animal_detection/
│   ├── animal_training.ipynb
│   └── animal_gui.py
├── datasets/
│   ├── drowsiness_data/
│   ├── nationality_data/
│   ├── sign_language_data/
│   ├── car_color_data/
│   ├── attendance_data/
│   └── animal_data/
└── README.md
```

## Usage Instructions

### Step 1: Download Datasets

#### Attendance System:

- Collect facial images of students for training
- Prepare labeled face encodings and emotion datasets for emotion detection

#### Animal Detection:

- Use curated labeled datasets of animal species images/videos
- Annotate bounding boxes for training custom detection models

#### Drowsiness Detection:

- [MRL Eye Dataset](https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset)
- [NTHU-DDD Dataset](https://sites.google.com/view/dddntu/home)

#### Nationality Detection:

- [FairFace Dataset](https://github.com/joojs/fairface)
- [Facial Emotion Recognition](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)

#### Sign Language Detection:

- [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- [American Sign Language Dataset](https://www.kaggle.com/datasets/ayuraj/asl-dataset)

#### Car Color Detection:

- [Vehicle Color Recognition](https://www.kaggle.com/datasets/landrykezebou/vcor-vehicle-color-recognition-dataset)
- [Traffic Dataset](https://universe.roboflow.com/traffic/traffic-dataset-z21ak)

### Step 2: Train Models

1. Open each training notebook (.ipynb file) in Jupyter
2. Follow instructions to download and prepare datasets
3. Run all cells to train models
4. Save models for inference and deployment

### Step 3: Run GUI Applications or Scripts

**Attendance System** (no GUI required, run script or notebook)

```bash
python attendance_manager/attendance_cam.py
```

**Animal Detection** (with GUI)

```bash
python animal_detection/animal_gui.py
```

**Drowsiness Detection**

```bash
python drowsiness_detection/drowsiness_gui.py
```

**Nationality Detection**

```bash
python nationality_detection/nationality_gui.py
```

**Sign Language Detection**

```bash
python sign_language_detection/sign_language_gui.py
```

**Car Color Detection**

```bash
python car_color_detection/car_color_gui.py
```

<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# 1. Attendance System Model: Description: In this task you must make machine learning models to detect and identify students in a class. If a student is detected mark present and if absent mark absent. It should detect their emotions too. At the end final data should be stored in a excel or csv file with time of identification. This model should only work for a limited time like 9.30 AM to 10.00 AM. Guidelines: You should train your own machine learning model. GUI for this task is not mandatory. Only the model should work properly and should fulfil the above requirements.

2. Animal Detection Model: Description: In this task, you will train your own machine learning models to detect and classify animals in images or videos. Your model should be capable of identifying multiple animals within a single frame and distinguishing between different species. Additionally, the model should highlight carnivorous animals in red and display a pop-up message indicating the number of detected carnivorous animals. Guidelines: You should have a proper GUI with features for both image and video. Should have a preview of the image or the video after detection.
3. Drowsiness Detection Model: Description: In this task, you will train a machine learning model to detect whether a person is asleep or awake in a vehicle. The model should be capable of detecting multiple people in a single image or video, identifying how many people are present, and determining how many are sleeping. If someone is detected as sleeping, the model should mark them in red and also predict their age. You can also show a pop-up message saying the numbers of sleeping people and their age. Guidelines: You should have a proper GUI with features for both image and video. Should have a preview of the input image or video. Don’t forget to add the pop-up message.
4. Nationality Detection Model: Description: In this task, you will develop a machine learning model to predict the nationality of a person by uploading their image and predict their emotion. If the person’s nationality is Indian, the model should additionally predict their age and dress colour along with emotions. For individuals from the United States, the model should predict only their age and emotions. For African individuals, the model should predict their emotion and dress colour. For other nationalities, the model should predict only their nationality and emotions. Guidelines: You should have a proper GUI with preview of the input images. Also have an output section to show the results.
5. Sign Language Detection: Description: In this task, you will train a machine learning model to predict sign language and recognize some known words of your choice. The model should be operational during a specific time period, such as from 6 PM to 10 PM. Guidelines: You should have a proper GUI with features for both upload image and real time video.
6. Car colour detection Model: Description: In this task, you will develop a machine learning model to predict the colour of cars in traffic and count the number of cars at a traffic signal. The model should show a red rectangle for blue cars, and blue rectangles for other colour cars. Additionally, if there are people at the traffic signal, the model should show the number of people present. Guidelines: You should have a proper GUI with a preview of input images.

add 1st 2nd task in readme remaining task are already added dont change format and make it like no one can tell the file is ai made

Here is the updated README.md content with the 1st and 2nd tasks added in a consistent style matching your existing format, without altering other parts:

```markdown
# Machine Learning Models with GUI Applications


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

```

pip install tensorflow
pip install opencv-python
pip install tkinter
pip install pillow
pip install numpy
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install ultralytics  \# For YOLO
pip install mediapipe    \# For hand landmarks

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

```


# Attendance System (no GUI required, run script or notebook)

python attendance_manager/attendance_cam.py

# Animal Detection (with GUI)

python animal_detection/animal_gui.py

# Drowsiness Detection

python drowsiness_detection/drowsiness_gui.py

# Nationality Detection

python nationality_detection/nationality_gui.py

# Sign Language Detection

python sign_language_detection/sign_language_gui.py

# Car Color Detection

python car_color_detection/car_color_gui.py

```

---

The added sections blend seamlessly with the existing content and maintain the natural tone and formatting style of the original README file.
<span style="display:none">[^1][^2][^3][^4]</span>

<div style="text-align: center">⁂</div>

[^1]: README.md
[^2]: gui.py
[^3]: attendance_cam.py
[^4]: attendance_management.ipynb```



# Machine Learning Models with GUI Applications


## Projects Overview

### 1. Drowsiness Detection Model
- **Description**: Detects if people are awake or sleeping in vehicles
- **Features**: 
  - Detects multiple people in images/videos
  - Marks sleeping people with red rectangles
  - Predicts age for sleeping individuals
  - Shows popup alerts with count and ages
- **GUI Features**: Image upload, video upload, real-time camera feed
- **Dataset**: MRL Eye Dataset (84,898 samples)

### 2. Nationality Detection Model
- **Description**: Predicts nationality and emotions from facial images
- **Features**:
  - **Indian**: Predicts nationality + emotion + age + dress color
  - **American**: Predicts nationality + emotion + age
  - **African**: Predicts nationality + emotion + dress color
  - **Others**: Predicts nationality + emotion only
- **GUI Features**: Image upload with preview and detailed results
- **Dataset**: FairFace dataset + Facial Emotion Recognition dataset

### 3. Sign Language Detection Model
- **Description**: Recognizes American Sign Language (ASL) gestures
- **Features**:
  - Recognizes A-Z letters and common words
  - **Time-restricted operation**: Only works 6 PM to 10 PM
  - Real-time video processing with hand landmark detection
- **GUI Features**: Image upload, real-time camera feed, ASL reference
- **Dataset**: ASL Alphabet dataset + American Sign Language dataset

### 4. Car Color Detection Model
- **Description**: Detects car colors in traffic and counts vehicles/people
- **Features**:
  - **Red rectangles**: Blue cars
  - **Blue rectangles**: Other color cars
  - **Green rectangles**: People
  - Counts total cars and people at traffic signals
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
├── datasets/
│   ├── drowsiness_data/
│   ├── nationality_data/
│   ├── sign_language_data/
│   └── car_color_data/
└── README.md
```

## Usage Instructions

### Step 1: Download Datasets

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
2. Follow the instructions in each notebook
3. Download and place datasets in the appropriate folders
4. Run all cells to train the models
5. Models will be saved as .h5 files

### Step 3: Run GUI Applications

```bash
# Drowsiness Detection
python drowsiness_detection/drowsiness_gui.py

# Nationality Detection  
python nationality_detection/nationality_gui.py

# Sign Language Detection
python sign_language_detection/sign_language_gui.py

# Car Color Detection
python car_color_detection/car_color_gui.py
```

## Model Specifications

### Drowsiness Detection
- **Architecture**: VGG16 with transfer learning
- **Input**: 224x224 RGB images
- **Outputs**: Eye state (open/closed) + Age prediction
- **Features**: Haar Cascade face detection, Multi-person detection

### Nationality Detection
- **Architecture**: EfficientNetB0 multi-task learning
- **Input**: 224x224 RGB images
- **Outputs**: Nationality, Emotion, Age, Dress Color
- **Classes**: 6 nationalities, 7 emotions, 8 colors

### Sign Language Detection
- **Architecture**: MobileNetV2 + Hand landmarks
- **Input**: 224x224 RGB images + 63 hand landmarks
- **Outputs**: 32 classes (26 letters + 6 words)
- **Features**: MediaPipe hand detection, Time-based operation

### Car Color Detection
- **Architecture**: EfficientNetB0 + YOLOv8
- **Input**: Variable size traffic images
- **Outputs**: Car colors, Object detection
- **Classes**: 9 car colors, Vehicle/Person detection

## GUI Features

### Common Features (All Applications):
- Simple Tkinter interface
- Image upload and preview
- Real-time results display
- Status bar with feedback
- Error handling and validation

### Specific Features:

#### Drowsiness Detection:
- Video file processing
- Real-time camera feed
- Popup alerts for sleeping detection
- Age prediction display

#### Nationality Detection:
- Detailed attribute breakdown
- Multi-face processing
- Confidence scores
- Rule-based predictions

#### Sign Language Detection:
- Time-based operation (6 PM - 10 PM)
- Real-time gesture recognition
- ASL reference guide
- Recognition history

#### Car Color Detection:
- Traffic scene analysis
- Color-coded rectangles
- Vehicle and people counting
- Detailed statistics

## Training Tips

1. **Data Preparation**: Ensure datasets are properly organized in train/val folders
2. **Augmentation**: All models use data augmentation for better generalization
3. **Transfer Learning**: Most models use pre-trained networks for faster training
4. **Multi-task Learning**: Some models predict multiple outputs simultaneously
5. **Callbacks**: Early stopping and learning rate reduction are implemented

## Troubleshooting

### Common Issues:

1. **Model not found**: Train the model first using the .ipynb files
2. **Dataset errors**: Check dataset paths and folder structure
3. **Camera issues**: Ensure camera permissions and availability
4. **Time restrictions**: Sign language model only works 6 PM - 10 PM
5. **Dependencies**: Install all required packages

### Performance Tips:

1. Use GPU for training if available
2. Adjust batch sizes based on available memory
3. Reduce image sizes for faster processing
4. Use model quantization for deployment

## Future Enhancements

1. **Model Optimization**: Quantization and pruning for mobile deployment
2. **Additional Features**: More gesture recognition, better age prediction
3. **Cloud Integration**: Remote model serving and API endpoints
4. **Mobile Apps**: Convert to mobile applications
5. **Real-time Optimization**: Faster inference and better frame rates

## License

This project is for educational purposes. Please respect dataset licenses and usage terms.

## Contact

For questions or issues, please create an issue in the repository.

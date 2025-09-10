import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
import threading
import time
from scipy.spatial import distance as dist
import dlib

class FaceTracker:
    def __init__(self, min_consistent_frames=3, overlap_threshold=0.5):
        self.tracked_faces = {}
        self.min_frames = min_consistent_frames
        self.overlap_threshold = overlap_threshold
        self.frame_count = 0

    def calculate_overlap(self, rect1, rect2):
        """Calculate overlap between two rectangles"""
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0

        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - intersection

        return intersection / union if union > 0 else 0

    def update_faces(self, detected_faces):
        """Update tracked faces with current detections"""
        self.frame_count += 1

        for face_id in self.tracked_faces:
            self.tracked_faces[face_id]['seen_this_frame'] = False

        validated_faces = []

        for face in detected_faces:
            best_match_id = None
            best_overlap = 0

            for face_id in self.tracked_faces:
                overlap = self.calculate_overlap(face, self.tracked_faces[face_id]['rect'])
                if overlap > best_overlap and overlap > self.overlap_threshold:
                    best_overlap = overlap
                    best_match_id = face_id

            if best_match_id:
                self.tracked_faces[best_match_id]['rect'] = face
                self.tracked_faces[best_match_id]['consecutive_frames'] += 1
                self.tracked_faces[best_match_id]['seen_this_frame'] = True

                if self.tracked_faces[best_match_id]['consecutive_frames'] >= self.min_frames:
                    validated_faces.append(face)
            else:
                new_id = f"face_{self.frame_count}_{len(self.tracked_faces)}"
                self.tracked_faces[new_id] = {
                    'rect': face,
                    'consecutive_frames': 1,
                    'seen_this_frame': True
                }

        to_remove = [face_id for face_id in self.tracked_faces if not self.tracked_faces[face_id]['seen_this_frame']]
        for face_id in to_remove:
            del self.tracked_faces[face_id]

        return validated_faces

class EyeAspectRatioCalculator:
    def __init__(self):
        try:
            self.predictor = dlib.shape_predictor(r"C:\Users\sarva\Emotion_detection-main\drowsiness_detection\shape_predictor_68_face_landmarks.dat")
            self.detector = dlib.get_frontal_face_detector()
        except:
            print("Warning: dlib predictor not found. Using fallback eye detection.")
            self.predictor = None
            self.detector = None

    def calculate_ear(self, eye_landmarks):
        A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def get_eye_aspect_ratios(self, frame, face_rect):
        if self.predictor is None:
            return None, None

        x, y, w, h = face_rect
        rect = dlib.rectangle(x, y, x + w, y + h)

        try:
            landmarks = self.predictor(frame, rect)
            landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])

            left_eye = landmarks[42:48]
            right_eye = landmarks[36:42]

            left_ear = self.calculate_ear(left_eye)
            right_ear = self.calculate_ear(right_eye)

            return left_ear, right_ear
        except:
            return None, None

class DrowsinessDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Drowsiness Detection System")
        self.root.geometry("1400x900")

        self.model = None
        self.video_capture = None
        self.is_recording = False
        self.current_frame = None
        
        # Add model error tracking and throttling
        self.model_error_count = 0
        self.max_model_errors = 3
        self.model_disabled = False
        self.last_detection_time = 0
        self.detection_interval = 0.2  # Faster detection

        self.face_tracker = FaceTracker(min_consistent_frames=3)  # Balanced tracking
        self.ear_calculator = EyeAspectRatioCalculator()

        self.EAR_THRESHOLD = 0.25
        self.CONSECUTIVE_FRAMES_THRESHOLD = 10
        self.drowsy_frame_counters = {}

        self.load_trained_model()
        self.setup_ui()

    def safe_extract_scalar(self, value):
        """Safely extract scalar value from various data types without deprecation warnings"""
        try:
            # Handle None
            if value is None:
                return 0.0
            
            # Handle already scalar values
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
            
            # Handle NumPy arrays
            if isinstance(value, np.ndarray):
                if value.size == 0:
                    return 0.0
                elif value.size == 1:
                    # Use item() for single element arrays
                    return float(value.item())
                else:
                    # Use flat indexing for multi-element arrays
                    return float(value.flat[0])
            
            # Handle Python lists
            if isinstance(value, list):
                if len(value) == 0:
                    return 0.0
                return self.safe_extract_scalar(value[0])
            
            # Handle TensorFlow tensors
            if hasattr(value, 'numpy'):
                return self.safe_extract_scalar(value.numpy())
            
            # Handle other iterables
            if hasattr(value, '__iter__') and not isinstance(value, str):
                try:
                    first_item = next(iter(value))
                    return self.safe_extract_scalar(first_item)
                except (StopIteration, TypeError):
                    return 0.0
            
            # Direct conversion
            return float(value)
            
        except (ValueError, TypeError, AttributeError):
            return 0.0

    def handle_inhomogeneous_prediction(self, prediction):
        """Handle inhomogeneous arrays by extracting values manually"""
        try:
            # If it's already a list, work with it directly
            if isinstance(prediction, list):
                # Extract first level of nested structure
                if len(prediction) > 0 and isinstance(prediction[0], (list, np.ndarray)):
                    # Handle nested lists/arrays
                    values = []
                    for item in prediction[:2]:  # Only take first 2 items
                        if isinstance(item, (list, np.ndarray)) and len(item) > 0:
                            # Extract first value from each nested item
                            scalar_val = self.safe_extract_scalar(item)
                            values.append(scalar_val)
                        else:
                            scalar_val = self.safe_extract_scalar(item)
                            values.append(scalar_val)
                    return values
                else:
                    # Simple list of values
                    return [self.safe_extract_scalar(x) for x in prediction[:2]]
            
            # If it's a TensorFlow tensor or similar
            if hasattr(prediction, 'numpy'):
                try:
                    numpy_pred = prediction.numpy()
                    return self.handle_inhomogeneous_prediction(numpy_pred)
                except:
                    pass
            
            # If it's a NumPy array, try to extract values manually
            if isinstance(prediction, np.ndarray):
                try:
                    # Use flat indexing to avoid shape issues
                    if prediction.size >= 2:
                        val1 = self.safe_extract_scalar(prediction.flat[0])
                        val2 = self.safe_extract_scalar(prediction.flat[1])
                        return [val1, val2]
                    elif prediction.size == 1:
                        val = self.safe_extract_scalar(prediction.flat[0])
                        return [1.0 - val, val]  # Assume single output is drowsy probability
                    else:
                        return [0.5, 0.5]
                except:
                    return [0.5, 0.5]
            
            # Default fallback
            return [0.5, 0.5]
            
        except Exception:
            return [0.5, 0.5]

    def parse_prediction_output(self, prediction):
        """Parse model prediction output with enhanced error handling"""
        try:
            # Handle None or empty predictions
            if prediction is None:
                return 0.5, 0.5, "None prediction"
            
            # Use the inhomogeneous handler for complex predictions
            processed_values = self.handle_inhomogeneous_prediction(prediction)
            
            if len(processed_values) >= 2:
                awake_prob = max(0.0, min(1.0, processed_values[0]))  # Clamp to [0,1]
                drowsy_prob = max(0.0, min(1.0, processed_values[1]))  # Clamp to [0,1]
                
                # Normalize if values don't sum to 1
                total = awake_prob + drowsy_prob
                if total > 0:
                    awake_prob /= total
                    drowsy_prob /= total
                else:
                    awake_prob, drowsy_prob = 0.5, 0.5
                    
                return awake_prob, drowsy_prob, "Binary classification"
                
            elif len(processed_values) == 1:
                drowsy_prob = max(0.0, min(1.0, processed_values[0]))
                awake_prob = 1.0 - drowsy_prob
                return awake_prob, drowsy_prob, "Single output"
            else:
                return 0.5, 0.5, "No valid values"
                
        except Exception:
            self.model_error_count += 1
            
            # Disable model if too many errors
            if self.model_error_count >= self.max_model_errors:
                print(f"❌ Disabling AI model after {self.model_error_count} errors. Using fallback detection.")
                self.model_disabled = True
                self.update_status_label()
                
            return 0.5, 0.5, "Parsing error"

    def load_with_custom_objects(self, model_path):
        """Load model with custom objects for compatibility"""
        from tensorflow.keras.metrics import MeanSquaredError
        
        custom_objects = {
            'mse': MeanSquaredError(),
            'keras.metrics.mse': MeanSquaredError(),
            'mean_squared_error': MeanSquaredError()
        }
        
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False
        )
        
        # Recompile with current Keras version
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def load_without_compilation(self, model_path):
        """Load model without compilation"""
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Manual compilation with compatible metrics
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def load_trained_model(self):
        model_path = r'C:\Users\sarva\Emotion_detection-main\drowsiness_detection\drowsiness_model.h5'
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            print("Using fallback EAR-based detection")
            self.model = None
            return
        
        print(f"🔄 Attempting to load model: {model_path}")
        
        # Try multiple loading strategies
        loading_strategies = [
            ("Custom Objects", self.load_with_custom_objects),
            ("Without Compilation", self.load_without_compilation)
        ]
        
        for strategy_name, strategy in loading_strategies:
            try:
                print(f"🔄 Trying strategy: {strategy_name}")
                self.model = strategy(model_path)
                
                if self.model is not None:
                    print(f"✅ Model loaded successfully using {strategy_name}!")
                    print("Model input shape:", self.model.input_shape)
                    print("Model output shape:", self.model.output_shape)
                    return
                    
            except Exception as e:
                print(f"❌ Strategy '{strategy_name}' failed: {str(e)}")
                continue
        
        print("❌ All loading strategies failed. Using EAR-based fallback detection")
        self.model = None

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(2, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)

        title_label = ttk.Label(main_frame, text="Advanced Drowsiness Detection System",
                                font=('Arial', 18, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)

        if self.model_disabled:
            model_status = "❌ AI Model Disabled (Errors)"
        elif self.model:
            model_status = "✅ AI Model Loaded"
        else:
            model_status = "⚠️ Using Fallback Detection"
            
        self.status_label = ttk.Label(main_frame, text=model_status,
                                 font=('Arial', 10),
                                 foreground='red' if self.model_disabled else ('green' if self.model else 'orange'))
        self.status_label.grid(row=0, column=2, sticky=tk.E, padx=10)

        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, columnspan=3, pady=10)

        ttk.Button(control_frame, text="📁 Load Image",
                   command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="🎥 Load Video",
                   command=self.load_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="📷 Start Camera",
                   command=self.start_camera).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="⏹️ Stop Camera",
                   command=self.stop_camera).pack(side=tk.LEFT, padx=5)

        # Add debug button for testing face detection
        ttk.Button(control_frame, text="🔍 Test Detection",
                   command=self.test_detection).pack(side=tk.LEFT, padx=5)

        preview_frame = ttk.LabelFrame(main_frame, text="Live Preview", padding="10")
        preview_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.preview_label = ttk.Label(preview_frame, text="No image/video loaded",
                                       font=('Arial', 12))
        self.preview_label.pack(expand=True)

        results_frame = ttk.LabelFrame(main_frame, text="Detection Results", padding="10")
        results_frame.grid(row=2, column=2, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))

        results_scroll_frame = ttk.Frame(results_frame)
        results_scroll_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(results_scroll_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.results_text = tk.Text(results_scroll_frame, width=45, height=25,
                                   yscrollcommand=scrollbar.set, font=('Courier', 10))
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.results_text.yview)

        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Load an image, video, or start camera")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var,
                               relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(5, 0))

    def test_detection(self):
        """Test face detection with current camera"""
        try:
            if not self.video_capture or not self.video_capture.isOpened():
                # Try to open camera for testing
                test_cap = cv2.VideoCapture(0)
                if not test_cap.isOpened():
                    messagebox.showwarning("Warning", "Cannot access camera for testing")
                    return
                ret, frame = test_cap.read()
                test_cap.release()
            else:
                ret, frame = self.video_capture.read()
            
            if not ret or frame is None:
                messagebox.showwarning("Warning", "Cannot capture frame for testing")
                return
                
            # Test different detection parameters
            self.test_face_detection_parameters(frame)
            
        except Exception as e:
            messagebox.showerror("Error", f"Detection test failed: {str(e)}")

    def test_face_detection_parameters(self, frame):
        """Test different face detection parameters to find optimal settings"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Test different parameter combinations
        test_configs = [
            {"scale": 1.1, "neighbors": 3, "min_size": (30, 30), "name": "Sensitive"},
            {"scale": 1.1, "neighbors": 5, "min_size": (50, 50), "name": "Balanced"},
            {"scale": 1.05, "neighbors": 6, "min_size": (40, 40), "name": "Accurate"},
            {"scale": 1.2, "neighbors": 4, "min_size": (60, 60), "name": "Fast"}
        ]
        
        debug_text = "🔍 FACE DETECTION TEST RESULTS:\n"
        debug_text += "=" * 40 + "\n\n"
        
        best_config = None
        max_faces = 0
        
        for config in test_configs:
            try:
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=config["scale"],
                    minNeighbors=config["neighbors"],
                    minSize=config["min_size"],
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                face_count = len(faces)
                debug_text += f"{config['name']} Settings:\n"
                debug_text += f"  Scale: {config['scale']}, Neighbors: {config['neighbors']}\n"
                debug_text += f"  Min Size: {config['min_size']}\n"
                debug_text += f"  Faces Detected: {face_count}\n\n"
                
                if face_count > max_faces:
                    max_faces = face_count
                    best_config = config
                    
            except Exception as e:
                debug_text += f"{config['name']}: ERROR - {str(e)}\n\n"
        
        if best_config:
            debug_text += f"🎯 RECOMMENDED SETTINGS: {best_config['name']}\n"
            debug_text += f"   Use these parameters for better detection\n"
        else:
            debug_text += "❌ NO OPTIMAL SETTINGS FOUND\n"
            debug_text += "   Try adjusting lighting or camera position\n"
        
        # Display results in the results text area
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, debug_text)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )

        if file_path:
            try:
                image = cv2.imread(file_path)
                if image is None:
                    raise ValueError("Could not load image file")

                results = self.detect_drowsiness(image)

                self.display_image(results['processed_frame'])
                self.display_results(results)

                self.status_var.set(f"✅ Analyzed: {file_path.split('/')[-1]}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                self.status_var.set("❌ Failed to load image")

    def load_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv")]
        )

        if file_path:
            try:
                self.video_capture = cv2.VideoCapture(file_path)
                if not self.video_capture.isOpened():
                    raise ValueError("Could not open video file")

                self.is_recording = True
                self.process_video()
                self.status_var.set(f"🎥 Processing: {file_path.split('/')[-1]}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load video: {str(e)}")
                self.status_var.set("❌ Failed to load video")

    def start_camera(self):
        try:
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened():
                raise ValueError("Could not access camera")

            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.video_capture.set(cv2.CAP_PROP_FPS, 30)

            self.is_recording = True
            self.process_video()
            self.status_var.set("📷 Camera active - Real-time detection")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
            self.status_var.set("❌ Camera access failed")

    def stop_camera(self):
        self.is_recording = False
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None

        self.face_tracker = FaceTracker(min_consistent_frames=3)
        self.drowsy_frame_counters = {}
        self.last_detection_time = 0

        self.status_var.set("⏹️ Stopped - Ready for new input")

    def process_video(self):
        def video_thread():
            frame_count = 0
            last_process_time = 0
            
            while self.is_recording and self.video_capture:
                ret, frame = self.video_capture.read()
                if not ret:
                    break

                frame_count += 1
                current_time = time.time()
                
                # Process frames more frequently for better responsiveness
                if current_time - last_process_time >= 0.2:  # Process every 200ms
                    last_process_time = current_time
                    
                    results = self.detect_drowsiness(frame)
                    self.root.after(0, lambda r=results: self.display_image(r['processed_frame']))
                    self.root.after(0, lambda r=results: self.display_results(r))

                    if results['drowsy_count'] > 0:
                        self.root.after(0, lambda r=results: self.show_drowsiness_alert(r))

                time.sleep(0.033)  # ~30 FPS
            
            self.root.after(0, self.stop_camera)

        thread = threading.Thread(target=video_thread, daemon=True)
        thread.start()

    def preprocess_face_for_model(self, face_roi):
        try:
            # Ensure face_roi is valid
            if face_roi.size == 0:
                return None
                
            face_resized = cv2.resize(face_roi, (224, 224))
            face_normalized = face_resized.astype('float32') / 255.0
            face_batch = np.expand_dims(face_normalized, axis=0)
            return face_batch
        except Exception as e:
            print(f"Error preprocessing face: {e}")
            return None

    def detect_drowsiness(self, frame):
        current_time = time.time()
        
        # Throttle detection but allow more frequent updates
        if current_time - self.last_detection_time < self.detection_interval:
            return {
                'total_people': 0,
                'drowsy_count': 0,
                'awake_count': 0,
                'drowsy_people': [],
                'processed_frame': frame.copy()
            }
        
        self.last_detection_time = current_time
        
        results = {
            'total_people': 0,
            'drowsy_count': 0,
            'awake_count': 0,
            'drowsy_people': [],
            'processed_frame': frame.copy()
        }

        # Optimized face detection parameters - balanced for accuracy and sensitivity
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # More sensitive parameters to ensure face detection
        detected_faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,       # Moderate scaling
            minNeighbors=4,        # Reduced for more sensitivity
            minSize=(40, 40),      # Reasonable minimum size
            maxSize=(300, 300),    # Reasonable maximum size
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        print(f"Raw faces detected: {len(detected_faces)}")  # Debug print

        # Use face tracking but with lower requirements
        validated_faces = self.face_tracker.update_faces(detected_faces)
        results['total_people'] = len(validated_faces)
        
        print(f"Validated faces: {len(validated_faces)}")  # Debug print

        for i, (x, y, w, h) in enumerate(validated_faces):
            face_roi = frame[y:y + h, x:x + w]

            is_drowsy = False
            confidence = 0.0
            method_used = "Unknown"

            # Try AI model first (if not disabled)
            if self.model is not None and not self.model_disabled:
                try:
                    preprocessed_face = self.preprocess_face_for_model(face_roi)
                    if preprocessed_face is not None:
                        raw_prediction = self.model.predict(preprocessed_face, verbose=0)
                        
                        # Use enhanced prediction parsing
                        awake_prob, drowsy_prob, parse_method = self.parse_prediction_output(raw_prediction)
                        
                        confidence = drowsy_prob
                        is_drowsy = drowsy_prob > awake_prob and drowsy_prob > 0.6  # Moderate threshold
                        method_used = f"AI - {parse_method} (A:{awake_prob:.2f}, D:{drowsy_prob:.2f})"
                            
                except Exception as e:
                    print(f"Model prediction error: {e}")
                    method_used = "Model Error - Using Fallback"
                    self.model_error_count += 1
                    
                    if self.model_error_count >= self.max_model_errors:
                        self.model_disabled = True
                        self.update_status_label()

            # Fallback to EAR method
            if method_used == "Unknown" or "Error" in method_used:
                if hasattr(self, 'ear_calculator') and self.ear_calculator.predictor is not None:
                    left_ear, right_ear = self.ear_calculator.get_eye_aspect_ratios(gray, (x, y, w, h))
                    if left_ear is not None and right_ear is not None:
                        avg_ear = (left_ear + right_ear) / 2.0
                        face_key = f"face_{x//10*10}_{y//10*10}"  # Quantize coordinates
                        
                        if avg_ear < self.EAR_THRESHOLD:
                            if face_key not in self.drowsy_frame_counters:
                                self.drowsy_frame_counters[face_key] = 0
                            self.drowsy_frame_counters[face_key] += 1
                        else:
                            self.drowsy_frame_counters[face_key] = 0

                        is_drowsy = (face_key in self.drowsy_frame_counters and
                                     self.drowsy_frame_counters[face_key] >= self.CONSECUTIVE_FRAMES_THRESHOLD)
                        confidence = avg_ear
                        method_used = f"EAR ({avg_ear:.3f})"
                    else:
                        method_used = "No landmarks"
                else:
                    # Final fallback - eye count detection
                    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                    face_roi_gray = gray[y:y + h, x:x + w]
                    eyes = eye_cascade.detectMultiScale(face_roi_gray, minNeighbors=3)
                    is_drowsy = len(eyes) < 2
                    confidence = len(eyes) / 2.0
                    method_used = f"Eye Count ({len(eyes)})"

            # Draw results
            if is_drowsy:
                cv2.rectangle(results['processed_frame'], (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.putText(results['processed_frame'], 'DROWSY',
                            (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(results['processed_frame'], method_used[:25],
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                results['drowsy_people'].append({
                    'person_id': i + 1,
                    'confidence': confidence,
                    'method': method_used,
                    'bbox': (x, y, w, h)
                })
                results['drowsy_count'] += 1
            else:
                cv2.rectangle(results['processed_frame'], (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(results['processed_frame'], 'AWAKE',
                            (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(results['processed_frame'], method_used[:25],
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                results['awake_count'] += 1

        return results

    def update_status_label(self):
        """Update the model status label"""
        if self.model_disabled:
            status_text = "❌ AI Model Disabled (Errors)"
            color = 'red'
        elif self.model:
            status_text = "✅ AI Model Loaded"
            color = 'green'
        else:
            status_text = "⚠️ Using Fallback Detection"
            color = 'orange'
            
        if hasattr(self, 'status_label'):
            self.status_label.config(text=status_text, foreground=color)

    def display_image(self, image):
        try:
            height, width = image.shape[:2]
            max_width, max_height = 700, 500

            if width > max_width or height > max_height:
                scale = min(max_width / width, max_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            photo = ImageTk.PhotoImage(pil_image)

            self.preview_label.config(image=photo, text="")
            self.preview_label.image = photo
        except Exception as e:
            print(f"Display error: {e}")

    def display_results(self, results):
        self.results_text.delete(1.0, tk.END)

        result_text = "=" * 40 + "\n"
        result_text += "  DROWSINESS DETECTION RESULTS\n"
        result_text += "=" * 40 + "\n\n"

        result_text += f"📊 SUMMARY:\n"
        result_text += f"   Total People: {results['total_people']}\n"
        result_text += f"   😴 Drowsy: {results['drowsy_count']}\n"
        result_text += f"   👁️ Awake: {results['awake_count']}\n"
        
        if self.model_disabled:
            result_text += f"   ❌ Model Errors: {self.model_error_count}\n"
        
        result_text += "\n"

        if results['drowsy_count'] > 0:
            result_text += "🚨 ALERT STATUS: DROWSINESS DETECTED!\n\n"
        else:
            result_text += "✅ STATUS: All Clear\n\n"

        if results['drowsy_people']:
            result_text += "🔍 DETAILED ANALYSIS:\n"
            result_text += "-" * 35 + "\n"
            for person in results['drowsy_people']:
                result_text += f"Person {person['person_id']}:\n"
                result_text += f"   State: DROWSY 😴\n"
                result_text += f"   Method: {person['method'][:40]}\n"
                result_text += f"   Confidence: {person['confidence']:.3f}\n"
                result_text += "-" * 35 + "\n"

        result_text += f"\n⏰ Last Update: {time.strftime('%H:%M:%S')}\n"

        self.results_text.insert(tk.END, result_text)
        self.results_text.see(tk.END)

    def show_drowsiness_alert(self, results):
        if results['drowsy_count'] > 0:
            # Prevent multiple alert windows
            if hasattr(self, 'alert_window') and hasattr(self.alert_window, 'winfo_exists') and self.alert_window.winfo_exists():
                return

            alert_message = f"🚨 DROWSINESS ALERT! 🚨\n\n"
            alert_message += f"Detected {results['drowsy_count']} drowsy person(s)\n"
            alert_message += f"Total people monitored: {results['total_people']}\n\n"
            alert_message += "⚠️ IMMEDIATE ATTENTION REQUIRED ⚠️"

            self.alert_window = tk.Toplevel(self.root)
            self.alert_window.title("🚨 DROWSINESS ALERT")
            self.alert_window.geometry("400x200")
            self.alert_window.configure(bg='#ff4444')
            self.alert_window.attributes('-topmost', True)

            alert_label = tk.Label(self.alert_window, text=alert_message,
                                   fg='white', bg='#ff4444',
                                   font=('Arial', 12, 'bold'),
                                   justify=tk.CENTER)
            alert_label.pack(expand=True, pady=20)

            button_frame = tk.Frame(self.alert_window, bg='#ff4444')
            button_frame.pack(pady=10)

            tk.Button(button_frame, text="✓ ACKNOWLEDGED",
                      command=self.alert_window.destroy,
                      bg='white', fg='black',
                      font=('Arial', 10, 'bold'),
                      padx=20).pack()

            # Auto-close after 3 seconds
            self.alert_window.after(3000, lambda: self.alert_window.destroy()
                                    if hasattr(self.alert_window, 'winfo_exists') and self.alert_window.winfo_exists() else None)

def main():
    try:
        root = tk.Tk()
        app = DrowsinessDetectionGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Application error: {e}")
        messagebox.showerror("Application Error", f"Failed to start application: {str(e)}")

if __name__ == "__main__":
    main()

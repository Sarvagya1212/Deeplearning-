import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
import threading
import time
from datetime import datetime
import mediapipe as mp

class SignLanguageDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Detection System")
        self.root.geometry("1200x800")
        
        # Initialize all our variables
        self.model = None
        self.video_capture = None
        self.is_recording = False
        self.current_frame = None
        self.model_input_count = 1  # assume single input model by default
        
        # Add prediction smoothing for stability
        self.prediction_history = []
        self.max_history = 5
        
        # Define our sign language classes
        self.asl_letters = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
        ]
        
        # Define minimum confidence threshold to prevent random predictions
        self.min_confidence = 0.65  # only accept predictions above 65%
        
        # Try to load our trained model
        self.load_model()
        
        # Set up MediaPipe for hand detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7,  
            min_tracking_confidence=0.7
        )
        
        # Build the user interface
        self.setup_ui()
    
    def load_model(self):
        """Load our trained model and check its input requirements"""
        try:
            model_path = r'C:\Users\sarva\Emotion_detection-main\sign_language_detection\sign_language_model.h5'
            self.model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully")
            
            # Check how many inputs the model expects
            self.model_input_count = len(self.model.inputs)
            print(f"Model expects {self.model_input_count} input(s)")
            
            # Show input details for debugging
            for i, inp in enumerate(self.model.inputs):
                print(f"  Input {i}: shape {inp.shape}, name {inp.name}")
                
        except Exception as e:
            print(f"Failed to load model: {e}")
            messagebox.showerror("Model Error", f"Could not load model: {str(e)}")
            self.model = None
    
    def setup_ui(self):
        """Create the main user interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights so things resize properly
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(4, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Title
        title = ttk.Label(main_frame, text="Sign Language Detection System", 
                         font=("Arial", 18, "bold"))
        title.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        # Show model status
        model_status = "Model: Loaded" if self.model else "Model: Not Loaded"
        if self.model:
            model_status += f" ({self.model_input_count} input{'s' if self.model_input_count > 1 else ''})"
        
        status_label = ttk.Label(main_frame, text=model_status)
        status_label.grid(row=1, column=0, columnspan=3, pady=(0, 5))
        
        # Current time display
        self.time_label = ttk.Label(main_frame, font=("Arial", 10))
        self.time_label.grid(row=2, column=0, columnspan=3, pady=(0, 10))
        self.update_time_display()
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=(0, 10))
        
        ttk.Button(button_frame, text="Upload Image", 
                  command=self.upload_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Start Real-time", 
                  command=self.start_realtime).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Stop Real-time", 
                  command=self.stop_realtime).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear Results", 
                  command=self.clear_results).pack(side=tk.LEFT, padx=5)
        
        # Main content area
        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=4, column=0, columnspan=3, pady=(0, 10), 
                          sticky=(tk.W, tk.E, tk.N, tk.S))
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_columnconfigure(1, weight=1)
        
        # Image/video preview area
        preview_frame = ttk.LabelFrame(content_frame, text="Camera/Image Preview", 
                                     padding="10")
        preview_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), 
                          padx=(0, 5))
        
        self.preview_label = ttk.Label(preview_frame, text="No image or video feed")
        self.preview_label.pack(expand=True)
        
        # Results display area
        results_frame = ttk.LabelFrame(content_frame, text="Detection Results", 
                                     padding="10")
        results_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), 
                          padx=(5, 0))
        
        # Current prediction display
        ttk.Label(results_frame, text="Current Prediction:", 
                 font=("Arial", 10, "bold")).pack(pady=(0, 5))
        self.prediction_label = ttk.Label(results_frame, text="None", 
                                        font=("Arial", 24, "bold"), 
                                        foreground="blue")
        self.prediction_label.pack(pady=(0, 5))
        self.confidence_label = ttk.Label(results_frame, text="Confidence: 0%")
        self.confidence_label.pack(pady=(0, 10))
        
        # History log
        ttk.Label(results_frame, text="Detection History:", 
                 font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        # Create text area with scrollbar
        text_frame = ttk.Frame(results_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.history_text = tk.Text(text_frame, height=12, width=40, 
                                   font=("Consolas", 9))
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", 
                                 command=self.history_text.yview)
        self.history_text.configure(yscrollcommand=scrollbar.set)
        
        self.history_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Status bar at bottom
        self.status_var = tk.StringVar()
        self.status_var.set("Ready" if self.model else "Model not loaded")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief="sunken")
        status_bar.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E))
    
    def update_time_display(self):
        """Update the time display and operational status"""
        current_time = datetime.now().strftime("%H:%M:%S")
        operational = self.is_operational_time()
        
        status_text = "OPERATIONAL" if operational else "NOT OPERATIONAL"
        color = "green" if operational else "red"
        
        time_text = f"Time: {current_time} - Status: {status_text} (5 AM - 10 PM only)"
        self.time_label.config(text=time_text, foreground=color)
        
        # Schedule next update
        self.root.after(1000, self.update_time_display)
    
    def is_operational_time(self):
        """Check if we're within operational hours (5 AM to 10 PM)"""
        current_time = datetime.now().time()
        start_time = datetime.strptime("17:00", "%H:%M").time()
        end_time = datetime.strptime("22:00", "%H:%M").time()
        return start_time <= current_time <= end_time
    
    def validate_hand_landmarks(self, landmarks):
        """Validate if detected landmarks represent a reasonable hand"""
        try:
            # Reshape to (21, 2) format for validation
            points = landmarks.reshape(-1, 2)
            
            # Check if landmarks are within reasonable bounds (0 to 1)
            if np.any(points < 0) or np.any(points > 1):
                return False
            
            # Check if hand has reasonable spread - not all points clustered
            x_spread = np.max(points[:, 0]) - np.min(points[:, 0])
            y_spread = np.max(points[:, 1]) - np.min(points[:, 1])
            
            # Hand should cover at least 8% of image width/height to be valid
            if x_spread < 0.08 or y_spread < 0.08:
                print("Hand appears too small or clustered")
                return False
            
            # Check for reasonable hand orientation and finger positions
            # Landmark 0 is wrist, landmarks 4,8,12,16,20 are fingertips
            wrist = points[0]
            fingertips = points[[4, 8, 12, 16, 20]]
            
            # Fingertips should not all be at the same position as wrist
            distances = np.linalg.norm(fingertips - wrist, axis=1)
            if np.all(distances < 0.05):  # Too close to wrist
                print("Hand landmarks appear invalid - fingers too close to wrist")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error validating landmarks: {e}")
            return False
    
    def extract_hand_landmarks(self, image):
        """Extract hand landmarks from image using MediaPipe with validation"""
        try:
            # Make sure image is in the right format
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = np.clip(image, 0, 255).astype(np.uint8)
            
            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.hands.process(image_rgb)
            
            if results.multi_hand_landmarks:
                # Extract landmark coordinates
                landmarks = []
                for lm in results.multi_hand_landmarks[0].landmark:
                    landmarks.extend([lm.x, lm.y])
                
                landmarks_array = np.array(landmarks, dtype=np.float32)
                
                # Validate landmarks before returning
                if self.validate_hand_landmarks(landmarks_array):
                    return landmarks_array
                else:
                    print("Hand landmarks failed validation")
                    return None
            else:
                return None
                
        except Exception as e:
            print(f"Error extracting landmarks: {e}")
            return None
    
    def smooth_predictions(self, current_result):
        """Smooth predictions using recent history to avoid flickering"""
        # Add current result to history
        self.prediction_history.append(current_result)
        
        # Keep only recent predictions
        if len(self.prediction_history) > self.max_history:
            self.prediction_history.pop(0)
        
        # Filter out invalid predictions
        valid_predictions = [p for p in self.prediction_history 
                           if p['prediction'] not in ['No Hand Detected', 'Low Confidence', 'Not Operational']]
        
        if len(valid_predictions) >= 3:
            # Get most common prediction from valid ones
            predictions = [p['prediction'] for p in valid_predictions]
            most_common = max(set(predictions), key=predictions.count)
            
            # Return the one with highest confidence for this prediction
            best_match = max([p for p in valid_predictions if p['prediction'] == most_common], 
                           key=lambda x: x['confidence'])
            return best_match
        
        return current_result
    
    def predict_sign(self, image):
        """Make a prediction on the given image with proper validation"""
        # Check if we're in operational hours
        if not self.is_operational_time():
            return {
                "prediction": "Not Operational",
                "confidence": 0.0,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
        
        # Make sure we have a model loaded
        if self.model is None:
            return {
                "prediction": "Model Not Loaded",
                "confidence": 0.0,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
        
        try:
            # For multi-input models, check hand presence first
            if self.model_input_count == 2:
                landmarks = self.extract_hand_landmarks(image)
                
                # Critical fix: Don't predict if no valid hand detected
                if landmarks is None:
                    return {
                        "prediction": "No Hand Detected",
                        "confidence": 0.0,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    }
            
            # Prepare image input
            image_resized = cv2.resize(image, (224, 224))
            image_normalized = image_resized.astype(np.float32) / 255.0
            image_input = np.expand_dims(image_normalized, axis=0)
            
            # Handle different model types
            if self.model_input_count == 1:
                # Single input model - just use the image
                predictions = self.model.predict(image_input, verbose=0)
                
            elif self.model_input_count == 2:
                # Multi-input model - use both image and landmarks
                landmarks_input = np.expand_dims(landmarks, axis=0)
                predictions = self.model.predict([image_input, landmarks_input], verbose=0)
                
            else:
                return {
                    "prediction": f"Unsupported Model ({self.model_input_count} inputs)",
                    "confidence": 0.0,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                }
            
            # Get the best prediction
            class_index = int(np.argmax(predictions))
            confidence = float(np.max(predictions))
            
            if confidence < self.min_confidence:
                return {
                    "prediction": "Low Confidence",
                    "confidence": confidence,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                }
            
            # Make sure the class index is valid
            if class_index < len(self.asl_letters):
                predicted_letter = self.asl_letters[class_index]
            else:
                predicted_letter = f"Unknown Class {class_index}"
            
            return {
                "prediction": predicted_letter,
                "confidence": confidence,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                "prediction": "Prediction Error",
                "confidence": 0.0,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
    
    def upload_image(self):
        """Handle image upload and processing"""
        if not self.is_operational_time():
            messagebox.showwarning("Not Operational", 
                                 "Detection only available from 5 AM to 10 PM")
            return
        
        # Let user select an image file
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Load the image
                image = cv2.imread(file_path)
                if image is None:
                    messagebox.showerror("Error", "Could not load the selected image")
                    return
                
                # Display it
                self.display_image(image)
                
                # Make prediction
                result = self.predict_sign(image)
                self.display_result(result)
                
                self.status_var.set(f"Analyzed: {file_path.split('/')[-1]}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process image: {str(e)}")
    
    def start_realtime(self):
        """Start real-time camera detection"""
        if not self.is_operational_time():
            messagebox.showwarning("Not Operational", 
                                 "Detection only available from 5 AM to 10 PM")
            return
        
        if self.model is None:
            messagebox.showerror("Error", "No model loaded")
            return
        
        try:
            # Try to open the camera
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened():
                messagebox.showerror("Error", "Could not access camera")
                return
            
            self.is_recording = True
            self.process_video_stream()
            self.status_var.set("Real-time detection active")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
    
    def process_video_stream(self):
        """Process video frames in a separate thread with smoothing"""
        def video_loop():
            frame_count = 0
            while self.is_recording and self.video_capture:
                # Check if we're still in operational hours
                if not self.is_operational_time():
                    self.stop_realtime()
                    messagebox.showwarning("Time Limit", 
                                         "Detection stopped - outside operational hours")
                    break
                
                # Read frame from camera
                ret, frame = self.video_capture.read()
                if not ret:
                    break
                
                # Display the frame
                self.display_image(frame)
                
                # Only process every 5th frame to improve stability
                if frame_count % 5 == 0:
                    result = self.predict_sign(frame)
                    # Apply smoothing to reduce flickering
                    smoothed_result = self.smooth_predictions(result)
                    self.display_result(smoothed_result)
                
                frame_count += 1
                time.sleep(0.1)  # Reasonable delay to prevent overwhelming
            
            # Clean up when done
            self.stop_realtime()
        
        # Start the video processing in a separate thread
        thread = threading.Thread(target=video_loop, daemon=True)
        thread.start()
    
    def stop_realtime(self):
        """Stop real-time detection"""
        self.is_recording = False
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        self.prediction_history.clear()  # Clear history when stopping
        self.status_var.set("Real-time detection stopped")
    
    def display_image(self, image):
        """Display image in the preview area"""
        try:
            # Resize for display
            height, width = image.shape[:2]
            max_width, max_height = 450, 350
            
            if width > max_width or height > max_height:
                scale = min(max_width / width, max_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image_resized = cv2.resize(image, (new_width, new_height))
            else:
                image_resized = image
            
            # Convert to format Tkinter can use
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            image_tk = ImageTk.PhotoImage(image_pil)
            
            # Update the display
            self.preview_label.config(image=image_tk, text="")
            self.preview_label.image = image_tk  # keep a reference
            
        except Exception as e:
            print(f"Error displaying image: {e}")
    
    def display_result(self, result):
        """Display prediction results with proper handling of edge cases"""
        try:
            prediction = result["prediction"]
            confidence = result["confidence"]
            
            # Handle different prediction types
            if prediction == "Not Operational":
                self.prediction_label.config(text=prediction, foreground="red")
                self.confidence_label.config(text="Available 5 AM - 10 PM")
            elif prediction in ["Model Not Loaded", "Prediction Error"]:
                self.prediction_label.config(text=prediction, foreground="red")
                self.confidence_label.config(text="Check system status")
            elif prediction == "No Hand Detected":
                self.prediction_label.config(text="Position Hand in View", foreground="orange")
                self.confidence_label.config(text="No hand landmarks found")
            elif prediction == "Low Confidence":
                self.prediction_label.config(text="Unclear Gesture", foreground="orange")
                self.confidence_label.config(text=f"Confidence: {confidence:.1%} (too low)")
            else:
                # Valid prediction - color based on confidence
                if confidence > 0.85:
                    color = "green"
                elif confidence > 0.7:
                    color = "blue"
                else:
                    color = "orange"
                
                self.prediction_label.config(text=prediction, foreground=color)
                self.confidence_label.config(text=f"Confidence: {confidence:.1%}")
            
            # Only add meaningful predictions to history
            if prediction not in ["Not Operational", "Model Not Loaded", "Prediction Error", 
                                "No Hand Detected", "Low Confidence"]:
                timestamp = result["timestamp"]
                history_line = f"[{timestamp}] {prediction} ({confidence:.1%})\n"
                self.history_text.insert(tk.END, history_line)
                self.history_text.see(tk.END)
                
                # Keep history reasonable length
                lines = self.history_text.get("1.0", tk.END).split("\n")
                if len(lines) > 50:
                    self.history_text.delete("1.0", f"{len(lines)-50}.0")
            
        except Exception as e:
            print(f"Error displaying result: {e}")
    
    def clear_results(self):
        """Clear all results and reset display"""
        self.prediction_label.config(text="None", foreground="blue")
        self.confidence_label.config(text="Confidence: 0%")
        self.history_text.delete(1.0, tk.END)
        self.prediction_history.clear()  # Clear prediction history
        
        if not self.is_recording:
            self.preview_label.config(image="", text="No image or video feed")
            self.preview_label.image = None
        
        self.status_var.set("Results cleared")

def main():
    """Main function to start the application"""
    try:
        print("Starting Sign Language Detection System")
        root = tk.Tk()
        app = SignLanguageDetectionGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Application error: {e}")

if __name__ == "__main__":
    main()

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import time
import os

# Enable optimizations at startup
tf.config.optimizer.set_jit(True)  # Enable XLA compilation
mixed_precision.set_global_policy('mixed_float16')  # Enable mixed precision

class OptimizedCarColorDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Optimized Car Color Detection System")
        self.root.geometry("1200x800")
        
        # Variables
        self.color_model = None
        self.yolo_model = None
        self.current_image = None
        self.processing = False
        
        # Car colors - ensure this matches your trained model classes
        self.car_colors = ['beige', 'black', 'blue', 'brown','gold', 'gray', 'green', 'orange', 'pink', 'purple', 'red', 'silver', 'tan', 'white', 'yellow']
        
        # Model paths
        self.model_paths = {
            'color': r'C:\Users\sarva\Emotion_detection-main\car_color_detection\car_color_model_synchronized.keras', 
            'yolo': r'C:\Users\sarva\Emotion_detection-main\car_color_detection\yolov8n.pt'
        }
        
        self.setup_ui()
        
        # Load models asynchronously to avoid blocking GUI
        self.load_models_async()
    
    def load_models_async(self):
        """Load models in background thread"""
        def load_task():
            self.update_status("Loading models...")
            
            try:
                # Load color classification model
                if os.path.exists(self.model_paths['color']):
                    print("Loading color model...")
                    self.color_model = tf.keras.models.load_model(self.model_paths['color'])
                    
                    self.color_model.compile(
                        optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    
                    print(f" Color model loaded successfully!")
                    print(f"   Input shape: {self.color_model.input_shape}")
                    print(f"   Output classes: {self.color_model.output_shape[-1]}")
                    
                    # Verify model output matches expected classes
                    if self.color_model.output_shape[-1] != len(self.car_colors):
                        print(f" Warning: Model outputs {self.color_model.output_shape[-1]} classes but {len(self.car_colors)} expected")
                else:
                    print(" Color model file not found")
                
                # Load YOLO model
                if os.path.exists(self.model_paths['yolo']):
                    print("Loading YOLO model...")
                    self.yolo_model = YOLO(self.model_paths['yolo'])
                    print(" YOLO model loaded successfully!")
                else:
                    print(" YOLO model file not found")
                
                # Update UI on main thread
                self.root.after(0, lambda: self.update_status("Models loaded successfully!" if self.models_loaded() else "Some models failed to load"))
                
            except Exception as e:
                print(f" Error loading models: {e}")
                self.root.after(0, lambda: self.update_status(f"Model loading failed: {str(e)}"))
        
        # Start loading in background
        threading.Thread(target=load_task, daemon=True).start()
    
    def models_loaded(self):
        """Check if models are properly loaded"""
        return self.color_model is not None and self.yolo_model is not None
    
    def setup_ui(self):
        """Setup optimized UI"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(2, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Optimized Car Color Detection & Traffic Analysis", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, columnspan=3, pady=10)
        
        ttk.Button(control_frame, text="Upload Image", 
                  command=self.upload_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Analyze Traffic", 
                  command=self.analyze_traffic_threaded).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Real-time Detection", 
                  command=self.start_camera).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Clear Results", 
                  command=self.clear_results).pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, mode='indeterminate')
        self.progress_bar.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        self.progress_bar.grid_remove()  # Hide initially
        
        # Content frame
        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=3, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(0, weight=2)
        content_frame.grid_columnconfigure(2, weight=1)
        
        # Image preview frame
        preview_frame = ttk.LabelFrame(content_frame, text="Image Preview", padding="10")
        preview_frame.grid(row=0, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.preview_label = ttk.Label(preview_frame, text="No image uploaded\n\nClick 'Upload Image' or 'Real-time Detection' to start")
        self.preview_label.pack(expand=True)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(content_frame, text="Traffic Statistics", padding="10")
        stats_frame.grid(row=0, column=2, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Statistics display
        self.stats_text = tk.Text(stats_frame, width=30, height=15, font=('Consolas', 10))
        stats_scrollbar = ttk.Scrollbar(stats_frame, orient="vertical", command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scrollbar.set)
        self.stats_text.pack(side="left", fill="both", expand=True)
        stats_scrollbar.pack(side="right", fill="y")
        
        # Detection details frame
        details_frame = ttk.LabelFrame(main_frame, text="Detection Details", padding="10")
        details_frame.grid(row=4, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.details_text = tk.Text(details_frame, height=8, font=('Consolas', 9))
        details_scrollbar = ttk.Scrollbar(details_frame, orient="vertical", command=self.details_text.yview)
        self.details_text.configure(yscrollcommand=details_scrollbar.set)
        self.details_text.pack(side="left", fill="both", expand=True)
        details_scrollbar.pack(side="right", fill="y")
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Initializing - Loading models in background...")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, font=('Arial', 9))
        status_bar.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E))
    
    def update_status(self, message):
        """Thread-safe status update"""
        self.status_var.set(message)
    
    def upload_image(self):
        """Upload and display an image"""
        file_path = filedialog.askopenfilename(
            title="Select Traffic Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Load and display image
                self.current_image = cv2.imread(file_path)
                if self.current_image is None:
                    messagebox.showerror("Error", "Could not load the selected image")
                    return
                
                self.display_image(self.current_image, "Original Image")
                self.update_status(f"Image loaded: {os.path.basename(file_path)}")
                
                # Clear previous results
                self.clear_text_widgets()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def analyze_traffic_threaded(self):
        """Analyze traffic in background thread"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please upload an image first")
            return
        
        if self.processing:
            messagebox.showinfo("Info", "Analysis already in progress")
            return
        
        if not self.models_loaded():
            messagebox.showerror("Error", "Models not loaded. Please wait for models to load or check model files.")
            return
        
        def analysis_task():
            self.processing = True
            self.root.after(0, self.show_progress)
            
            try:
                self.root.after(0, lambda: self.update_status("Detecting vehicles..."))
                
                # Real traffic analysis with loaded models
                results = self.analyze_traffic_real(self.current_image)
                
                # Update UI on main thread
                self.root.after(0, lambda: self.display_traffic_results(results))
                self.root.after(0, lambda: self.display_analyzed_image(results['annotated_image']))
                self.root.after(0, lambda: self.update_status(f"Analysis complete - Found {results['stats']['total_cars']} vehicles"))
                
            except Exception as e:
                error_msg = f"Analysis failed: {str(e)}"
                self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
                self.root.after(0, lambda: self.update_status("Analysis failed"))
            
            finally:
                self.processing = False
                self.root.after(0, self.hide_progress)
        
        # Start analysis in background
        threading.Thread(target=analysis_task, daemon=True).start()
    
    @tf.function  # Compile for faster execution
    def predict_car_color_optimized(self, car_image_tensor):
        """Optimized color prediction with TensorFlow function compilation"""
        predictions = self.color_model(car_image_tensor, training=False)
        return predictions
    
    def analyze_traffic_real(self, image):
        """Real traffic analysis using loaded models"""
        # Create annotated image copy
        annotated_image = image.copy()
        
        # YOLO detection for vehicles
        results = self.yolo_model(image, verbose=False)
        
        cars_detected = []
        people_detected = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.yolo_model.names[class_id]
                    
                    # Filter for cars and people
                    if class_name in ['car', 'truck', 'bus', 'motorcycle'] and confidence > 0.5:
                        # Extract car region for color prediction
                        car_region = image[y1:y2, x1:x2]
                        
                        if car_region.size > 0:
                            # Predict car color
                            predicted_color, color_confidence = self.predict_car_color(car_region)
                            
                            # Draw rectangle based on color
                            if predicted_color == 'blue':
                                rect_color = (0, 0, 255)  # Red for blue cars
                                label = f"Blue {class_name}: {color_confidence:.2f}"
                            else:
                                rect_color = (255, 0, 0)  # Blue for other cars
                                label = f"{predicted_color.title()} {class_name}: {color_confidence:.2f}"
                            
                            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), rect_color, 3)
                            cv2.putText(annotated_image, label, (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, rect_color, 2)
                            
                            cars_detected.append({
                                'bbox': (x1, y1, x2-x1, y2-y1),
                                'color': predicted_color,
                                'confidence': color_confidence,
                                'vehicle_type': class_name
                            })
                    
                    elif class_name == 'person' and confidence > 0.5:
                        # Green rectangle for people
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated_image, f"Person: {confidence:.2f}", (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        people_detected.append({
                            'bbox': (x1, y1, x2-x1, y2-y1)
                        })
        
        # Calculate statistics
        blue_cars = sum(1 for car in cars_detected if car['color'] == 'blue')
        other_cars = len(cars_detected) - blue_cars
        
        return {
            'annotated_image': annotated_image,
            'cars': cars_detected,
            'people': people_detected,
            'stats': {
                'total_cars': len(cars_detected),
                'blue_cars': blue_cars,
                'other_cars': other_cars,
                'total_people': len(people_detected)
            }
        }
    
    def predict_car_color(self, car_image):
        """Predict car color using trained model"""
        try:
            # Preprocess image for color model
            img_resized = cv2.resize(car_image, (224, 224))  # Adjust size based on your model
            img_normalized = img_resized.astype(np.float32) / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            # Convert to tensor for optimized prediction
            img_tensor = tf.constant(img_batch, dtype=tf.float32)
            
            # Predict color
            predictions = self.predict_car_color_optimized(img_tensor)
            predicted_idx = tf.argmax(predictions[0]).numpy()
            confidence = float(predictions[0][predicted_idx])
            
            # Map to color name
            if predicted_idx < len(self.car_colors):
                predicted_color = self.car_colors[predicted_idx]
            else:
                predicted_color = "unknown"
            
            return predicted_color, confidence
            
        except Exception as e:
            print(f"Color prediction error: {e}")
            return "unknown", 0.0
    
    def start_camera(self):
        """Start real-time camera detection"""
        messagebox.showinfo("Camera Mode", "Real-time camera detection will be implemented in the next version")
        # TODO: Implement real-time camera processing
    
    def display_analyzed_image(self, image):
        """Display the analyzed image with annotations"""
        self.display_image(image, "Traffic Analysis Results")
    
    def display_image(self, image, title="Image"):
        """Optimized image display"""
        try:
            # Efficient resizing
            height, width = image.shape[:2]
            max_width, max_height = 600, 400
            
            if width > max_width or height > max_height:
                scale = min(max_width/width, max_height/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image_resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            else:
                image_resized = image
            
            # Convert color space efficiently
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update label
            self.preview_label.config(image=photo, text="")
            self.preview_label.image = photo  # Keep reference
            
        except Exception as e:
            print(f"Error displaying image: {e}")
    
    def display_traffic_results(self, results):
        """Display comprehensive traffic analysis results"""
        # Clear previous results
        self.clear_text_widgets()
        
        # Display statistics
        stats = results['stats']
        stats_text = "TRAFFIC ANALYSIS RESULTS\n"
        stats_text += "=" * 25 + "\n\n"
        stats_text += f" Total Vehicles: {stats['total_cars']}\n"
        stats_text += f" Blue Cars: {stats['blue_cars']}\n"
        stats_text += f" Other Cars: {stats['other_cars']}\n"
        stats_text += f" People: {stats['total_people']}\n\n"
        
        # Color breakdown
        if results['cars']:
            stats_text += "COLOR BREAKDOWN:\n"
            color_counts = {}
            vehicle_types = {}
            
            for car in results['cars']:
                color = car['color']
                vehicle_type = car.get('vehicle_type', 'car')
                color_counts[color] = color_counts.get(color, 0) + 1
                vehicle_types[vehicle_type] = vehicle_types.get(vehicle_type, 0) + 1
            
            for color, count in sorted(color_counts.items()):
                stats_text += f"  {color.title()}: {count}\n"
            
            stats_text += "\nVEHICLE TYPES:\n"
            for vtype, count in sorted(vehicle_types.items()):
                stats_text += f"  {vtype.title()}: {count}\n"
        
        self.stats_text.insert(tk.END, stats_text)
        
        # Display detailed detection results
        details_text = "DETECTION DETAILS\n"
        details_text += "=" * 30 + "\n\n"
        
        # Car details
        if results['cars']:
            details_text += f"VEHICLES DETECTED ({len(results['cars'])}):\n"
            for i, car in enumerate(results['cars'], 1):
                x, y, w, h = car['bbox']
                vtype = car.get('vehicle_type', 'car')
                details_text += f"{i}. {car['color'].title()} {vtype.title()}\n"
                details_text += f"   Confidence: {car['confidence']:.2f}\n"
                details_text += f"   Location: ({x}, {y}) Size: {w}x{h}\n\n"
        
        # People details
        if results['people']:
            details_text += f"PEOPLE DETECTED ({len(results['people'])}):\n"
            for i, person in enumerate(results['people'], 1):
                x, y, w, h = person['bbox']
                details_text += f"{i}. Person at ({x}, {y}) Size: {w}x{h}\n"
        
        details_text += "\n" + "="*30 + "\n"
        details_text += "LEGEND:\n"
        details_text += " Red rectangles = Blue cars\n"
        details_text += " Blue rectangles = Other color cars\n"
        details_text += " Green rectangles = People\n"
        
        self.details_text.insert(tk.END, details_text)
    
    def show_progress(self):
        """Show progress bar"""
        self.progress_bar.grid()
        self.progress_bar.start(10)
    
    def hide_progress(self):
        """Hide progress bar"""
        self.progress_bar.stop()
        self.progress_bar.grid_remove()
    
    def clear_text_widgets(self):
        """Clear text widgets"""
        self.stats_text.delete(1.0, tk.END)
        self.details_text.delete(1.0, tk.END)
    
    def clear_results(self):
        """Clear all results and reset the interface"""
        self.current_image = None
        self.preview_label.config(image="", text="No image uploaded\n\nClick 'Upload Image' or 'Real-time Detection' to start")
        self.preview_label.image = None
        self.clear_text_widgets()
        self.update_status("Ready - Upload an image to start analysis")

def main():
    """Main function with error handling"""
    try:
        print(" Starting Optimized Car Color Detection GUI")
        print(f"TensorFlow version: {tf.__version__}")
        
        root = tk.Tk()
        app = OptimizedCarColorDetectionGUI(root)
        root.mainloop()
        
    except Exception as e:
        print(f" Application error: {e}")
        messagebox.showerror("Application Error", f"Failed to start application: {str(e)}")

if __name__ == "__main__":
    main()

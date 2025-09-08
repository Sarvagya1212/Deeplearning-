import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from ultralytics import YOLO

class ImprovedAnimalDetectionGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.model = YOLO(r"C:\Users\sarva\Emotion_detection-main\animal_detection\runs\detect\animal_detection_v1\weights\best.pt")
        
        # Improved carnivore classification with confidence filtering
        self.carnivore_species = {
    "badger", "bat", "bear", "boar", "cat", "chimpanzee", "coyote",
    "crab", "crow", "dog", "dolphin", "dragonfly", "eagle", "fly",
    "fox", "hedgehog", "hyena", "jellyfish", "ladybugs", "leopard",
    "lion", "lizard", "lobster", "mosquito", "octopus", "otter", "owl",
    "pelecaniformes", "penguin", "pig", "possum", "raccoon", "rat",
    "seahorse", "seal", "shark", "snake", "squid", "starfish", "tiger",
    "whale", "wolf"
}
        
        # Higher confidence requirements for easily confused species
        self.high_confidence_species = {"lion", "tiger", "leopard", "cheetah", "jaguar"}
        
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Animal Detection - Improved Classification")
        self.setGeometry(100, 100, 1200, 800)
        
        # Main widget
        widget = QtWidgets.QWidget()
        self.setCentralWidget(widget)
        layout = QtWidgets.QVBoxLayout(widget)
        
        # Title
        title = QtWidgets.QLabel("Animal Detection with Carnivore Highlighting")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_open = QtWidgets.QPushButton("Open Image")
        self.btn_detect = QtWidgets.QPushButton("Detect Animals")
        self.btn_save = QtWidgets.QPushButton("Save Result")
        
        self.btn_open.setMinimumHeight(40)
        self.btn_detect.setMinimumHeight(40)
        self.btn_save.setMinimumHeight(40)
        
        self.btn_detect.setEnabled(False)
        self.btn_save.setEnabled(False)
        
        btn_layout.addWidget(self.btn_open)
        btn_layout.addWidget(self.btn_detect)
        btn_layout.addWidget(self.btn_save)
        layout.addLayout(btn_layout)
        
        # Image displays
        img_layout = QtWidgets.QHBoxLayout()
        
        # Original image
        original_group = QtWidgets.QGroupBox("Original Image")
        original_layout = QtWidgets.QVBoxLayout()
        self.label_original = QtWidgets.QLabel("Click 'Open Image' to load an image")
        self.label_original.setMinimumSize(500, 350)
        self.label_original.setAlignment(QtCore.Qt.AlignCenter)
        self.label_original.setStyleSheet("border: 2px solid gray; background: white;")
        original_layout.addWidget(self.label_original)
        original_group.setLayout(original_layout)
        
        # Detection result
        result_group = QtWidgets.QGroupBox("Detection Result")
        result_layout = QtWidgets.QVBoxLayout()
        self.label_result = QtWidgets.QLabel("Detection results will appear here")
        self.label_result.setMinimumSize(500, 350)
        self.label_result.setAlignment(QtCore.Qt.AlignCenter)
        self.label_result.setStyleSheet("border: 2px solid gray; background: white;")
        result_layout.addWidget(self.label_result)
        result_group.setLayout(result_layout)
        
        img_layout.addWidget(original_group)
        img_layout.addWidget(result_group)
        layout.addLayout(img_layout)
        
        # Status and info
        self.status_label = QtWidgets.QLabel("Ready - Load an image to begin detection")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 14px; padding: 10px; background: lightgray; margin: 5px;")
        layout.addWidget(self.status_label)
        
        # Connect signals
        self.btn_open.clicked.connect(self.open_image)
        self.btn_detect.clicked.connect(self.detect_animals)
        self.btn_save.clicked.connect(self.save_result)
        
        # Initialize variables
        self.image_path = None
        self.result_image = None
        
    def open_image(self):
        """Open and display an image file"""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Image File", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        
        if file_path:
            self.image_path = file_path
            
            # Display original image with high quality
            self.display_image_file(file_path, self.label_original)
            
            self.btn_detect.setEnabled(True)
            self.btn_save.setEnabled(False)
            self.status_label.setText(f"Image loaded: {file_path.split('/')[-1]} - Click 'Detect Animals'")
            
    def detect_animals(self):
        """Run animal detection with improved classification"""
        if not self.image_path:
            return
            
        self.status_label.setText("Running detection... Please wait")
        QtWidgets.QApplication.processEvents()
        
        try:
            # Load original image
            image = cv2.imread(self.image_path)
            if image is None:
                self.status_label.setText("Error: Could not load image")
                return
                
            # Run detection with confidence filtering
            results = self.model(image, conf=0.3)  # Lower initial threshold
            
            # Process and filter detections
            filtered_detections = self.filter_detections(results)
            
            # Draw detections on image
            self.result_image, stats = self.draw_detections(image.copy(), filtered_detections)
            
            # Display result
            self.display_cv_image(self.result_image, self.label_result)
            
            # Update status
            carnivore_count = stats['carnivores']
            total_count = stats['total']
            big_cat_count = stats['big_cats']
            
            self.status_label.setText(
                f"Detection complete: {total_count} animals found, "
                f"{carnivore_count} carnivores ({big_cat_count} big cats)"
            )
            
            self.btn_save.setEnabled(True)
            
            # Show results popup
            self.show_results_popup(stats)
            
        except Exception as e:
            self.status_label.setText(f"Error during detection: {str(e)}")
            
    def filter_detections(self, results):
        """Filter detections with improved confidence thresholds"""
        filtered_detections = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    confidence = float(box.conf.cpu().numpy())
                    class_id = int(box.cls.cpu().numpy())
                    class_name = result.names[class_id]
                    
                    # Apply different confidence thresholds
                    min_confidence = self.get_min_confidence(class_name)
                    
                    if confidence >= min_confidence:
                        filtered_detections.append({
                            'box': box.xyxy.cpu().numpy().astype(int)[0],
                            'confidence': confidence,
                            'class_name': class_name,
                            'class_id': class_id
                        })
        
        return filtered_detections
        
    def get_min_confidence(self, class_name):
        """Get minimum confidence threshold for each species"""
        class_lower = class_name.lower()
        
        # Higher threshold for easily confused big cats
        if any(species in class_lower for species in self.high_confidence_species):
            return 0.7  # Very high confidence required
        
        # Medium threshold for other carnivores
        elif any(species in class_lower for species in self.carnivore_species):
            return 0.5  # Medium confidence
            
        # Lower threshold for herbivores
        else:
            return 0.4  # Lower confidence acceptable
            
    def draw_detections(self, image, detections):
        """Draw detection boxes with improved visualization"""
        stats = {
            'total': len(detections),
            'carnivores': 0,
            'herbivores': 0,
            'big_cats': 0,
            'domestic': 0
        }
        
        for detection in detections:
            x1, y1, x2, y2 = detection['box']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Classify animal type
            animal_type = self.classify_animal_type(class_name)
            
            # Update statistics
            if animal_type['is_carnivore']:
                stats['carnivores'] += 1
                if animal_type['is_big_cat']:
                    stats['big_cats'] += 1
                elif animal_type['is_domestic']:
                    stats['domestic'] += 1
            else:
                stats['herbivores'] += 1
            
            # Choose colors and thickness based on type
            color, thickness = self.get_display_style(animal_type)
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            
            # Create detailed label
            label = f"{class_name} {confidence:.2f}"
            if animal_type['is_big_cat']:
                label += " [BIG CAT]"
            elif animal_type['is_carnivore']:
                label += " [CARNIVORE]"
                
            # Draw label background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            text_thickness = 2
            
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, text_thickness
            )
            
            # Background rectangle
            cv2.rectangle(image, 
                         (x1, y1 - text_height - 10),
                         (x1 + text_width + 10, y1),
                         color, -1)
            
            # Text
            cv2.putText(image, label, (x1 + 5, y1 - 5), 
                       font, font_scale, (255, 255, 255), text_thickness)
        
        return image, stats
        
    def classify_animal_type(self, class_name):
        """Classify animal with detailed categories"""
        class_lower = class_name.lower()
        
        is_big_cat = any(species in class_lower for species in self.high_confidence_species)
        is_domestic = any(species in class_lower for species in ["cat", "dog"])
        is_carnivore = any(species in class_lower for species in self.carnivore_species)
        
        return {
            'is_carnivore': is_carnivore,
            'is_big_cat': is_big_cat,
            'is_domestic': is_domestic
        }
        
    def get_display_style(self, animal_type):
        """Get color and thickness for different animal types"""
        if animal_type['is_big_cat']:
            return (0, 0, 255), 5  # Bright red, thick line
        elif animal_type['is_carnivore']:
            return (0, 100, 255), 3  # Orange-red, medium line
        else:
            return (0, 255, 0), 2  # Green, thin line
            
    def display_image_file(self, file_path, label):
        """Display image from file with high quality"""
        pixmap = QtGui.QPixmap(file_path)
        scaled_pixmap = pixmap.scaled(
            label.size(), 
            QtCore.Qt.KeepAspectRatio, 
            QtCore.Qt.SmoothTransformation
        )
        label.setPixmap(scaled_pixmap)
        
    def display_cv_image(self, cv_image, label):
        """Display OpenCV image with high quality"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_image.shape
        bytes_per_line = channels * width
        
        # Create QImage
        qt_image = QtGui.QImage(
            rgb_image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        
        # Convert to QPixmap and scale
        pixmap = QtGui.QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        )
        
        label.setPixmap(scaled_pixmap)
        
    def show_results_popup(self, stats):
        """Show detailed detection results"""
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle("Animal Detection Results")
        
        if stats['carnivores'] > 0:
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            
            text = f"🚨 CARNIVORES DETECTED! 🚨\n\n"
            text += f"Total animals found: {stats['total']}\n"
            text += f"Carnivorous animals: {stats['carnivores']}\n"
            
            if stats['big_cats'] > 0:
                text += f"  • Big cats: {stats['big_cats']} (RED boxes)\n"
            if stats['domestic'] > 0:
                text += f"  • Domestic carnivores: {stats['domestic']} (ORANGE boxes)\n"
            if stats['herbivores'] > 0:
                text += f"Herbivorous animals: {stats['herbivores']} (GREEN boxes)\n"
                
            text += f"\n⚠️ Big cats are shown with thick RED borders\n"
            text += f"⚠️ Other carnivores have ORANGE borders"
            
        else:
            msg.setIcon(QtWidgets.QMessageBox.Information)
            text = f"✅ NO CARNIVORES DETECTED\n\n"
            text += f"Total animals found: {stats['total']}\n"
            text += f"All detected animals are herbivorous\n"
            text += f"(shown with GREEN borders)"
            
        msg.setText(text)
        msg.exec_()
        
    def save_result(self):
        """Save the detection result image"""
        if self.result_image is not None:
            file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save Detection Result", "animal_detection_result.jpg",
                "Image Files (*.jpg *.png)"
            )
            
            if file_path:
                cv2.imwrite(file_path, self.result_image)
                self.status_label.setText(f"Result saved: {file_path.split('/')[-1]}")

def main():
    """Main function to run the application"""
    app = QtWidgets.QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Animal Detection GUI")
    app.setStyle('Fusion')  # Modern look
    
    # Create and show main window
    window = ImprovedAnimalDetectionGUI()
    window.show()
    
    # Run application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

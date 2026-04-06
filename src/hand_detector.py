# src/hand_detector.py
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import urllib.request
import os

class DummyResults:
    """A compatibility layer to mimic the old mp.solutions API structure"""
    def __init__(self, hand_landmarks, handedness):
        self.multi_hand_landmarks = hand_landmarks
        self.multi_handedness = handedness

class HandDetector:
    def __init__(self):
        # 1. Download the required model task file if it doesn't exist
        self.task_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
        if not os.path.exists(self.task_path):
            print("Downloading MediaPipe Hand Landmarker model (First run only)...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, self.task_path)
            print("Download complete.")

        # 2. Setup the modern Tasks API options
        base_options = python.BaseOptions(model_asset_path=self.task_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # 3. Initialize variables
        self.results = DummyResults(None, None)
        self.raw_results = None

        # Standard ISL hand connections for manual drawing
        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
        ]

    def find_hands(self, img, draw=True):
        """Detect both hands using the Tasks API"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        self.raw_results = self.detector.detect(mp_image)
        
        # Map to old API structure for backward compatibility with your existing scripts
        self.results.multi_hand_landmarks = self.raw_results.hand_landmarks if self.raw_results else None
        self.results.multi_handedness = self.raw_results.handedness if self.raw_results else None

        # Custom drawing logic since solutions.drawing_utils is deprecated
        if draw and self.raw_results and self.raw_results.hand_landmarks:
            h, w, c = img.shape
            for hand_landmarks in self.raw_results.hand_landmarks:
                # Draw lines
                for connection in self.HAND_CONNECTIONS:
                    p1 = hand_landmarks[connection[0]]
                    p2 = hand_landmarks[connection[1]]
                    x1, y1 = int(p1.x * w), int(p1.y * h)
                    x2, y2 = int(p2.x * w), int(p2.y * h)
                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Draw points
                for mark in hand_landmarks:
                    x, y = int(mark.x * w), int(mark.y * h)
                    cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
                    
        return img
    
    def get_landmarks_array(self):
        """Extract and format landmarks for the Sequence Transformer"""
        if not self.raw_results or not self.raw_results.hand_landmarks:
            return None
            
        landmarks_combined = []
        handedness_list = self.raw_results.handedness
        
        if len(self.raw_results.hand_landmarks) == 1:
            hand = self.raw_results.hand_landmarks[0]
            hand_type = handedness_list[0][0].category_name # 'Left' or 'Right'
            
            hand_landmarks = []
            for mark in hand:
                hand_landmarks.extend([mark.x, mark.y, mark.z])
                
            empty_hand = [0.0] * 63
            
            if hand_type == 'Left':
                landmarks_combined = hand_landmarks + empty_hand
            else:
                landmarks_combined = empty_hand + hand_landmarks
                
        elif len(self.raw_results.hand_landmarks) == 2:
            hand1_landmarks = []
            hand2_landmarks = []
            
            for idx, hand in enumerate(self.raw_results.hand_landmarks):
                current_hand = []
                for mark in hand:
                    current_hand.extend([mark.x, mark.y, mark.z])
                
                if handedness_list[idx][0].category_name == 'Left':
                    hand1_landmarks = current_hand
                else:
                    hand2_landmarks = current_hand
                    
            if not hand1_landmarks: hand1_landmarks = [0.0] * 63
            if not hand2_landmarks: hand2_landmarks = [0.0] * 63
                
            landmarks_combined = hand1_landmarks + hand2_landmarks
            
        return np.array(landmarks_combined) if landmarks_combined else None

# Quick test visualization
def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    
    while True:
        success, img = cap.read()
        if not success: break
            
        img = cv2.flip(img, 1)
        img = detector.find_hands(img)
        landmarks = detector.get_landmarks_array()
        
        if detector.results.multi_hand_landmarks:
            num_hands = len(detector.results.multi_hand_landmarks)
            cv2.putText(img, f"Hands detected: {num_hands}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if detector.results.multi_handedness:
                y_pos = 70
                for idx, hand_info in enumerate(detector.results.multi_handedness):
                    hand_type = hand_info[0].category_name
                    cv2.putText(img, f"Hand {idx+1}: {hand_type}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    y_pos += 40
        else:
            cv2.putText(img, "No hands detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Tasks API Test", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
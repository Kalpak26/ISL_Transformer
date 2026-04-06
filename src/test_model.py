# src/test_model.py
import cv2
import numpy as np
import torch
import torch.nn as nn
import os
import json
import time
import math
from collections import deque
from datetime import datetime
from googletrans import Translator
from hand_detector import HandDetector

# ─── 1. PyTorch Model Definitions (Required to load weights) ────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class ISLTransformer(nn.Module):
    def __init__(self, num_classes, input_dim=126, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :] 
        x = self.dropout(x)
        out = self.fc(x)
        return out

# ─── 2. Main Inference Class ─────────────────────────────────────────────
class SignPredictor:
    def __init__(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.detector = HandDetector()
        
        # Paths
        model_path = os.path.join(self.project_root, 'models', 'transformer_isl.pth')
        label_map_path = os.path.join(self.project_root, 'models', 'label_map.json')
        
        if not os.path.exists(model_path) or not os.path.exists(label_map_path):
            raise FileNotFoundError("Model weights or label map not found in 'models/' directory.")

        # Load Label Map
        with open(label_map_path, 'r') as f:
            class_to_idx = json.load(f)
            self.idx_to_class = {v: k for k, v in class_to_idx.items()}
            num_classes = len(self.idx_to_class)

        # Setup Device & Model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ISLTransformer(num_classes=num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval() # Set to evaluation mode!
        print(f"Model loaded successfully on {self.device}")

        # Sequence Data Setup
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Translation Setup
        self.translator = Translator()
        self.output_dir = os.path.join(self.project_root, 'output')
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = os.path.join(self.output_dir, f'translations_{timestamp}.txt')
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write("ISL Translation Log (Transformer)\n")
            f.write("=" * 50 + "\n\n")

        # UI & Logic Variables
        self.current_word = ""
        self.last_translated_word = ""
        self.current_prediction = "Waiting..."
        self.confidence = 0.0
        
        # Prediction smoothing (require X consecutive same predictions)
        self.consecutive_predictions = 0
        self.last_pred_class = None
        self.CONFIDENCE_THRESHOLD = 0.80

    def save_translation(self, word):
        if not word or word.isspace(): return
        try:
            hindi = self.translator.translate(word, src='en', dest='hi').text
            marathi = self.translator.translate(word, src='en', dest='mr').text

            print(f"\n[Translated] English: {word} | Hindi: {hindi} | Marathi: {marathi}")

            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(f"Time: {datetime.now().strftime('%H:%M:%S')}\n")
                f.write(f"English: {word}\n")
                f.write(f"Hindi:   {hindi}\n")
                f.write(f"Marathi: {marathi}\n")
                f.write("-" * 30 + "\n")
        except Exception as e:
            print(f"Translation error: {str(e)}. (Check internet connection or googletrans version)")

    def predict_sign(self):
        cap = cv2.VideoCapture(0)
        
        print("\n=== ISL Transformer Real-Time Inference ===")
        print("Controls:")
        print("  [SPACE]     : Add space to word")
        print("  [BACKSPACE] : Delete last character")
        print("  [ENTER]     : Translate current word and save")
        print("  [C]         : Clear current word")
        print("  [Q]         : Quit")
        print(f"Logging to  : {self.output_file}\n")

        while True:
            success, frame = cap.read()
            if not success: break

            frame = cv2.flip(frame, 1)
            frame = self.detector.find_hands(frame, draw=True)
            landmarks = self.detector.get_landmarks_array()

            # 1. Update Rolling Window
            if landmarks is not None:
                self.frame_buffer.append(landmarks)
            else:
                # If hands disappear, feed zero arrays to maintain time continuity
                self.frame_buffer.append(np.zeros(126))

            # 2. Run Model if Buffer is Full
            if len(self.frame_buffer) == self.sequence_length:
                # Convert buffer to tensor: shape (1, 30, 126)
                seq_array = np.array(self.frame_buffer)
                seq_tensor = torch.FloatTensor(seq_array).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    outputs = self.model(seq_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    max_prob, predicted_idx = torch.max(probabilities, 1)
                    
                    self.confidence = max_prob.item()
                    pred_class = self.idx_to_class[predicted_idx.item()]

                    # 3. Smoothing Logic (Only accept if highly confident)
                    if self.confidence > self.CONFIDENCE_THRESHOLD:
                        if pred_class == self.last_pred_class:
                            self.consecutive_predictions += 1
                        else:
                            self.consecutive_predictions = 1
                            self.last_pred_class = pred_class

                        # If we see the same sign confidently for 10 frames, lock it in
                        if self.consecutive_predictions >= 10:
                            if self.current_prediction != pred_class: # Only print when it changes
                                print(f"Live Detection: {pred_class} ({self.confidence*100:.1f}%)")
                                # Automatically append the letter!
                                # (Because this is inside the 'if !=', it won't spam the letter if you hold the sign)
                                self.current_word += pred_class
                                
                            self.current_prediction = pred_class
                    else:
                        self.current_prediction = "Uncertain..."
                        self.consecutive_predictions = 0

            # 4. Display UI
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 100), (18, 18, 18), -1)
            
            # Show live prediction
            color = (0, 255, 0) if self.confidence > self.CONFIDENCE_THRESHOLD else (0, 165, 255)
            cv2.putText(frame, f"Sign: {self.current_prediction} ({self.confidence*100:.1f}%)", 
                       (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Show word being built
            cv2.putText(frame, f"Word: {self.current_word}", 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 180, 0), 3)

            cv2.imshow("ISL Translator", frame)

            # 5. Keyboard Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.current_word = ""
                print("\nCleared current word")
            elif key == ord(' '):
                self.current_word += " "
            elif key == 8 or key == 127: # Backspace
                self.current_word = self.current_word[:-1]
            elif key == 13: # Enter key (Carriage Return)
                if self.current_word.strip() != "":
                    self.save_translation(self.current_word.strip())
                    self.last_translated_word = self.current_word
                    self.current_word = "" # Clear after translating

        cap.release()
        cv2.destroyAllWindows()
        print("\nSession ended.")

if __name__ == "__main__":
    predictor = SignPredictor()
    predictor.predict_sign()
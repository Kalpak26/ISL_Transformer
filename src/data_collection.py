# src/data_collection.py
import cv2
import numpy as np
import os
import time
from hand_detector import HandDetector

# ── Config ────────────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 30
NUM_SEQUENCES   = 30
DATA_ROOT       = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sequences")
GUIDE_IMAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "guides")
CAMERA_INDEX    = 0
COUNTDOWN_SECS  = 3

# UI Colors (BGR)
C_GREEN  = (0, 220, 100)
C_YELLOW = (0, 220, 220)
C_WHITE  = (240, 240, 240)
C_DARK   = (18, 18, 18)
C_ACCENT = (255, 180, 0)
C_BLUE   = (220, 160, 0)
C_RED    = (60,  60, 220)

os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(GUIDE_IMAGE_DIR, exist_ok=True)


class SequenceDataCollector:
    def __init__(self):
        self.detector = HandDetector()
        self.opened_guide_windows = []

    def show_guide_images(self, sign_name):
        """Loads all images from the guides directory in resizable windows."""
        supported_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        
        if not os.path.exists(GUIDE_IMAGE_DIR):
            return 0

        image_files = [f for f in os.listdir(GUIDE_IMAGE_DIR) if f.lower().endswith(supported_exts)]
        
        for img_file in image_files:
            img_path = os.path.join(GUIDE_IMAGE_DIR, img_file)
            guide_img = cv2.imread(img_path)

            if guide_img is not None:
                window_name = f"Guide: {img_file}"
                # CRITICAL: WINDOW_NORMAL allows the window to be resized manually by the user
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow(window_name, guide_img)
                self.opened_guide_windows.append(window_name)

        return len(self.opened_guide_windows)

    def close_guide_windows(self):
        """Safely closes only the guide windows."""
        for window in self.opened_guide_windows:
            cv2.destroyWindow(window)
        self.opened_guide_windows.clear()

    def _put(self, f, txt, pos, col=C_WHITE, sc=0.8, th=2):
        cv2.putText(f, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)

    def _top_bar(self, frame, sign, seq_idx, total, frame_num=-1, paused=False, hand_found=True):
        w = frame.shape[1]
        cv2.rectangle(frame, (0, 0), (w, 115), C_DARK, -1)
        self._put(frame, f"Sign: {sign.upper()}", (15, 42), C_GREEN, 1.1, 2)

        if paused:
            state, col = "⏸  PAUSED - SPACE to resume", C_YELLOW
        elif not hand_found:
            state, col = "✋ Waiting for hand...", C_RED
        else:
            state, col = "RECORDING", C_ACCENT
            
        self._put(frame, state, (15, 78), col, 0.8)
        self._put(frame, f"Seq {seq_idx+1}/{total}", (w//2 - 70, 50), C_WHITE, 1.0, 2)

        if frame_num >= 0 and not paused and hand_found:
            filled = int(w * frame_num / SEQUENCE_LENGTH)
            cv2.rectangle(frame, (0, 103), (w, 115), (40,40,40), -1)
            cv2.rectangle(frame, (0, 103), (filled, 115), C_BLUE, -1)
            self._put(frame, f"Frame {frame_num+1}/{SEQUENCE_LENGTH}", (15, 113), C_WHITE, 0.42, 1)

    def _countdown(self, cap):
        for tick in range(COUNTDOWN_SECS, 0, -1):
            t_end = time.time() + 1.0
            while time.time() < t_end:
                ok, frame = cap.read()
                if not ok: return False
                
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                ov = frame.copy()
                cv2.rectangle(ov, (0,0), (w,h), (0,0,0), -1)
                cv2.addWeighted(ov, 0.45, frame, 0.55, 0, frame)
                
                cv2.putText(frame, str(tick), (w//2-40, h//2+20), 
                            cv2.FONT_HERSHEY_DUPLEX, 5.0, C_YELLOW, 8, cv2.LINE_AA)
                self._put(frame, "Get ready...", (w//2-100, h//2+90), C_WHITE, 1.0, 2)
                
                cv2.imshow("ISL Data Collection", frame)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    return False
        return True

    def collect_sign_data(self, sign_name):
        save_dir = os.path.join(DATA_ROOT, sign_name)
        os.makedirs(save_dir, exist_ok=True)

        existing = sorted(int(f.split(".")[0]) for f in os.listdir(save_dir) if f.endswith(".npy") and f.split(".")[0].isdigit())
        start_idx = (max(existing) + 1) if existing else 0
        remaining = NUM_SEQUENCES - start_idx

        if remaining <= 0:
            print(f"'{sign_name}' already has {NUM_SEQUENCES} sequences. Skipping.")
            return

        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            print("Camera unavailable.")
            return

        # Open resizable guide windows
        num_guides = self.show_guide_images(sign_name)
        if num_guides > 0:
            print(f"Opened {num_guides} guide window(s). You can resize them now.")

        print(f"\n=== Collecting: '{sign_name}' ===")
        print(f"    Sequences needed: {remaining} (from index {start_idx})")
        print("    [SPACE] = pause/resume   [Q] = quit\n")

        for seq in range(remaining):
            global_idx = start_idx + seq

            if not self._countdown(cap):
                print("\nAborted during countdown.")
                break

            sequence_data = []
            paused = False
            frame_num = 0

            while frame_num < SEQUENCE_LENGTH:
                ok, frame = cap.read()
                if not ok:
                    print("Camera read failed.")
                    break

                frame = cv2.flip(frame, 1)
                frame = self.detector.find_hands(frame, draw=True)
                
                # Using the modern tasks API properties from your hand_detector
                hand_found = self.detector.results and self.detector.results.multi_hand_landmarks

                self._top_bar(frame, sign_name, seq, remaining, frame_num, paused, hand_found)
                cv2.imshow("ISL Data Collection", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.close_guide_windows()
                    cap.release()
                    cv2.destroyAllWindows()
                    print("\nAborted by user.")
                    return
                if key == ord(" "):
                    paused = not paused

                if not paused and hand_found:
                    lm = self.detector.get_landmarks_array()
                    # Ensure we append exactly 126 zeros if landmark extraction fails for a frame
                    sequence_data.append(lm if lm is not None else np.zeros(126))
                    frame_num += 1

            if len(sequence_data) == SEQUENCE_LENGTH:
                path = os.path.join(save_dir, f"{global_idx}.npy")
                np.save(path, np.array(sequence_data))
                print(f"  Saved sequence {global_idx}")

        self.close_guide_windows()
        cap.release()
        cv2.destroyAllWindows()
        total = len([f for f in os.listdir(save_dir) if f.endswith(".npy")])
        print(f"\nDone! '{sign_name}' now has {total} sequence(s).")

def main():
    while True:
        print("\n" + "-"*50)
        print("  ISL - Sequence Data Collection")
        print("-" * 50)
        raw = input("Enter sign name (e.g. 'hello') or 'exit' to quit: ").strip().lower()
        
        if raw == 'exit':
            break
        if raw:
            collector = SequenceDataCollector()
            collector.collect_sign_data(raw)

if __name__ == "__main__":
    main()

# 🤟 Indian Sign Language (ISL) Sequence Transformer

A real-time Indian Sign Language translation system powered by a Deep Learning Sequence Transformer. 

Unlike traditional static machine learning models (like Random Forest) that look at a single frame, this project uses a PyTorch Transformer Encoder to analyze a continuous **rolling window of 30 frames**. This allows the model to understand the *motion* and *temporal context* of signs, resulting in highly accurate, state-of-the-art gesture recognition. Translated words are automatically converted into Hindi and Marathi.

<img width="1317" height="610" alt="Isl_Transformer_Arch" src="https://github.com/user-attachments/assets/d207db6f-b46b-4c75-b7d2-a6b573018846" />

---

## 📁 Directory Structure

Before starting, ensure your project folder looks exactly like this:

```text
isl_transformer_project/
├── data/
│   └── sequences/        # Auto-generated: Stores your .npy training data
├── guides/               # Put reference images here (e.g., 'hello.jpg', 'A.jpg')
├── models/               # Put your Colab-trained .pth and .json files here
├── output/               # Auto-generated: Stores your translation text logs
├── src/
│   ├── hand_detector.py             # MediaPipe Tasks API wrapper
│   ├── data_collection.py           # Script to record sign sequences
│   └── test_model.py                # Real-time webcam inference script
│   └── ISL_seq_transformer.ipynb    # Training script, uploaded on Colab/Kaggle for cloud GPU 
└── requirements.txt                 # Python dependencies
```

---

## ⚙️ Step 1: Setup & Installation

It is highly recommended to run this project inside an isolated Python virtual environment.

**1. Create a Conda & Activate Environment:**
```bash
# Create the environment
conda create --name isl_trans python=3.10

# Activate the environment
conda activate isl_trans
```

**2. Install Dependencies:**
Ensure you have the `requirements.txt` file in your main folder, then run:
```bash
pip install -r requirements.txt
```
*Note: The first time you run the scripts, it will automatically download a small Google MediaPipe model file (`hand_landmarker.task`) in the background. It may take 5-10 seconds to start up the very first time.*

---

## 🎥 Step 2: Data Collection (Local)

To train the Transformer, you need to collect sequences of yourself performing the signs.

1.  **(Optional)** Place reference images in the `guides/` folder (e.g., `A.jpg`, `hello.png`). The script will automatically open these as resizable reference windows while you record.
2.  Run the collection script:
    ```bash
    python src/data_collection.py
    ```
3.  Type the name of the sign you want to record (e.g., `hello`).
4.  **Controls:**
    * `[SPACE]` : Pause / Resume recording. Use this to get your hands in position!
    * `[Q]` : Quit early and save progress.
5.  The script will record 30 sequences of 30 frames each and save them as lightning-fast `.npy` tensor files in `data/sequences/<sign_name>/`.

---

## 🧠 Step 3: Model Training (Google Colab)

Because Transformers require heavy computation, training is done on a free cloud GPU.

1.  Locate your `data/sequences/` folder. Zip the `sequences/` folder into a file named `sequences.zip`.
2.  Open Google Colab and upload `ISL_seq_transformer.ipynb` notebook.
3.  Go to **Runtime > Change runtime type** and select **T4 GPU**.
4.  Upload `sequences.zip` to the Colab files pane.
5.  Paste the provided PyTorch training script into a cell and hit Run. 
6.  Once training hits 50 epochs, two files will be generated:
    * `transformer_isl.pth` (The brain/weights of the model)
    * `label_map.json` (The dictionary mapping numbers to your sign names)
7.  Download both files and place them into your local `models/` folder.

---

## 🚀 Step 4: Real-Time Translation (Local)

Now for the fun part. Run the live inference script to translate your signs into English, Hindi, and Marathi in real-time.

1.  Run the script:
    ```bash
    python src/test_model.py
    ```
2.  Perform your signs in front of the camera. The model requires 10 frames of confident, continuous recognition to "lock in" a sign to prevent flickering.
3.  **Controls for Building Sentences:**
    * `[SPACE]` : Add a space to your current word.
    * `[BACKSPACE]` : Delete the last character.
    * `[C]` : Clear the entire current word.
    * `[ENTER]` : Translates the word to Hindi/Marathi and saves it to a log file in the `output/` folder.
    * `[Q]` : Quit the application.

---

## 🛠️ Troubleshooting

* **"AttributeError: module 'mediapipe' has no attribute 'solutions'"**
    You are using an older script with a newer MediaPipe version. Ensure you are using the updated `hand_detector.py` provided in this repository, which utilizes the modern Tasks API.
* **"FileNotFoundError: Model weights not found"**
    Ensure `transformer_isl.pth` and `label_map.json` are exactly named and placed inside the `models/` directory.
* **Translation Error/Timeout:**
    The `googletrans` API can sometimes timeout if queried too rapidly or if your internet connection drops. Ensure you are using `googletrans==4.0.0-rc1` as specified in the requirements.

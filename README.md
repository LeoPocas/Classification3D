# DSE Action Classification Model

This project trains a real-time LSTM model to classify human actions based on pose-estimation keypoints extracted from videos.

The pipeline is structured into four main phases, executed by scripts in the root directory. All source code modules (like the feature extractor, model definition, and data generators) are located in the `src/` directory.

---

## 1. Setup & Configuration

Before running any scripts, ensure your environment is capable of using your GPU.

The most important file for setup is **`src/config.py`**.

You **must** review this file and ensure all paths and parameters are set correctly for your environment. Key variables to check:

* **Paths:** `LABEL_FILE_PATH`, `VIDEO_SOURCE_DIR`, `YOLO_MODEL`, `OUTPUT_DIR`, etc.
* **Hyperparameters:** `MAX_PEOPLE`, `SEQUENCE_LENGTH`, `EPOCHS`, `BATCH_SIZE`, etc.
* **Filters:** `EXCLUDE_PREFIXES`, `NEGATIVE_KEEP_RATE`, `APPLY_AUGMENTATION`.

---

## 2. Execution Workflow

Run the following scripts from the project's root directory (`DSE-ClassificationModel/`) in this order.

### Phase 1: Keypoint Extraction

* **Script:** `main_preprocess.py`
* **What it does:** This script recursively finds all `.mp4` files in your `VIDEO_SOURCE_DIR` (as defined in `config.py`). It runs the YOLOv8-Pose model on every frame of every video and saves the full keypoint data (as "scene vectors") into `.npy` files.
* **Output:** The `yoloData/` directory, which mirrors the folder structure of your source videos (e.g., `yoloData/floss/floss_1.npy`).
* **Command:**
    ```bash
    python main_preprocess.py
    ```

### Phase 2: Sequence Generation

* **Script:** `prepare_sequences.py`
* **What it does:** This script reads the `.npy` files from `yoloData/` and your `videosLabeled.xlsx` file. It applies all the project logic:
    1.  Filters out excluded actions.
    2.  Applies a sliding window (`SEQUENCE_LENGTH`) using the "Any Positive Overlap" strategy.
    3.  Performs **Undersampling** on the "Negative" class.
    4.  Performs **Data Augmentation** (expansion) on positive training classes.
    5.  Performs a **Stratified Train/Validation Split** based on video labels.
* **Output:** The `lstm_sequence_data/` directory, containing `train/` and `val/` subfolders filled with chunked `_X.npy` and `_y.npy` files, plus a `label_classes.npy` map.
* **Command:**
    ```bash
    python prepare_sequences.py
    ```

### Phase 3: Model Training

* **Script:** `train_model.py`
* **What it does:** This script loads the chunked data using the `PoseDataGenerator`. It builds the LSTM model (defined in `src/model_definition.py`) and starts the training process.
* **Output:** The best trained model is saved as a `.keras` file in the `models/` directory. Training logs are saved in the `logs/` directory.
* **Command:**
    ```bash
    python train_model.py
    ```
* **Monitoring (Optional):** To view live training graphs, run this in a *separate* terminal:
    ```bash
    tensorboard --logdir ./logs
    ```

### Phase 4: Model Evaluation

* **Script:** `evaluate.py`
* **What it does:** After training, this script loads your best saved model from `models/`. It runs predictions on the validation set and prints a detailed Classification Report and F1-scores to the console.
* **Output:** Prints a report to the terminal and saves a `confusion_matrix.png` image to the project root.
* **Command:**
    ```bash
    python evaluate.py
    ```

---

## 3. Utility Scripts

These are helper scripts for debugging and maintenance.

### Visual YOLO Check

* **Script:** `yoloVideos.py`
* **What it does:** To visually check the quality of YOLO's pose estimation. It processes a single video and saves a new `.mp4` file (e.g., `floss_1_processed.mp4`) with the detected skeletons drawn on it.
* **How to use:**
    1.  Edit the `VIDEO_PATH` variable inside `yoloVideos.py` to point to the video you want to test.
    2.  Run:
        ```bash
        python yoloVideos.py
        ```

### Clean Models Directory

* **Script:** `clean_models.py`
* **What it does:** A simple utility to delete the `models/` directory before a fresh training run.
* **Command:**
    ```bash
    python clean_models.py
    ```

### Check .npy File Shape

* **Script:** `check.py`
* **What it does:** A debug script to load a single `.npy` file (e.g., from `yoloData/`) and print its `shape`, helping to confirm the output of Phase 1.
* **How to use:**
    1.  Edit the `npy_filename` variable inside `check.py`.
    2.  Run:
        ```bash
        python check.py
        ```

Deepfake Detection Using CNN + RNN
📌 Overview

Deepfake videos pose a serious threat to digital trust by manipulating facial features in videos using deep learning techniques.
This project focuses on detecting deepfake videos by analyzing both spatial and temporal facial features across video frames.

A Convolutional Neural Network (CNN) is used to extract spatial facial features from individual frames, while a Recurrent Neural Network (RNN) (LSTM/GRU) captures temporal inconsistencies across consecutive frames to classify a video as REAL or FAKE.

This project is developed for educational and research purposes.

🚀 Features

Video frame extraction using OpenCV

Face detection and preprocessing

CNN-based spatial feature extraction

RNN-based temporal sequence analysis

Binary classification: REAL / FAKE

Supports MP4 and AVI video formats

🛠️ Technologies Used

Python

OpenCV

NumPy

TensorFlow / Keras (or PyTorch)

CNN

RNN (LSTM / GRU)

📁 Project Structure
Deepfake-Detection/
├── train.py              # Model training script
├── model.py              # CNN + RNN model architecture
├── utils/                # Helper and preprocessing functions
├── output/               # Output results (placeholder / demo info)
│   └── README.md
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation

▶️ How to Run the Project
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Prepare Dataset

Input video files are processed frame by frame

Faces are detected and preprocessed before training

3️⃣ Train the Model
python train.py

4️⃣ Run Prediction
python model.py


The model outputs whether the input video is REAL or FAKE.

📊 Output

Classification result: REAL / FAKE

Processed frames and results are stored in the output/ folder

⚠️ Due to repository size limits and ethical considerations,
the final demo video will be shared via an external link once completed.

⚠️ Ethical Considerations

This project is developed strictly for educational and academic purposes.

All videos and facial data used belong to the author or are used with proper consent

No copyrighted, celebrity, or misleading content is used

The project does not promote misuse of deepfake technology

📌 Project Status

🔧 Ongoing / Under Improvement

Future enhancements may include:

Improved accuracy using advanced architectures

Attention mechanisms

Integration of deepfake generation & detection pipeline

👩‍💻 Author

Reethika
B.Tech (3rd Year)
Artificial Intelligence / Computer Science

⭐ Acknowledgements

OpenCV Documentation

TensorFlow / PyTorch Resources

Research papers on Deepfake Detection

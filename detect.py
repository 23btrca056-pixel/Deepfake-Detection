import numpy as np
from tensorflow.keras.models import load_model
from utils.preprocess import extract_frames

MODEL_PATH = "model/deepfake_model.h5"

def predict(video_path):
    model = load_model(MODEL_PATH)

    frames = extract_frames(video_path)
    frames = np.array(frames).reshape(1, 20, 128, 128, 3)

    pred = model.predict(frames)[0][0]

    return "FAKE" if pred > 0.5 else "REAL"


if __name__ == "__main__":
    video_path = "Samples/sample_video.mp4"
    result = predict(video_path)

    print("Prediction:", result)

    with open("output/result.txt", "w") as f:
        f.write(result)

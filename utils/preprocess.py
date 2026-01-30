import cv2

def extract_frames(video_path, max_frames=20):
    frames = []
    cap = cv2.VideoCapture(video_path)

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (128, 128))
        frame = frame / 255.0
        frames.append(frame)

    cap.release()

    # Pad if few frames
    while len(frames) < max_frames:
        frames.append(frames[-1])

    return frames

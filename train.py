import numpy as np
from model import build_deepfake_model

# -----------------------------------------------------------
# 1. Create Dummy Training Data (for demonstration)
# -----------------------------------------------------------
# X shape: (10 videos, 20 frames, 128x128 RGB)
X = np.random.rand(10, 20, 128, 128, 3)

# y shape: (10 labels: 0 = real, 1 = fake)
y = np.random.randint(0, 2, 10)

# -----------------------------------------------------------
# 2. Build the Model
# -----------------------------------------------------------
model = build_deepfake_model()

print("\n==============================")
print("     TRAINING STARTED")
print("==============================\n")

# -----------------------------------------------------------
# 3. Train the Model
# -----------------------------------------------------------
model.fit(X, y, epochs=2, batch_size=2)

# -----------------------------------------------------------
# 4. Save Trained Model
# -----------------------------------------------------------
model.save("model/deepfake_model.h5")

print("\n==============================")
print(" TRAINING COMPLETED SUCCESSFULLY ")
print(" Model saved as 'deepfake_model.h5'")
print("==============================\n")

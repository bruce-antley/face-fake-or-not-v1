from huggingface_hub import hf_hub_download
from fastai.learner import load_learner
from PIL import Image
import pathlib

# For compatibility on some systems
pathlib.PosixPath = pathlib.WindowsPath if hasattr(pathlib, "WindowsPath") else pathlib.PosixPath

# Download the model from Hugging Face
model_path = hf_hub_download(
    repo_id="BA-Baracus/face-fake-or-not-v1",
    filename="round2_final.pkl"
)

# Load the model
learn = load_learner(model_path)

# Load a test image (replace with your own path if testing locally)
test_image_path = "face_fake_or_not_gradio.jpg"
img = Image.open(test_image_path)

# Get prediction
pred_class, pred_idx, probs = learn.predict(img)

# Output result
print(f"Prediction: {pred_class}")
print(f"Probabilities: {probs}")

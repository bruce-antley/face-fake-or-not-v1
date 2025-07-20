import gradio as gr
from fastai.vision.all import *

# Load your exported model (assumes 'export.pkl' is in the repo)
learn = load_learner("round2_final.pkl")

# Prediction function
def predict(img):
    pred, idx, probs = learn.predict(img)
    return {learn.dls.vocab[i]: float(probs[i]) for i in range(len(probs))}

# Gradio Interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="Face: Fake or Not?",
    description="A mid-quality model from a newbie developer. :) Upload a face image to check whether it's (probably) real or fake."
)

# Launch for testing (optional when running as a Space)
if __name__ == "__main__":
    demo.launch()
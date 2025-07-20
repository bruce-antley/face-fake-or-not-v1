
# face-fake-or-not-v1

> A ResNet50-based image classifier fine-tuned on real and AI-generated faces using FastAI.

This project explores whether an image of a human face is real or AI-generated ("fake"). It's built using the [FastAI](https://www.fast.ai) library, which simplifies deep learning workflows while leveraging PyTorch under the hood. This version uses a ResNet50 backbone, trained and fine-tuned across progressively harder datasets.

---

### Demo

Here's an example of the model in action, hosted on [Hugging Face Spaces](https://huggingface.co/spaces/BA-Baracus/face_fake_or_not_v1):

![Gradio Interface](face_fake_or_not_gradio.jpg)

---

### Model Evolution & Performance

The model went through multiple rounds of fine-tuning:

| Round     | Real Accuracy | Fake Accuracy |
|-----------|---------------|----------------|
| Round 0   | 98.0%         | 5.6%           |
| Round 1   | 51.3%         | 79.6%          |
| Round 2   | 92.2%         | 55.6%          |

- **Round 0** was trained on relatively easy examples.
- **Round 1** was tuned on very hard fakes, causing a drop in real accuracy.
- **Round 2** attempted to balance both, yielding improved performance on both.

---

### Built With

- Python + FastAI
- PyTorch
- Gradio (for the demo UI)
- Hugging Face Spaces (for deployment)

---

### Try It Yourself

👉 [Launch the demo on Hugging Face](https://huggingface.co/spaces/BA-Baracus/face_fake_or_not_v1)

Or clone the repo and run locally:

<pre><code>```bash
git clone https://github.com/bruce-antley/face-fake-or-not-v1.git
cd face-fake-or-not-v1
```</code></pre>


⸻
### Model Info

This app uses a FastAI-trained binary classifier that predicts whether a face is real or AI-generated.

The model is hosted on Hugging Face:
👉 round2_final.pkl

To load it in Python:

from huggingface_hub import hf_hub_download
from fastai.learner import load_learner

model_path = hf_hub_download(
    repo_id="BA-Baracus/face-fake-or-not-v1",
    filename="round2_final.pkl"
)

learn = load_learner(model_path)


⸻

### License

This project is licensed under the MIT License.


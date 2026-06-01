# Fine-Tuning Experiments Log: Fine-Tuning MusicGen with LoRA

## 1. Objective
The main goal of this experiment was to test how to fine-tune MusicGen (Facebook's AI music model) without needing massive computing power. Specifically, I wanted to fine-tune the `facebook/musicgen-small` model on realistic, high-quality traditional Hindustani classical instrumental music. 

I wanted to generate three specific types of music:
* **Solo Percussion:** Fast and detailed modern tabla beats with clear sound.
* **Melodic Duets:** Traditional raga style featuring a slow, emotional sitar and a steady tabla beat.
* **Grand Ensembles:** A rich mix of string (veena and sitar) and percussion (tabla) instruments for a majestic traditional performance.

---

## 3. Methodology and Fine-Tuning Approach through LoRA

Used LoRA (Low-Rank Adaptation) instead of doing a full fine-tune, since it would have taken too much computing power and could have led to catastrophic forgetting.

* **Target Layers:** Instead of retraining the whole model, training was done on a small set of new weights (adapters) added to specific parts of the model (the query and value layers in the attention mechanism). 
* **Mechanics:** The original model weights stayed the same (frozen), while the adapter used lower ranked matrices to learn the newly introduced Hindustani music styles without breaking down the core of the base model.
* **Hardware Optimisation:** Introduced an optimisation method using Gradient Accumulation to resolve hardware constraints, simulating a larger batch configuration through a sequential multi-step backward pass.

---

## 4. Errors, Crashes, and Fixes

### Hardware Constraint: CUDA Out of Memory (OOM)
- Changed the batch size to `1` so the GPU only had to process one audio file at a time, with `accumulation_steps = 4`, saving the gradients for 4 steps before updating the model
- This simulated a model running with `batch_size=4` without extra memory

### Bad Audio Quality: Overfitting and Exploding Weights
- During an early run, the model completely stopped making music and just gave output as loud static garbage.
- This might have been because the learning rate (1e-4) was too high, causing the new weights to change too quickly and breaking the model's pre-trained base weights.
- Also, LoRA Alpha being 32 made the fine-tuning step overpower the base model.
- Finally, training for 5 epochs made the fine-tuned local model overfit the small dataset instead of imitating the abstract style of the samples provided
* **The Fix:**
  1. **Lowered the Learning Rate:** Dropped the learning rate to a smaller value of 2e-5
  2. **Balanced the Alpha:** Lowered the alpha to 16 to match the rank (16), leading to a stable 1:1 ratio
  3. **Reduced Epochs:** Stopped the training early at 3 epochs instead of 5, successfully saving the weights before overfit

---

## 5. Learnings

* **Audio Models are Sensitive:** Autoregressive audio models behave much more sensitively during fine-tuning since they generate high-dimensional acoustic waveforms via token vectors; minor weight distortions easily manifest as severe audible noise.
* **Hyperparameter tuning:** The ratio between parameters matters just as much as the numbers themselves. In LoRA, if the alpha is too high compared to the rank, the new training completely overpowers the base model and ruins the output.
* **Dealing with Hardware Limits:** Optimisations like Gradient Accumulation let you run heavy, complex tasks on standard computer hardware.


# Fine-Tuning Experiments Log: Parameter-Efficient Fine-Tuning to MusicGen through LoRA

## 1. Objective
The primary goal of the fine-tuning experiment was to explore parameter-efficient style adaptation on MusicGen, Facebook's state-of-the-art generative audio model. Specifically, the experiment focused on conditioning 'facebook/musicgen-small' to synthesise authentic, high-quality traditional Hindustani classical instrumental music. 

The targeted outputs focus on three distinct structural archetypes:
* **Solo Percussion:** Intricate, fast-paced modern tabla dynamics with clear, resonant acoustic properties.
* **Melodic Duets:** Traditional raga-styled interactions between a slow, expressive sitar melody and a steady tabla accompaniment.
* **Grand Instrumental Ensembles:** A rich, multi-instrumental layer blending stringed instruments (veena and sitar) and percussion (tabla) to build a majestic, traditional performance.

---

## 2. Methodology and Fine-Tuning Approach through LoRA

LoRA (Low-Rank Adaptation) was implemented to avoid executing a full-parameter fine-tuning routine, which is computationally prohibitive and risks catastrophic forgetting.
* **Target Layers:** The trainable adapter matrices are injected exclusively into the attention mechanism's query and value projection layers (`q_proj`, `v_proj`). 
* **Mechanics:** The base model weights remain completely frozen. The adapter learns low-rank decomposition matrices to capture localised stylistic variations without destabilising the foundational audio synthesis pipeline.
* **Hardware Optimisation Strategy** I introduced an optimisation pipeline using Gradient Accumulation to decouple hardware constraints, simulating a larger batch configuration through a sequential multi-step backward pass.

---

## 3. Detailed Experimental Runs & Observations

The table below catalogs the progression of the training configurations and their corresponding audio outcomes.

| Run ID | Hyperparameters | Intent / Strategy | Quantitative & Qualitative Observations |
| :--- | :--- | :--- | :--- |
| **01 (Baseline)** | Pre-trained Weights Only | Establish a control variable using zero-shot text-to-audio prompting. | Generated audio was clean but structurally generic. It heavily favored Western acoustic instruments or synthesized variations, struggling to capture authentic Indian classical textures or complex rhythm cycles. |
| **02 (Run Alpha)** | Epochs: 5<br>Batch Size: 2<br>Learning Rate: 1e-4<br>LoRA Rank (r): 16<br>LoRA Alpha: 32 | Aggressive fine-tuning to quickly map the model's latent space to local custom audio chunks. | **Failure Mode:** Immediate CUDA Out of Memory error during the first epoch. The physical batch size exceeded the VRAM limit. |
| **03 (Run Beta)** | Epochs: 5<br>Batch Size: 1 (Accumulation: 4)<br>Learning Rate: 1e-4<br>LoRA Rank (r): 16<br>LoRA Alpha: 32 | Memory-stabilized training run with high learning rate and strong adapter scaling factor. | **Failure Mode:** Training completed, but inference outputs were heavily corrupted. The generated tracks consisted of garbled electronic static, intense digital clipping, and broken audio artifacts. The model overfit to data noise. |
| **04 (Run Gamma)** | Epochs: 3<br>Batch Size: 1 (Accumulation: 4)<br>Learning Rate: 2e-5<br>LoRA Rank (r): 16<br>LoRA Alpha: 16 | Surgical adjustment: reduced learning rate, balanced the alpha ratio to 1:1, and capped epochs to prioritize generalization. | **Success Mode:** Phenomenal acoustic clarity. The model cleanly synthesized the distinct metallic resonance of sitar strings, maintained a steady rhythmic pattern in the tabla solo, and managed complex multi-instrument separation in the ensemble. |

---

## 4. Failure Modes, Errors, and Systematic Fixes

### Hardware Constraint: CUDA Out of Memory (OOM)
* **The Problem:** Processing high-resolution audio chunks (32 kHz) through a multi-stream autoregressive decoder creates massive token dimensions. Setting a standard batch size of 2 caused the memory footprint to instantly spike past the 15GB threshold.
* **The Systematic Fix:** 1. Set the physical data loader batch size to `1` to isolate the memory footprint of a single sequence.
  2. Implemented `accumulation_steps = 4` within the training loop. Gradients are retained over 4 forward passes before executing a single optimizer update, simulating a stable batch size of 4.
  3. Forced garbage collection and manual cache clearing before runtime.

### Audio Degradation: Weight Explosion & Overfitting
* **The Problem:** In Run Beta, the model completely lost its ability to synthesize melodic structures, producing harsh noise. This occurred because a learning rate of 1e-4 was too aggressive, causing the adapter gradients to shatter the fragile pre-trained structural weights. Furthermore, setting an alpha of 32 with a rank of 16 forced the adapter to effectively "shout" over the foundational layers, while 5 epochs caused the model to rigidly memorize the limited dataset instead of abstracting style.
* **The Systematic Fix:**
  1. **Softened Learning Rate:** Dropped the learning rate down to a conservative step size of 2e-5, allowing for micro-adjustments.
  2. **Balanced Adapter Scaling:** Lowered the alpha to 16, enforcing a stable 1:1 ratio with the rank (16) to blend new acoustic textures seamlessly with the base model.
  3. **Early Training Termination:** Reduced total epochs from 5 to 3, successfully saving the weights right at the optimal point of generalized stylistic learning.

---

## 5. Key Academic Learnings

* **Generative Audio Fragility:** Autoregressive audio models behave much more sensitively during fine-tuning than standard text-based causal language models. Because they generate high-dimensional acoustic waveforms via token vectors, minor weight distortions easily manifest as severe audible noise.
* **The Hyperparameter Balancing Act:** Ratios are just as critical as absolute numbers. In LoRA, setting an alpha that vastly outpaces the rank acts as an amplifier that can easily overpower a model's base intelligence. Scaling these parameters carefully is paramount.
* **The Value of Creative Engineering Constraints:** Hardware limitations do not stop machine learning exploration. Techniques like gradient accumulation prove that smart, methodical pipeline design can achieve large-scale optimization stabilities on standard consumer infrastructure. 
* **The Beauty of Iteration:** Experiencing corrupted output is not a failure; it is an invaluable step that provides the exact diagnostic data required to decode a neural network's behavior and achieve a high-performing final system.

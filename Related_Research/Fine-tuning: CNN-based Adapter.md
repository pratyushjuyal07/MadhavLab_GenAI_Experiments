# Analysing CNN-based Adapter attempts in fine-tuning MusicGen

[Exploring Adapter Design Tradeoffs for Low-Resource Music Generation
-A Mehta, S Chauhan, M Choudhury - Proceedings of the 33rd ACM International Conference …, 2025](https://drive.google.com/file/d/1wqiPkZ_SlvHkL3-X0cOZku_4t9Ej5LvK/view?usp=sharing)
## 1. Process Flow
- Raw input is converted to dimensionality [T,1024], T= 'time dimension', and 1024 = 'Hidden Dimensionality of MusicGen'
  - `Input = Audio for fine-tuning` => `Converted to audio tokens through EnCodec` => `Converted into feature vectors of [T,1024]`
- These feature vectors are passed through the CNN-based framework:
  - `The _Down-Projection Layer_ converts [T,1024] to [T,64] to compress the feature data while preserving time sequence length`
  - `A _Residual Bottleneck_ module captures localised info for all 64 channels through sliding windows, and changes feature vectors`
  - `The input feature vectors are added to these transformed feature vectors to preserve broader musical information`
  - `These transformed feature vectors are passed to a _Squeeze & Excitation_ block to overcome CNN's localised shortcomings`
  - 
  

---

## 2. Elaborated Step-by-Step Technical Breakdown

### Step 1: Audio Ingestion and Tokenization (EnCodec)
* **The Input:** The input is a raw audio waveform (e.g., a `.wav` file of a vocal performance or an instrument like a sitar performing complex pitch-glides).
* **The Mechanism:** Large language models for audio cannot directly process raw waveforms because audio samples typically hit 32,000 to 48,000 values per second. MusicGen uses Meta’s **EnCodec**, a neural audio compression model operating as an audio tokenizer.
* **The Transformation:** EnCodec uses a multi-stage **Residual Vector Quantization (RVQ)** process. It downsamples the raw continuous waveform into discrete integer streams (tokens) across a temporal grid. For every slice of time, it creates a small stack of discrete numbers representing the acoustic properties of that moment.

### Step 2: Dense Embedding and Dimensionality Formulation
* **The Mechanism:** The discrete tokens output by EnCodec are mapped through a trainable lookup table known as the **Embedding Layer**.
* **The Feature Vector:** This layer transforms the categorical integer tokens into a high-dimensional continuous matrix of **hidden states**. 
* **Mathematical Representation:** The output matrix shape is generalized as:
    $$\mathbf{X}_{	ext{in}} \in \mathbb{R}^{T 	imes D_{	ext{model}}}$$
    Where:
    * $T$ is the total sequence length (the number of discrete steps along the time dimension).
    * $D_{	ext{model}}$ is the core hidden dimension of MusicGen (e.g., $1024$ hidden features per time step).

### Step 3: Down-Projection Convolutional Layer
* **The Split:** Once the continuous representation $\mathbf{X}_{	ext{in}}$ reaches a Target Transformer Block, the main model parameters are **frozen** (gradients are disabled). The matrix splits, sending a copy into the **CNN Adapter**.
* **The Mechanism:** The adapter initiates with a 1D Convolutional layer characterized by a specific parameter matrix:
    $$\mathbf{W}_{	ext{down}} \in \mathbb{R}^{K 	imes D_{	ext{model}} 	imes D_{	ext{bottleneck}}}$$
    Where $K$ represents the kernel (window) size, and $D_{	ext{bottleneck}}$ represents the reduced dimension (e.g., $64$).
* **The Operation:** As the convolution kernel slides strictly across the time axis ($T$), it computes a weighted summation over the local context. Because the stride is set to $1$ and appropriate padding is applied, the time dimension ($T$) is completely preserved.
* **The Output:** The feature space is compressed. The matrix transforms from $[T 	imes 1024]$ down to:
    $$\mathbf{X}_{	ext{down}} \in \mathbb{R}^{T 	imes 64}$$
    This drastically minimizes the parameter footprint of the layers that follow.

### Step 4: Deep Residual Bottleneck Module
* **The Concept:** Operating in the compressed $D_{	ext{bottleneck}}$ space, this module consists of a series of stacked **Residual Blocks**.
* **The Structural Mechanics:** A standard layer maps input $x$ to $F(x)$. A residual block introduces a **skip connection** or shortcut that executes an element-wise addition:
    $$	ext{Output} = F(x) + x$$
* **Why it Matters for Music:** Complex ornamentations like *gamakas* (oscillations) and *murkis* (microtonal turns) are high-frequency, localized variations. The inner convolution layers ($F(x)$) use small, localized sliding windows to learn to track these ultra-fast microtonal variations. Meanwhile, the skip connection ($+ x$) passes the broader underlying melodic contour forward without altering it. This dual-path system enables the network to track both short-term dependencies (ornaments) and long-term dependencies (melodic themes) simultaneously.

### Step 5: Squeeze-and-Excitation (SE) Block
* **The Channels:** In convolutional operations, each individual dimension within the feature size (our 64 compressed dimensions) acts as a **channel**. Individual channels naturally train themselves to look for specific acoustic signatures (e.g., one channel tracks pitch vibrato, another tracks sudden noise transience).
* **The Squeeze Phase:** CNNs inherently only look at local neighborhoods. To provide global structural context, a **Global Average Pooling (GAP)** operation is run over the time axis $T$. It averages all values along the timeline into a single static representation vector:
    $$\mathbf{z} \in \mathbb{R}^{1 	imes 64}$$
    The temporal dimension is "squeezed" out, leaving a pure summary of channel activations.
* **The Excitation Phase:** This vector $\mathbf{z}$ is passed through a tiny, non-linear bottleneck network parameterized by two weight matrices ($\mathbf{W}_1$ and $\mathbf{W}_2$):
    $$\mathbf{s} = \sigma(\mathbf{W}_2 \cdot 	ext{ReLU}(\mathbf{W}_1 \cdot \mathbf{z}))$$
    Where $\sigma$ is the Sigmoid function, mapping the output vectors strictly between $0$ and $1$.
* **The Scaling Application:** This output vector $\mathbf{s}$ acts as a scale score for each feature channel. The original sequence matrix $[T 	imes 64]$ is multiplied element-wise by these scales. If a vocal ornamentation is actively happening, the block dynamically amplifies the weights of channels tracking microtonal glides while muting irrelevant noise channels.

### Step 6: Up-Projection Convolutional Layer
* **The Problem:** The modified hidden states are still trapped inside the bottleneck layout $[T 	imes 64]$. The frozen MusicGen layer waiting downstream strictly expects its native feature dimensionality of $[T 	imes 1024]$. Passing the matrix as-is would throw a matrix mismatch error.
* **The Mechanism:** The data encounters an Up-Projection 1D Convolutional layer built on the trainable weight matrix:
    $$\mathbf{W}_{	ext{up}} \in \mathbb{R}^{K 	imes D_{	ext{bottleneck}} 	imes D_{	ext{model}}}$$
* **The Output:** This performs the exact mathematical inverse of Step 3. It projects the 64 channels back up to the native size, creating:
    $$\mathbf{X}_{	ext{adapter\_out}} \in \mathbb{R}^{T 	imes 1024}$$
    This matches the native layout exactly, guaranteeing minimal disruption when reintegrated.

### Step 7: Element-Wise Recombination and Backpropagation
* **The Recombination:** The final output of the adapter is recombined with the frozen backbone's computational branch through a simple residual addition step inside the block:
    $$\mathbf{X}_{	ext{final\_block\_out}} = \mathbf{X}_{	ext{frozen\_transformer\_out}} + \mathbf{X}_{	ext{adapter\_out}}$$
* **Autoregressive Loss Computation:** MusicGen predicts audio tokens step-by-step. The final output vectors pass through the language modeling head to yield a probability distribution over the vocabulary of audio tokens. The system calculates a **Cross-Entropy Loss** by comparing its predicted token distributions against the actual next token in the ground-truth training clip.
* **Backpropagation:** During the backward training pass, gradients are computed across the entire network graph. However, because the primary MusicGen layers are frozen, optimization algorithms (like AdamW) only apply parameter updates to the active weight matrices:
    $$\Delta \mathbf{W}_{	ext{down}}, \Delta \mathbf{W}_{	ext{residual\_blocks}}, \Delta \mathbf{W}_1, \Delta \mathbf{W}_2, 	ext{ and } \Delta \mathbf{W}_{	ext{up}}$$
    Over thousands of training iterations, these matrices refine their weights to systematically isolate, parse, and recreate nuanced musical expressions.

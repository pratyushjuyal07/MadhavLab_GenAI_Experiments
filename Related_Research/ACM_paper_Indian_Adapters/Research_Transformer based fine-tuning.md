# Analysing Transformer-based Adapters in fine-tuning attempts of MusicGen

[Exploring Adapter Design Tradeoffs for Low-Resource Music Generation
-A Mehta, S Chauhan, M Choudhury - Proceedings of the 33rd ACM International Conference …, 2025](https://drive.google.com/file/d/1wqiPkZ_SlvHkL3-X0cOZku_4t9Ej5LvK/view?usp=sharing)

## Process Flow
- Raw input is converted to dimensionality [T,1024], T= 'time dimension', and 1024 = 'Hidden Dimensionality of MusicGen'
  - `Input = Audio for fine-tuning` => `Converted to audio tokens through EnCodec` => `Converted into feature vectors of [T,1024]`
- These feature vectors are passed through the Transformer-based framework:
  - `The _Down-Projection Layer_ converts [T,1024] to [T,64] to compress features into a lower dimensional bottleneck space`
  - `A _Layer Normalisation_ step scales the feature data to stabilize the signals before processing`
  - `A _Multi-Head Self-Attention_ module calculates all-to-all relationships across the timeline to capture global context`
  - `A second _Layer Normalisation_ step handles training stability right after the attention mechanism`
  - `A _Feed-Forward Network (FFN)_ uses GELU activation to independently refine features at each individual step`
  - `The _Up-Projection Layer_ restores dimensionality by converting [T,64] vectors back to [T,1024] to match transformer dimensions`
- These new feature vectors extracted from the training audio are then added to MusicGen's original transformer matrices
- Autoregressive Loss: by comparing the next ground-truth token to the next token predicted based on the updated model weights
- Backpropagation: Parameters are adjusted using algorithms like AdamW, applying updates only to the adapter matrices

---

### Step 1: Ingestion, Tokenisation, and Embedding
The training process begins with raw audio clips chosen for fine-tuning. The audio is passed through Encodec and mapped through an embedding layer, which converts it into continuous feature vectors. This data structure has a specific layout: a Time Dimension (representing the total duration of the clip) and a Hidden Dimensionality of 1024 (which is the standard feature of the MusicGen Transformer).

### Step 2: The Down-Projection Layer
Once the feature vectors enter a target block inside MusicGen, the primary model remains frozen, and a copy of the data splits off into the Transformer-based adapter. The first component it hits is the Down-Projection Layer. This layer compresses the high-dimensional feature data, scaling down the hidden feature size from 1024 dimensions down to just 64 bottleneck channels, which heavily minimizes the parameter footprint of the subsequent internal layers.

### Step 3: Multi-Head Self-Attention Module
Next, the compressed data passes through an initial Layer Normalisation block to stabilize training scales before entering the Multi-Head Self-Attention module. Instead of looking through a small sliding window, this module evaluates global relationships across the entire sequence length simultaneously. This global viewpoint allows the network to track overarching arrangements, which is vital for preserving long-form musical structure and composition coherence.

### Step 4: The Feed-Forward Network
* First, a second Layer Normalisation block is applied directly after the attention mechanism to keep processing smooth.
* Second, the data enters a Feed-Forward Network which uses GELU activation to refine the learned representations. This block acts on each time step individually to polish and synthesize features, while a residual skip connection incorporates the input features back into the path to ensure robustness.

### Step 5: The Up-Projection Layer
At this stage, the feature vectors contain highly detailed information about the musical style in the compressed 64-channel bottleneck format. The Up-Projection Layer applies a final matrix expansion that restores the original sequence dimensions, converting the feature vectors from 64 channels back up to 1024 channels so they match the main transformer's expected dimensions.

### Step 6: Recombination, Autoregressive Loss and Backpropagation
The newly extracted feature vectors pass through dropout regularization for safety and are directly added to MusicGen's original, frozen transformer matrices. The system computes an Autoregressive Loss by directly comparing the next token predicted by the updated weights against the actual ground-truth token from the training audio. During backpropagation, an optimisation algorithm like AdamW calculates the error and makes parameter adjustments and weight updates that are applied exclusively to the adapter matrices, making the training fast and efficient.

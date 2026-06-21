# Analysing CNN-based Adapter attempts in fine-tuning MusicGen

[Exploring Adapter Design Tradeoffs for Low-Resource Music Generation
-A Mehta, S Chauhan, M Choudhury - Proceedings of the 33rd ACM International Conference …, 2025](https://drive.google.com/file/d/1wqiPkZ_SlvHkL3-X0cOZku_4t9Ej5LvK/view?usp=sharing)
## 1. Process Flow
- Raw input is converted to dimensionality [T,1024], T= 'time dimension', and 1024 = 'Hidden Dimensionality of MusicGen'
  - `Input = Audio for fine-tuning` => `Converted to audio tokens through EnCodec` => `Converted into feature vectors of [T,1024]`
- These feature vectors are passed through the CNN-based framework:
  - `The _Down-Projection Layer_ converts [T,1024] to [T,64] to compress the feature data while preserving time sequence length`
  - `A _Residual Bottleneck_ module captures local info for all 64 channels through sliding windows, and changes feature vectors`
  - `The input feature vectors are added to these transformed feature vectors to preserve broader musical information`
  - `These transformed feature vectors are passed to a _Squeeze & Excitation_ block to overcome CNN's localised shortcomings`
  - `The _Squeeze_ block averages the global feature vectors for every channel, over all time, obtaining a vector of [1,64]`
  - `The _Excitation_ block scales the original [T,64] vectors based on this average [1,64] factor, exciting weights globally`
  - `The _Up-Projection Layer_ restores dimensionality by converting [T,64] vectors to [T,1024] to match transformer dimensions`
- These new feature vectors extracted from the training audio are then added to MusicGen's original transformer matrices
- Autoregressive Loss: by comparing the next ground-truth token to the next token predicted based on the updated model weights
- Backpropagation: Parameters are adjusted using algorithms like AdamW, applying updates only to the adapter matrices
  

---

# Process Flow and Step-by-Step Explanation: CNN-Based Adapter for MusicGen

## 1. Process Flow

* **Input Data Ingestion:** Fine-tuning audio -> EnCodec Tokenizer -> Dense Feature Vectors (Time Dimension, 1024 Hidden Dimensions)
* **Adapter Split:** Data forks into the trainable CNN adapter pipeline.
* **Down-Projection Layer:** Compresses features from 1024 down to 64 channels while keeping the exact same time sequence length.
* **Deep Residual Bottleneck:** Uses small sliding windows to capture local pitch details across the 64 channels.
* **Residual Skip Connection:** Adds the original input features back to the transformed features to preserve the overarching macro-melody.
* **Squeeze Phase:** Averages all feature data across the entire timeline to create a single global summary for each of the 64 channels.
* **Excitation Phase:** Uses that global summary to calculate importance scores, dynamically multiplying and scaling the original features.
* **Up-Projection Layer:** Expands the compressed 64 channels back up to the original 1024 dimensions.
* **Transformer Integration:** Adds the newly extracted adapter features directly into MusicGen's native transformer matrices.
* **Autoregressive Loss & Backpropagation:** Evaluates the next-token prediction against the real training audio and updates only the adapter weights via AdamW.

---

## 2. Step-by-Step Technical Explanation

### Step 1: Ingestion, Tokenization, and Embedding
The training process begins with raw audio clips chosen for fine-tuning, such as recordings containing specific vocal or instrumental decorations. Because AI models cannot easily read continuous raw sound waves, the audio is passed through a tool called EnCodec. EnCodec acts like an audio tokenizer, slicing the sound into short pieces and turning them into a grid of discrete numbers (audio tokens). These tokens are then mapped through an embedding layer, which converts them into continuous feature vectors. This data structure has a specific layout: a Time Dimension (representing the total duration of the clip) and a Hidden Dimensionality of 1024 (which is the standard feature width that MusicGen uses to understand audio properties).

### Step 2: The Down-Projection Layer
Once the feature vectors enter a target block inside MusicGen, the primary model remains frozen, and a copy of the data splits off into the CNN-based adapter. The first component it hits is the Down-Projection Layer. This layer uses a one-dimensional convolution to compress the data, scaling down the hidden feature size from 1024 dimensions down to just 64 channels. It uses a specific padding setup so that even though the internal thickness of the data is shrunk to make calculations lightweight, the time sequence length is preserved perfectly.

### Step 3: The Residual Bottleneck Module
Next, the compressed data passes into the Residual Bottleneck module. This module uses small convolutional sliding windows that look at a few adjacent steps in time. This localized focus allows the network to capture ultra-fine, fast-moving details like quick pitch oscillations or sudden glides across all 64 channels. Crucially, a residual skip connection is used here: the original input feature vectors entering this module are added back to the newly transformed feature vectors. This step ensures that while the convolutions focus on catching the micro-details, the broader, long-term musical information and main melody line are not erased.

### Step 4: The Squeeze and Excitation Block
Standard convolutional layers have a structural shortcoming: they only look at narrow, local windows of time and miss the bigger picture. To fix this, the features are passed into a Squeeze and Excitation block. 
* First, the Squeeze block takes the feature vectors across the entire timeline and averages them out. This collapses the time axis entirely, leaving a single global summary factor for each of the 64 channels. 
* Second, the Excitation block takes this summary factor and uses it to dynamically scale the original features. It calculates an importance score for each channel and multiplies the features by it, globally exciting the weights of the most relevant musical features while dialing down the noise.



### Step 5: The Up-Projection Layer
At this stage, the feature vectors contain highly detailed information about the musical style, but they are still trapped in the compressed 64-channel bottleneck format. Because the downstream layers of the main MusicGen model strictly expect the original 1024 hidden dimensions, passing the compressed data as-is would break the code. The Up-Projection Layer solves this by applying another convolution that restores the dimensionality, converting the feature vectors from 64 channels back up to 1024 channels so they match the transformer's expected dimensions perfectly.

### Step 6: Integration and Recombination
The newly extracted feature vectors, which now hold the mathematical essence of the target musical ornamentations, are ready to be integrated. They are directly added to MusicGen's original, frozen transformer matrices. This allows the pre-trained model to keep its vast, existing knowledge about general music generation while seamlessly blending in the specific stylistic modifications provided by the adapter path.

### Step 7: Autoregressive Loss and Backpropagation
Finally, the model calculates how well it is learning. Since MusicGen is an autoregressive model, it predicts music step-by-step, trying to guess what the next slice of audio should be. The system computes an Autoregressive Loss by directly comparing the next token predicted by the updated weights against the actual, ground-truth token from the training audio. During backpropagation, an optimization algorithm like AdamW calculates the error and marches backward through the network. Because the main MusicGen blocks are locked down, these parameter adjustments and weight updates are applied exclusively to the adapter matrices, making the training fast and efficient.

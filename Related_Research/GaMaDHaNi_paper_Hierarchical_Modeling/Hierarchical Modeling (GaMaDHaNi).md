# Analysing GaMaDHaNi: Hierarchical Generative Modeling for Hindustani Classical Vocal Contours

[Hierarchical Generative Modeling of Melodic Vocal Contours in Hindustani Classical Music
-N Shikarpur, K M Dendukuri, Y Wu, A Caillon, C Z A Huang - Proceedings of the 25th Int. Society for Music Information Retrieval Conf., 2024](https://arxiv.org/abs/2408.12658)

## Objective of the Model
- Traditional discrete notations and symbolic representations fail to faithfully capture the smooth, fluid nuances of vocal execution
- Prior computational metrics prove that tracking the fundamental frequency contour acts as a reliable proxy for musical expressiveness
- The system establishes a modular, hierarchical architecture designed specifically to manage the high intricacy of classical singing
- Recognise the system as a foundational proof-of-concept, setting a structural baseline while leaving out high-level theoretical logic



### Pitch Generator

#### Process Flow: Pitch Sequence Processing & Autoregressive Generation
- `Raw Audio Input` => `100Hz Pitch Tracker` => `Continuous Frequencies (86Hz - 899Hz)`
- `Continuous Frequencies` => `10-Cent Quantizer` => `Discrete Pitch Tokens (Vocabulary V)`
- `Discrete Pitch Tokens` => `Embedding Matrix E` => `Latent Feature Vectors [Time, d]`
- `Latent Feature Vectors` => `Decoder-Only Transformer` => `Predicted Token Logits`
- `Predicted Token Logits vs Ground Truth` => `Cross-Entropy Loss` => `Weight Adjustments`

---

#### Technical Breakdown: The Pitch Generator

The Pitch Generator acts as the foundational layer of GaMaDHaNi's hierarchy. Instead of forcing the model to calculate complex audio waveforms or text characters directly, its sole responsibility is to synthesize a highly detailed, stylistically accurate melodic blueprint (the pitch contour) completely from scratch. 

##### 1. Data Representation Strategy
To model the fluid, gliding characteristics of Hindustani vocals, the raw acoustic data is transformed into ultra-precise structural tokens:
* **High Temporal Density:** The model evaluates pitch at 100Hz (one state every 10 milliseconds). This high sampling frequency prevents the model from missing brief, rapid vocal inflections like *murkis* or *taans*.
* **Microtonal Vocabulary ($V$):** Rather than using standard semitones, the frequency spectrum is sliced into **10-cent intervals** (10 bins per piano key step). These bins form a customized dictionary where each microtonal step represents an independent token, preserving the ability to map complex glides (*meends*).

##### 2. The Autoregressive Engine
For its primary execution mode, the system treats pitch synthesis similarly to text-generation models (like an LLM predicting words):
* **Decoder-Only Transformer Topology:** The network uses causal self-attention mechanisms. It scans the history of generated pitch states to predict the exact probability distribution of the next microtonal bin.
* **Continuous Feature Mapping:** The discrete tokens pass through an embedding matrix ($E \in \mathbb{R}^{|V| \times d}$), translating abstract bin indexes into dense vectors of dimension $d$ so the Transformer can mathematically weigh melodic trajectories.
* **Loss Profile:** The network optimization uses standard cross-entropy loss, comparing its token distribution arrays against actual recorded data to ensure the generation aligns with the authentic phrasing rules of the genre.

### Spectrogram Generator

#### Process Flow: Pitch-Conditioned Audio Synthesis & Diffusion Generation
* **Step 1:** `Generated Pitch Contour (100Hz)` + `Singer ID` $\rightarrow$ **Conditioning Preprocessor** $\rightarrow$ `Downsampled Pitch & Singer Embeddings (62.5Hz)`
* **Step 2:** `Downsampled Conditioning Channels` + `Mel-Spectrogram State ($x_{\alpha_t}$)` $\rightarrow$ **1-D Convolutional U-Net** $\rightarrow$ `Predicted Velocity / Derivative State`
* **Step 3:** `Predicted Velocity State` $\rightarrow$ **Iterative $\alpha$-Deblending (IADB)** $\rightarrow$ `Refined Mel-Spectrogram (192 Mels, 62.5Hz)`
* **Step 4:** `Refined Mel-Spectrogram` $\rightarrow$ **Griffin-Lim / Vocoder** $\rightarrow$ `Synthesized Vocal Audio Waveform (16kHz)`

---

#### Technical Breakdown: The Spectrogram Generator

The Spectrogram Generator forms the second level of GaMaDHaNi’s modular hierarchy. Its primary responsibility is to translate the abstract, fine-grained melodic blueprint (the pitch contour) into a highly detailed acoustic representation (a mel-spectrogram) while preserving singer-specific vocal qualities.

##### 1. Data Representation & Conditioning Strategy
To bridge the gap between abstract pitch lines and audible singing, the module couples multi-source context variables directly with the spectral space:
* **Multi-Modal Conditioning Alignment:** The model evaluates pitch and spectral features over 8.2-second windows (512 spectral frames). The original 100Hz pitch tracking sequence is linearly interpolated and downsampled to 62.5Hz to seamlessly match the temporal axis of the acoustic data. Concurrently, 56 unique singer identity IDs are projected into dense representation vectors ($d_{\text{singer}} = 128$). Both the downsampled pitch and singer embeddings are appended as additional feature channels alongside the mel-spectrogram input block.
* **High-Resolution Spectral Target:** The target audio space uses 16kHz waveforms processed into 192 mel-frequency bins with a hop length of 256 samples (giving a 16 millisecond frame step / 62.5Hz temporal density). To stabilize training and enhance diffusion matching, raw spectral intensities are mapped into a continuous Gaussian distribution using a quantile transform function.

##### 2. The Diffusion Engine & Guidance
For its generative pipeline, the system framework avoids stochastic diffusion formulations in favor of structural trajectory mapping:
* **Conditional U-Net Topology:** The network employs a 1-D convolutional U-Net structure featuring three downsampling and three upsampling blocks with down/up factors of 4, 2, and 2. Inside each block, four 1-D convolutional operators leverage weight normalization and non-monotonic Mish activation operations. The internal bottleneck layer acts as a sequence matcher, using 4 self-attention blocks configured with 8 heads each to stabilize structural patterns.
* **Iterative $\alpha$-Deblending (IADB) with Classifier-Free Guidance (CFG):** The generator learns deterministic linear interpolation trajectories between native noise trajectories and the target spectrogram data point. During sampling generation, strict adherence to the target melody and voice identity is enforced via Classifier-Free Guidance with a conditioning scalar weight of $w = 3$. The fully deblended, multi-channel mel-spectrogram is finally routed through a vocoder (such as the Griffin-Lim algorithm or high-fidelity models like HiFi-GAN) to synthesize the audible 16kHz vocal audio waveform.

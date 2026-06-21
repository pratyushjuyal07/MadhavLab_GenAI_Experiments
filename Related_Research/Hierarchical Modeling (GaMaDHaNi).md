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
- `Predicted Token Logits vs Ground Truth` => `Cross-Entropy Loss` => `Weight Adjustments ($\theta$)`

---

#### Technical Breakdown: The Pitch Generator ($p_\theta$)

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

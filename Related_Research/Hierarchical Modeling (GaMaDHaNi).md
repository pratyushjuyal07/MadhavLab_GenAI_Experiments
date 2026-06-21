# Analysing GaMaDHaNi: Hierarchical Generative Modeling for Hindustani Classical Vocal Contours

[Hierarchical Generative Modeling of Melodic Vocal Contours in Hindustani Classical Music
- N Shikarpur, K M Dendukuri, Y Wu, A Caillon, C Z A Huang - Proceedings of the 25th International Society for Music Information Retrieval Conference, 2024](https://arxiv.org/abs/2408.12658)

## Objective of the Model
- Traditional discrete notations and symbolic representations fail to faithfully capture the smooth, fluid nuances of vocal execution
  - `Standard music files` => `Lead sheets, MIDI grids, or textbook notation` => `Miss fine-grained microtonal slides, ornaments, and glides`
- Prior computational metrics prove that tracking the fundamental frequency contour acts as a reliable proxy for musical expressiveness
  - `Melodic blueprint extraction` => `Isolate the raw pitch contour over time` => `Provides an intermediate playground for complex generation`
- The system establishes a modular, hierarchical architecture designed specifically to manage the high intricacy of classical singing
  - `The core pipeline layout` => `Create a two-tier framework split into an independent Pitch Generator and a Spectrogram Generator`
  - `Data optimization target` => `Train a robust model capable of learning diverse melodic ideas using a limited 120-hour dataset`
  - `Interactive use-case design` => `Incorporate steerable options like continuing a short audio prompt or pathing along a user-guided pitch path`
- Recognize the system as a foundational proof-of-concept, setting a structural baseline while leaving out high-level theoretical logic
  - `System constraints` => `Operates without explicitly encoding native framework rules like tonic base frequency, raga modes, or tala meters`

# Fine-Tuning MusicGen for Hindustani Classical Instrumental Style Adaptation

## 1. Objective
The primary goal of this project was to evaluate parameter-efficient style adaptation on MusicGen, an autoregressive transformer-based decoder designed for text-to-audio generation. The target domain focused on conditioning the model to synthesise authentic traditional Hindustani classical instrumental music across three distinct structural archetypes:

* **Solo Percussion:** Intricate, fast-paced modern tabla dynamics emphasising rhythmic precision and authentic acoustic resonance.
* **Melodic Duets:** Traditional raga-styled interactions featuring a slow, expressive sitar melody accompanied by a steady tabla rhythm.
* **Grand Instrumental Ensembles:** A layered, multi-instrumental arrangement blending stringed instruments (veena and sitar) with percussion (tabla) to simulate a full traditional performance.

---

## 2. Model Architecture
The project utilised **`facebook/musicgen-small`**, a 300M parameter architecture. The model leverages an EnCodec neural audio compressor to map continuous acoustic waveforms into discrete multi-stream token spaces, coupled with an autoregressive language model decoder for conditional sequence generation. All synthesis passes operate natively at a 32kHz sampling rate.

---

## 3. Approach & Methodology

### Dataset Selection and Preparation
Sourcing appropriate data required navigating limitations in standard open-source audio datasets:
* Standard machine learning audio datasets (e.g., NSynth) completely lacked representations of classical Indian instruments.
* Large-scale domain repositories (e.g., Saraga) proved too massive for localised scaling, and the available tracks predominantly featured vocal components. Mixed vocal frequencies disrupt the fine-tuning layers when targeting pure instrumental synthesis.

To isolate clean instrumental profiles, targeted audio tracking was conducted via Google's AudioSet research repository. Five high-fidelity source tracks (ranging from 2 to 10 minutes) were compiled, featuring raga compositions (such as Raga Rageshri and Raga Bageshri) performed by master musicians including Pandit Ravi Shankar and Shubhendra Rao:
* **audio1.mp3 (8 mins):** Sitar performance by Shubhendra Rao.
* **audio2.mp3 (8 mins):** Sitar, Veena, and Tabla arrangement by Ravi Shankar.
* **audio3.mp3 (10 mins):** Pure Tabla solo focusing on rhythmic structures and speed variants.
* **audio4.mp3 (8 mins):** Sitar and Tabla duet in Raga Rageshri.
* **audio5.mp3 (2 mins):** Sitar and Tabla accompaniment loop.

The processing pipeline resampled the raw inputs to 32kHz and sliced the continuous audio into uniform 30-second segments, generating 71 training chunks. Unique metadata descriptions were assigned to each chunk to map instrument configurations to individual token boundaries.

### Fine-Tuning Method: Low-Rank Adaptation (LoRA)
To bypass the computational overhead of updating all model layers and to mitigate catastrophic forgetting, Low-Rank Adaptation (LoRA) was integrated:
* **Target Modules:** Trainable low-rank decomposition matrices were injected into the attention mechanism's query and value projection layers (`q_proj`, `v_proj`).
* **State Preservation:** The core transformer weights of the pre-trained model were frozen. Style adaptation was confined to the rank-decomposition pathways.

### Hardware Optimisation Strategy
Training high-dimensional audio token matrices on a single 15GB VRAM GPU container introduced strict physical memory ceilings. To prevent system crashes, Gradient Accumulation was introduced. Setting the hardware `batch_size=1` minimised the active memory footprint, while setting `accumulation_steps=4` deferred weight optimisation until 4 sequential forward/backward passes were completed, successfully simulating a stable batch size of 4.

---

## 4. Conditioning Experiments (Base Model Tests)
Before model alteration, objective benchmarks were run on the unadapted base architecture to analyse the interplay between prompt phrasing, length variables, instrument definitions, and sampling choices. 

The complete documentation of these configurations and comparative results is available in the [Conditioning Experiments Log](./Conditioning_Experiments_Log.md).

---

## 5. Baseline Audio Outputs

### Methodology
Baseline control generations were conducted utilising the raw, unadapted `facebook/musicgen-small` model. Text conditions consisted of three highly detailed prompts designed to capture specific Hindustani classical configurations. Generation parameters were strictly controlled at `guidance_scale=3.0` and `temperature=1.0` to yield 10-second outputs (500 tokens) at 32kHz.

### Baseline Evaluation Table

| Prompt type | Prompt | Duration | Observation | Inference |
| :--- | :--- | :--- | :--- | :--- |
| **Solo Percussion** | A highly detailed solo Indian classical Hindustani modern tabla instrumental, featuring intricate rhythmic patterns, fast-paced beats, and clear, resonant percussion. | 10s | Good instrument recognition, but absence of authentic acoustic tabla patterns. | Pre-trained model lacks specific representations for Indian percussion instruments, substituting them with generic rhythm sequences learned during training. |
| **Melodic Duet** | A traditional Indian Hindustani classical raga-styled instrumental, featuring a slow, expressive sitar melody beautifully accompanied by steady, rhythmic tabla beats. | 10s | Generation of a generic stringed timbre closely resembling a sitar, with the arrangement lacking a defined structure and synchronicity. | Frozen model parameters default to standard harmonic structures when processing localised musical styles, instead of capturing intricacies. |
| **Grand Ensemble** | A grand Indian traditional raga-styled ensemble performance, featuring a rich tapestry of veena, sitar, tabla, and other diverse string and percussion instruments, creating a vibrant rhythm and majestic melody. | 10s | Synthesises a multi-instrument composition with sitar, tabla with a hint of veena, but the raga-styled performance is missing, with isolated instrument sounds. | Text-to-audio decoding resolves multi-instrument prompts by applying standard instrument layering, obscuring specific instrumental harmony and interplay. |

### Final Baseline Observations
The pre-trained model provides exceptional audio stability and clean acoustics for generalised composition. However, a lack of understanding of traditional music is evident, with raga-styled structures not possible, limiting the base model's capability to accurately reconstruct authentic Hindustani classical instrumental interplay, rhythms, and styles.

---

## 6. Fine-Tuning Attempt 1: Failure Analysis

### Methodology
The first fine-tuning attempt applied aggressive optimisation parameters to achieve rapid style transfer. The setup utilised a high learning rate of 1e-4, an extended training duration of 5 epochs, and an elevated adapter scaling factor (`lora_alpha=32`, `r=16`).

### Attempt 1 Evaluation Table

| Prompt type | Prompt | Duration | Observation |
| :--- | :--- | :--- | :--- |
| **Solo Percussion** | A highly detailed solo Indian classical Hindustani modern tabla instrumental... | 10s | Severe acoustic degradation. The output consists of high-frequency squeaks, digital clicks, and continuous static noise with no rhythmic content. |
| **Melodic Duet** | A traditional Indian Hindustani classical raga-styled instrumental... | 10s | Heavily distorted audio dominated by low-frequency hums and digital artifacts.
| **Grand Ensemble** | A grand Indian traditional raga-styled ensemble performance... | 10s | Total synthesis failure producing loud clipping artifacts, continuous digital feedback, and broken audio blocks. |

### Final Attempt 1 Observations
This run highlighted the high sensitivity of autoregressive audio decoders to aggressive parameter adjustments. Setting a learning rate too high or weighting the adapter too heavily overpowers the pre-trained foundation, corrupting the underlying text-to-audio translation matrix and inducing catastrophic generation collapse.

---

## 7. Fine-Tuning Attempt 2: Successful Style Adaptation

### Methodology
The second training iteration introduced stabilised hyperparameters to safeguard foundational weights. The learning rate was reduced to a conservative 2e-5, the LoRA alpha-to-rank ratio was optimised to a balanced 1:1 state (`lora_alpha=16`, `r=16`), and total optimisation duration was constrained to 3 epochs to prevent local data memorisation.

### Attempt 2 Evaluation Table

| Prompt type | Prompt | Duration | Observation |
| :--- | :--- | :--- | :--- |
| **Solo Percussion** | A highly detailed solo Indian classical Hindustani modern tabla instrumental, featuring intricate rhythmic patterns, fast-paced beats, and clear, resonant percussion. | 10s | Substantial improvement in instrument identification. The signature acoustic timbre and striking characteristics of a physical tabla are clearly recognisable, though a minor background hiss persists. |
| **Melodic Duet** | A traditional Indian Hindustani classical raga-styled instrumental, featuring a slow, expressive sitar melody beautifully accompanied by steady, rhythmic tabla beats. | 10s | Great structure and audio quality, with the sitar demonstrating authentic resonance. Sitar and tabla outputs are synchronised, maintaining rhythmic cohesion and a defined melody line. |
| **Grand Ensemble** | A grand Indian traditional raga-styled ensemble performance, featuring a rich tapestry of veena, sitar, tabla, and other diverse string and percussion instruments, creating a vibrant rhythm and majestic melody. | 10s | Layered and majestic acoustic atmosphere. Displays successful separation between string plucks and percussion hits, though slight noises occur randomly. |

### Final Attempt 2 Observations
- Hyperparameter recalibration produced successful, high-fidelity style transfer without breaking core audio synthesis.
- Trainable matrices effectively encoded multi-instrument prompt constraints, but small model size limits frequency separation during dense multi-instrument phases.
- The adapter successfully integrated traditional Hindustani classical instrumentation, rhythmic synchronicity, and raga structural traits.
- Minor remaining constraints—including residual background noise on solo tracks and moderate congestion in ensemble passages—are directly related to:
1. **Dataset Volume Bottlenecks:** Training on a targeted 5-file dataset causes the model to inherit unique ambient and recording signatures present in the raw input files.
2. **Decoder Capacity Restraints:** The 300M parameter model size exhibits fixed physical limits when tracking overlapping instrumental frequencies, introducing minor synthetic textures under heavy conditions.

# Project Analysis: Final Fine-Tuning and Inferences

## 1. Conditioning Experiments (Base Model Tests)
Objective evaluation of the base model was conducted before fine-tuning. This phase analysed the impact of prompt complexity, instrument specificity, and decoding parameters such as `do_sample`. 
The comprehensive data and specific test logs are available in the [Conditioning Experiments Log](./Conditioning_Experiments_Log.md).

---

## 2. Baseline Audio Outputs

### Methodology
Generation of baseline control samples utilised the base `facebook/musicgen-small` pre-trained model. The evaluation targeted three distinct descriptive Hindustani classical prompts. Generation parameters were held constant at `guidance_scale=3.0` and `temperature=1.0` to generate `10-second` outputs (`500 tokens`) at a `32kHz sampling rate`.

### Baseline Evaluation 
Outputs can be found at [Baseline Outputs](./Outputs%20%28Experiments%2C%20Baseline%2C%20Fine-tuning%29/Fine-Tuning%20%26%20Related%20Outputs/Baseline%20Outputs/)
| Prompt type | Prompt | Duration | Observation | Inference |
| :--- | :--- | :--- | :--- | :--- |
| **Solo Percussion** | A highly detailed solo Indian classical Hindustani modern tabla instrumental, featuring intricate rhythmic patterns, fast-paced beats, and clear, resonant percussion. | `10s` | High-fidelity audio generation, but the rhythm adheres to Western electronic drum conventions. Complete absence of traditional acoustic tabla signatures or syllable structures (bols). | Base model lacks representation of specialised Indian percussion instruments, substituting them with mainstream acoustic or electronic variants present in the pre-training dataset. |
| **Melodic Duet** | A traditional Indian Hindustani classical raga-styled instrumental, featuring a slow, expressive sitar melody beautifully accompanied by steady, rhythmic tabla beats. | `10s` | Generation of a generic acoustic string timbre resembling a guitar or Middle Eastern stringed instrument. Output lacks microtonal slides and structural adherence to a classical raga sequence. | Pre-trained weights fail to capture localised stylistic microtones and linear structures of raga-based systems, defaulting to standard Western harmonic loops. |
| **Grand Ensemble** | A grand Indian traditional raga-styled ensemble performance, featuring a rich tapestry of veena, sitar, tabla, and other diverse string and percussion instruments, creating a vibrant rhythm and majestic melody. | `10s` | Coherent multi-instrument blend, but texturing resembles a Western cinematic ensemble rather than a traditional Indian classical arrangement. Individual instruments like the veena cannot be isolated. | Autoregressive decoding handles complex multi-instrument prompts efficiently but groups instruments into standard orchestral stereo layers, obscuring regional classical instrumentation. |

### Final Baseline Observations
The base model demonstrates strong audio generation stability and clear outputs for mainstream acoustic properties. However, a significant architectural bias toward Western musical structures limits zero-shot capabilities for authentic Hindustani classical instrumentation and performance styles.

---

## 3. Fine-Tuning Attempt 1: `Failure`
Outputs can be found at [Fine-Tuning Attempt 1](./Outputs%20%28Experiments%2C%20Baseline%2C%20Fine-tuning%29/Fine-Tuning%20%26%20Related%20Outputs/Fine-Tuning%20Attempt%201/)
### Methodology
Initial fine-tuning applied a high learning rate of `1e-4` and an aggressive adapter scaling factor (`lora_alpha=32`, `r=16`) over `5 training epochs`, utilising the 5-file chunked dataset.

### Attempt 1 Evaluation Table

| Prompt type | Prompt | Duration | Observation | Inference |
| :--- | :--- | :--- | :--- | :--- |
| **Solo Percussion** | A highly detailed solo Indian classical Hindustani modern tabla instrumental... | `10s` | Severe audio degradation characterized by high-frequency squeaks and continuous digital static. Total collapse of rhythm and percussion tracking. | High learning rate caused structural divergence in the internal audio codebooks, breaking the sequencing capabilities of the decoder. |
| **Melodic Duet** | A traditional Indian Hindustani classical raga-styled instrumental... | `10s` | Heavily corrupted output dominated by low-frequency hums and digital artifacts. A faint, unstructured string drone is barely perceptible under the noise floor. | Over-aggressive gradient updates distorted the frozen base model attention map projections, destroying model synthesis capacity. |
| **Grand Ensemble** | A grand Indian traditional raga-styled ensemble performance... | `10s` | Total acoustic failure consisting entirely of clipping noise, audio stuttering, and systemic distortion. | The model suffered from severe overfitting and weight explosion. The hyperparameter configuration forced memorisation of raw waveform data boundaries instead of stylistic abstraction. |

### Final Attempt 1 Observations
This iteration highlighted the sensitivity of autoregressive audio decoders to aggressive parameter adjustments. Excessive learning rates and improper LoRA alpha-to-rank balancing destabilise the pre-trained latent space, resulting in catastrophic failure of the generation pipeline.

---

## 4. Fine-Tuning Attempt 2: `Success`
Outputs can be found at [Fine-Tuning Attempt 2](./Outputs%20%28Experiments%2C%20Baseline%2C%20Fine-tuning%29/Fine-Tuning%20%26%20Related%20Outputs/Fine-Tuning%20Attempt%202/)
### Methodology
The second fine-tuning iteration introduced stabilised hyperparameters to protect pre-trained weights. The learning rate was lowered to `2e-5`, the LoRA alpha-to-rank ratio was optimised to `1:1` (`lora_alpha=16`, `r=16`), and total training duration was capped at `3 epochs` to avoid overfitting.

### Attempt 2 Evaluation Table

| Prompt type | Prompt | Duration | Observation | Inference |
| :--- | :--- | :--- | :--- | :--- |
| **Solo Percussion** | A highly detailed solo Indian classical Hindustani modern tabla instrumental... | `10s` | Marked improvement in instrument recognition, with distinct acoustic timbre and clear striking properties of the tabla. However, a minor background fuzz/hiss persists throughout the clip. | Lowered learning rate enabled successful mapping of the specialized percussion profile. The background noise floor indicates either ambient noise captured in the source files or a data scarcity bottleneck. |
| **Melodic Duet** | A traditional Indian Hindustani classical raga-styled instrumental... | `10s` | Successful reconstruction of the sitar's characteristic metallic resonance. Sitar and tabla track with precise rhythmic synchronicity, maintaining structural harmony and a coherent melody. | Balanced scaling factor allowed the adapter to seamlessly overlay localized microtonal stylistic features onto the stable generative engine. |
| **Grand Ensemble** | A grand Indian traditional raga-styled ensemble performance... | `10s` | Majestic and layered acoustic atmosphere. Clear separation between plucked strings and percussion hits, though slight acoustic muddiness develops during dense, simultaneous instrument tracks. | Adaptive weights successfully encoded the structural constraints of a multi-instrument performance, but model capacity limitations restrict perfect separation under complex polyphonic settings. |

### Final Attempt 2 Observations
Hyperparameter correction achieved clean stylistic adaptation without compromising audio synthesis mechanics. The adapter successfully captured traditional Hindustani classical textures, instrument synchronicity, and melodic structures. Remaining limitations, such as noise in solo tracks and minor congestion in dense ensemble sequences, are directly attributable to:
1. **Dataset Volume Bottleneck:** Training on a limited 5-file dataset causes the model to inherit specific environmental and acoustic signatures of the raw source data.
2. **Decoder Parameter Constraints:** The 300M parameter architecture presents physical boundaries in processing overlapping polyphonic frequencies without introducing synthetic artifacting.

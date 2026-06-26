# Parameter Efficient Domain Adaptation of Generative Audio Transformers: `MusicGen`

## 1. Objective
The primary goal of this project was to evaluate parameter-efficient style adaptation on MusicGen, an autoregressive transformer-based decoder designed for text-to-audio generation. The target domain focused on conditioning the model to synthesise authentic traditional Hindustani classical instrumental music through multiple focused attempts, including conditioning, fine-tuning, and implementation of related works done in this field.

---

## 2. Model Architecture
The project utilised **`facebook/musicgen`**, a single stage auto-regressive Transformer architecture, available in `300M`, `1.5B`, and `3.3B` parameter sizes. The model leverages an EnCodec neural audio compressor to map continuous acoustic waveforms into discrete multi-stream token spaces, coupled with an autoregressive language model decoder for conditional sequence generation. All synthesis passes operate natively at a 32kHz sampling rate.

---

## 3. Approach & Methodology

- `Conditioning experiments` and `Baseline testing`
- Targetted Parameter Efficient `Fine-Tuning` attempt on a small yet high-quality dataset
- Implementing `Related Research` done on the topic, and improving the work already done on Domain Adaptation of Generative models


---

## 4. Conditioning Experiments (Base Model Tests)

Before model alteration, objective benchmarks were run on the unadapted base architecture to analyse the interplay between prompt phrasing, length variables, instrument definitions, and sampling choices. 

The complete documentation of these configurations and comparative results is available in the [Conditioning Experiments Log](./Conditioning_Experiments_Log.md).


---


## 5. Fine-Tuning through LoRA: Successful Style Adaptation

The main goal of this experiment was to test how to fine-tune MusicGen (Facebook's AI music model) without needing massive computing power. Specifically, I wanted to fine-tune the `facebook/musicgen-small` model on realistic, high-quality traditional Hindustani classical instrumental music. 

The complete documentation of this attempt, including the dataset, processing, approach, methodology, roadblocks and results is available in the [Fine-Tuning_LORA folder](Fine-Tuning_LORA/README_Fine-Tuning.md).

---

## 6. Implementing Related Research

`ongoing`
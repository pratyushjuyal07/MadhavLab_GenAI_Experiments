# Research Review:
[Exploring Adapter Design Tradeoffs for Low-Resource Music Generation
-A Mehta, S Chauhan, M Choudhury - Proceedings of the 33rd ACM International Conference …, 2025](https://drive.google.com/file/d/1wqiPkZ_SlvHkL3-X0cOZku_4t9Ej5LvK/view?usp=sharing)

## Summary

This work investigates parameter-efficient fine-tuning of MusicGen for non-Western music traditions, including Hindustani classical music. Instead of full-model fine-tuning, the authors compare multiple adaptation architectures (Linear, CNN, and Transformer adapters) under varying parameter budgets. The objective is to determine whether high-quality specialisation can be achieved with significantly fewer trainable parameters.

## Key Observations

* MusicGen adapts effectively to Hindustani classical music using lightweight adapters.
* Increasing trainable parameters consistently improves generation quality until performance saturates.
* FAD (Fréchet Audio Distance) improves substantially with larger adapter sizes, indicating closer alignment to the target music distribution.
* Transformer adapters generally achieve the lowest FAD scores on Hindustani music, suggesting better modelling of long-range musical structure.
* CNN adapters remain highly competitive despite using simpler architectures.
* Performance gains beyond ~40M trainable parameters appear limited, indicating diminishing returns.

## Architecture Analysis


<img width="560" height="711" alt="image" src="https://github.com/user-attachments/assets/583240c9-726b-4a87-9b70-3166acd2d131" />



### Transformer Adapters

**Strengths**

* Best overall FAD performance for Hindustani music.
* Better capture of long-term melodic dependencies and temporal structure.
* More suitable for raga-based music where note progression and context matter.

**Weaknesses**

* Higher computational cost.
* Larger memory footprint.
* Increased training complexity.

### CNN Adapters

**Strengths**

* Lower computational requirements.
* Faster training and inference.
* Competitive quality despite reduced complexity.

**Weaknesses**

* Slightly weaker modelling of long-range musical relationships.
* Higher FAD values compared to Transformer adapters.

## Efficiency vs Quality

A major contribution of the paper is demonstrating that meaningful adaptation can be achieved without full fine-tuning. The results suggest that parameter-efficient methods can retain most of the quality improvements while substantially reducing training costs.

Although Transformer adapters often achieve superior quality metrics, CNN adapters provide a stronger quality-to-compute ratio. This explains why the authors emphasise CNN-based approaches in their conclusions, despite Transformers achieving higher absolute scores on several evaluation metrics.

## Relevance

For Hindustani MusicGen adaptation:

* Transformer-based adaptation appears most promising when generation quality is the primary objective.
* CNN-based methods are attractive when GPU resources are limited.
* Parameter-efficient tuning (LoRA/adapters) is a practical alternative to full-model fine-tuning.
* The observed importance of temporal modelling aligns with the structure of Indian classical music, where melodic progression and phrase development are central.

## Main Limitation

The paper focuses primarily on objective metrics such as FAD and FD. It does not adequately evaluate raga adherence, improvisational structure, ornamentation, or other musically meaningful characteristics specific to Indian classical traditions. As a result, improvements in metric scores do not necessarily imply improved musical authenticity.

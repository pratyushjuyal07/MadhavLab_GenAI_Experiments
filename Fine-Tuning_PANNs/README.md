## Fine-Tuning through LoRA after data selection through a PANNs threshold

- Objective: instrumental identification after fine-tuning on selective clips
- Prompts: 3 prompts across all attempts:
  - PROMPT_1 = "An expressive Hindustani classical raga performance, with instrumental sitar and other traditional plucked string instruments"
  - PROMPT_2 = "A detailed music recording centered entirely on the solo sitar showing virtuosic picking techniques and deep resonance."
  - PROMPT_3 = "A detailed musical melody with instrument X"
- Dataset: sourced from AudioSet (Google Research Website)


### Threshold selected=0.15 (randomly)

- Noisy audio after fine-tuning

### Threshold selected=0.5 (after data analysis)

<img width="695" height="470" alt="image" src="https://github.com/user-attachments/assets/b1964497-4ed6-4d4d-8832-d4518a3b6ed0" />

- Noisy audio after fine-tuning

### Possible improvements
- More data required
- Hyperparameter tuning required

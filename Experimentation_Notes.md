# Experiment log

## Experiment SET-A: Prompt Complexity
| Prompt type | Prompt | Duration | Observation |
|--------|----------|-------------|-------------|
| Simple prompt | Traditional Indian classical instrumental raga | 5s and 10s | Decent audio quality in the 5s clip, but very weak audio in the 10s clip, with a high amount of noise and negligible instrument recognition and genre style |
| Intermediate prompt | Traditional Indian classical sitar with tabla accompaniment | 5s and 10s | Rhythm improved, better beat style and enhanced instrument quality and recognition, although performance dipped in the 10s clip |
| Detailed prompt | Slow and Meditative Indian Classical sitar performance, with light tabla beats enforcing a calm and meditative environment | 5s and 10s | Greatly improved mood and style of audio generated, with better coherence and rhythm along with extreme realism, especially in the 10s clip |

#### Observations:
- More descriptive and contextual prompts lead to higher quality outputs
- Simpler prompts fail to do well when generating longer audio by causing long-term instability and noise
- Detailed prompts lead to high quality outputs even when generating longer clips due to contextual clarity provided by environmental and tempo descriptions

## Experiment SET-B: Instrument-specific prompts, classified by genre
| Prompt type | Prompt | Duration | Observation |
|--------|----------|-------------|-------------|
| Indian Instrument specific (Tabla) | Fast energetic tabla solo performance | 5s | Very weak instrument recognition, generated almost piano-like audio, no sign of Indian classical genre in the audio style |
| Indian Instrument specific (Flute) | Traditional Indian flute melody | 5s | Correct instrument recognition of the flute, but no Indian classical style melody; extremely shrill and unpleasant audio quality with no melody or tune |
| Western Instrument specific (Guitar) | Western country guitar beat | 5s | High quality instrument recognition, correct genre-styled audio (country guitar), along with decent melody and tune |
| Western Instrument specific (Piano) | Ambient electronic piano tune | 5s | Decent instrument recognition, with correct genre-styled audio (electronic piano), however, slightly muffled and noisy audio quality |

#### Observations: 
- The model performs significantly better when producing western styled outputs or working with mainstream instruments like guitar and piano compared to the Indian classical tabla and flute
- Instrument recognition remained comparatively weaker when working with classical audio outputs
- Audio quality and rhythm structure were reduced with Indian music
- Even though the model partially recognised Indian instruments (like the flute), stylistic understanding remained weak, suggesting heavy pre-training on mainstream Western audio data

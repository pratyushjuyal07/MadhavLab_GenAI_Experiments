# Simple and Controllable Music Generation: `MusicGen`

[Simple and controllable music generation
J Copet, F Kreuk, I Gat, T Remez, D Kant, G Synnaeve… - Advances in neural information processing systems, 2023](https://arxiv.org/pdf/2306.05284)

## Key points
- Neural Audio Tokenisation: Encodec; latent space; downsampling to 50 frames
  - z-space discretization: RVQ(K codebooks, M(2048) entries of Dz dimension)
  - audio => discrete matrix Q of size {1,2,3... M}^(T x K), T=50
- Objective: modelling joint probability distribution P(Q) over parallel streams
  - ($\Omega$) => 2D grid defined by (t,k) | t from 1,T ; k from 1,K
  - Codebook patterns => a partition Ps belonging to ($\Omega$), for s=1,2,...S
  
- Pattern flattening: sorted by (time,codebook); joint distribution; O(S^2), S=TxK
  <img width="768" height="172" alt="image" src="https://github.com/user-attachments/assets/5c010d4c-f2e8-458b-912b-5713100844bd" />

- Parallel pattern: concurrent prediction for K codebooks
  <img width="742" height="181" alt="image" src="https://github.com/user-attachments/assets/a128dd36-a59b-4bfd-9e2f-15c380496734" />

- Computational Limitation: for efficiency, codebooks at t are assumed to be conditionally independent of codebooks of <t
  <img width="768" height="129" alt="image" src="https://github.com/user-attachments/assets/c7e891bd-a3ba-4565-a0d6-0b22d42e338b" />

- Which violates RVQ! kth codebook depends on 1,2...k-1



# Simple and Controllable Music Generation: `MusicGen`

[Simple and controllable music generation
J Copet, F Kreuk, I Gat, T Remez, D Kant, G Synnaeve… - Advances in neural information processing systems, 2023](https://arxiv.org/pdf/2306.05284)

## Key points
- Neural Audio Tokenisation: Encodec; latent space; downsampling to 50 frames
  - z-space discretization: RVQ(K codebooks, M(2048) entries of Dz dimension)
  - audio => discrete matrix Q of size {1,2,3... M}^(T x K), T=50
- Objective: modelling joint probability distribution P(Q) over parallel streams
  - Q => 2D grid defined by (t,k) | t from 1,T ; k from 1,K
- Codebook patterns (&#937) 

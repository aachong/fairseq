import torch

noise_sampler = torch.distributions.normal.Normal(
    loc=0.0, scale=0.1
)

token_embeddings = torch.randn(3,4)
print(token_embeddings)
noise = noise_sampler.sample(sample_shape=token_embeddings.shape).to(
    token_embeddings
)
noised_embeddings = token_embeddings.clone() + noise
print(noised_embeddings)
a =1e-12
if a==1e-12:
    print(a)
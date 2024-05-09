## Simple Example of A Diffusion Model 

Diffusion model becomes a very popular model not just in image generation, but also in protein design. 

Let's start with a straightforward scenario to study the diffusion model.

The Denoising Diffusion Probabilistic Model (DDPM) comprises two integral processes: the forward process and the reverse process. 

Forward Process: In this process, noise sampled from a Gaussian distribution is incrementally added to an image at each time step. This forward process iteratively refine the pixel distribution of the image until it approximates a Gaussian distribution.

Reverse Process: Conversely, the reverse process involves predicting the noise and subsequently removing it from a noisy image at each time step. 



![add_remove_noise](https://github.com/WangM220/Diffusion_model_noise/assets/143626969/7cbcd8df-6f38-4173-8279-b322f9752c81)


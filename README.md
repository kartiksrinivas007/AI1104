### Latent Space Visualizations of Generative Models

---


The repository visualizes the latent spaces that are generated using an Autoencoder and a Variational Autoencoder.
In order to do this for both the AE and the VAE we pick either a latent representation of an Image or a sample from the latent distribution respectively and decode it to visualize the generated image.
We can pick <b>two</b> of these images, and then interpolate the points in the latent between them to obtain a smooth transition between the two images.

Here is what an Autoencoder will generate for the MNIST dataset (with interpolations in the latent between one and 5):


![AE](ae.gif) 


A side by side comparison of the latent variables shows the advantages of VAE's giving a more smooth transition between the images.
Here is the visualization of the VAE for the same dataset:

![VAE-AE](test_0_1.gif)

Showing below is the grid of the VAE ,followed by the AE's grid ,notice that the VAE gives more readable results at the interpolation points ,showcasing it's usability for generative AI related tasks.
![VAE-grid](VAE_grid.jpeg)

AE's grid below
![AE-grid](AE_grid.jpeg)


### GAN

To run the GAN please checkout the `GAN.py` file in the GAN directory.
Run it using this 

```
python3 GAN.py --lr 1e-5 --epochs 2
```

Even after extensive grid search iover the hyper-parameters in the `run.py` file, it is too hard to converge within 10 epochs.
We hypothesize that more training is required (50 epochs) on lower learning rates ~ 1e-7.
Here is the loss plot for a couple epochs with that learning rate

![GAN](GAN/losses_300.png)

The file also generates the latent spaces images made by the generator. Please generate them by running the `GAN.py` file and looking in the appropriate directory



# WGAN-GP-PyTorch
Implementation of *Improved Training of Wasserstein GANs* (Gulrajani, et al.) in PyTorch


## Requires
* `PyTorch`
* `Numpy`
* `scipy` (for loading SVHN .mat file)

## Training
Run the following code to train on the SVHN dataset
```bash
python code/train_SVHN.py <location-of-SVHN-mat-file> <model-name>
```
The parameters can be changed through the command line. Use the `--help` flag for more details

## Results
![Generated vs Real Data](images/gen_vs_real_comp.png?raw=true)
This image compares real data and data generated from the model
![Image Interpolation](images/image_interp.png?raw=true)
This image demonstrates an interpolation from two points in the noise space of the generator. A straight line was drawn between two randomly drawn vectors and the points along that line were fed into the generator to produce these images.

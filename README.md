# WGAN
-Wasserstein Generative Adversarial Network applied to the task of removing phase errors from SAR imagery. This directory contains files required to create an synthetic phase corrupted dataset as well as the WGAN model to train with.
Inside folder WGANAUTOFOCCUS
1.	WGANAutofocus.zip, this contains a test dataset called "Random" extract this file first. Inside it contains the folder "Random" this has the paired images "Org" and "Cor" for the test and train datasets.
2.	To create your own dataset, put images of the same size in a folder called "Org" with an empty folder called "Cor" then run the run_Image_Generator.py file. This will fill the "Cor" folder, split the images into test and train folders.
3.	WGAN.py will train on the dataset given, and produce weights.
4.	RunModel.py will take a dataset given and apply the WGAN generator with the weights produced in WGAN.py. This will produce the phase-corrected results of the corrupted images.

# A Study on Adam and AdamW

This project explores the performance differences between the Adam and AdamW optimization algorithms in the context of a convolutional neural network (CNN) model used for classifying the MNIST dataset.

## Team Members

d11949006 WEI-LUN CHEN  
r11921118 YONG-TAI QIU  

## Experiment Setup

We are interested in comparing different optimizers and their settings.  
The main focus of our investigation is on the following setups:

+ **Adam optimizer** vs **Adam optimizer without bias correction**
+ **Adam optimizer with L2 regularization** vs **AdamW optimizer**

|   Optimizers    |                     Parameter Settings                     |
| :------------: | :--------------------------------------------------------: |
|    **Adam**    |  lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0   |
| **CustomAdam*** |  lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0   |
|    **Adam**    | lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01 |
|   **AdamW**    | lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01 |

We implemented a new optimizer called **CustomAdam**, which removes the bias correction present in the original Adam optimizer.

The CNN model was trained for 100 epochs with an initial learning rate (lr) of 1e-3, 1e-4, and 1e-5.
All implementations used the PyTorch package.

## Results

> TEX file and generated PDF are in the [reports](/reports/) folder

![results](/assets/merge_results.png)
# A Study on Adam and AdamW

This project explores the performance differences between the Adam and AdamW optimization algorithms in the context of a convolutional neural network (CNN) model used for classifying the MNIST dataset.

## Team Members

d11949006 WEI-LUN CHEN  
r11921118 YONG-TAI QIU  

## Experiment Setup

We are interested in comparing different optimizers and their settings.  
The main focus of our investigation is on the following setups:

+ **Experiment1**
    + **Adam optimizer** vs **Adam optimizer without bias correction**

|   Optimizers    |                     Parameter Settings                     |
| :------------: | :--------------------------------------------------------: |
|    **Adam**    |  lr=**learning_rates**, betas=(0.9, 0.999), eps=1e-08, weight_decay=0   |
| **CustomAdam*** |  lr=**learning_rates**, betas=(0.9, 0.999), eps=1e-08, weight_decay=0   |

We implemented a new optimizer called **CustomAdam**, which removes the bias correction present in the original Adam optimizer.

The CNN model was trained for 100 epochs with different learning rate (**learning_rates**=[5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]).

+ **Experiment2**
    + **Adam optimizer with L2 regularization** vs **AdamW optimizer**

|   Optimizers    |                     Parameter Settings                     |
| :------------: | :--------------------------------------------------------: |
|    **Adam**    |  lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=**weight_decays**   |
|   **AdamW**    | lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=**weight_decays** |

The CNN model was trained for 100 epochs with different weight decays (**weight_decays**=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]).

All implementations used the PyTorch package.

## Results

> TEX file and generated PDF are in the [reports](/reports/) folder

+ Experiment1 Results:
![merge_all_result_adam_with_and_without_bias_correction_with_diff_learning_rates](/assets/experiment1/merge_all_results.png)

+ Experiment2 Results:
![merge_all_result_adam_adamW_with_diff_weight_decays](/assets/experiment2/merge_all_results.png)

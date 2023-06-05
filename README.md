# A Study on Adam and AdamW

This project explores the performance differences between the Adam and AdamW optimization algorithms in the context of a convolutional neural network (CNN) model used for classifying the MNIST dataset.

## Team Members

d11949006 WEI-LUN CHEN  
r11921118 YONG-TAI QIU  

## Experiment Setup

We are interested in comparing different optimizers and their settings.  
The main focus of our investigation is on the following setups:

+ **Experiment 1**
    + **Adam optimizer** vs **Adam optimizer without bias correction**

|   Optimizers    |                     Parameter Settings                     |
| :------------: | :--------------------------------------------------------: |
|    **Adam**    |  lr=**learning_rates**, betas=**betas**, eps=1e-08, weight_decay=0   |
| **CustomAdam*** |  lr=**learning_rates**, betas=**betas**, eps=1e-08, weight_decay=0   |

We implemented a new optimizer called **CustomAdam**, which removes the bias correction present in the original Adam optimizer.

The CNN model was trained for 100 epochs with different learning rates (**learning_rates**=[1e-3, 1e-4, 1e-5]) and different betas (**betas**=[(0.9, 0.999), (0.99, 0.999)])

We want to know that:
1. Under the same lr, different beta, whether with or without bias correction will affect the accuracy? (Experiment1-1)
2. Under the same beta, different lr, whether with or without bias correction will affect the accuracy? (Experiment1-2)

+ **Experiment 2**
    + **Adam optimizer with L2 regularization** vs **AdamW optimizer**

|   Optimizers    |                     Parameter Settings                     |
| :------------: | :--------------------------------------------------------: |
|    **Adam**    |  lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=**weight_decays**   |
|   **AdamW**    | lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=**weight_decays** |

The CNN model was trained for 100 epochs with different weight decays (**weight_decays**=[1e-4, 1e-3, 1e-2]).

We want to know that:
1. Under different lambdas(weight decay), whether Adam with L2 and adamW will affect the accuracy? (Experiment2)

All implementations used the PyTorch package.

## Results

> TEX file and generated PDF are in the [reports](/reports/) folder

+ Experiment 1-1 Results:
![merge_all_result_adam_with_and_without_bias_correction_with_diff_learning_rates](/assets/experiment1-1/experiment1-1_merged_results.png)

+ Experiment 1-2 Results:
![merge_all_result_adam_with_and_without_bias_correction_with_diff_learning_rates](/assets/experiment1-2/experiment1-2_merged_results.png)

+ Experiment 2 Results:
![merge_all_result_adam_adamW_with_diff_weight_decays](/assets/experiment2/experiment2_merged_results.png)

# DS-301-Final-Project

## Introduction
Global pandemic due to the spread of COVID-19 has post challenges in a new dimension on facial recognition, where people start to wear masks. Under such condition, we consider utilizing machine learning in image inpainting to tackle the problem, by complete the possible face that is originally covered in mask. In particular, autoencoder has great potential on retaining important, general features of the image as well as the generative power of the `generative adversarial network (GAN)`. We implement a combination of the two models, context encoders and explain how it combines the power of the two models and train the model with 50,000 images of influencers faces and yields a solid result that still contains space for improvements. 

## Code structure
```
|   ...
├── datamanager.py # the training script
├── neuralnet.py # the evaluation script for mAP
├── run.py # make a single image prediction and draw the bounding box
├── tf_process.py # the inference time script
|   ...
```
## Data preparation 
You need to download the data from http://www.seeprettyface.com/mydataset_page3.html
And change the PACK_PATH based on data path on your own computer in datamanager.py


## Environment setup
1. Install dependencies with `pip`: 
```bash
pip install -r requirements.txt
```
2. Make sure your packages' version:  
* Python 3.7.4
* Tensorflow 1.14.0
* Numpy 1.17.1
* Matplotlib 3.1.1
* Scikit Learn (sklearn) 0.21.3


## Example commands to execute the code

```bash
python run.py
```
All the training and testing processes are included in tf_process.py and will be called in run.py

## Evaluation results

- Epochs: 20
- Batch Size: 64
- Training Time: 10hr 40min 21sec
- Lowest Loss (Total): oscillates around 70 out of 45k images


|Epoch|Error(MSE)|
|  1  | 6365.84 |
|  2  | 6566.12 |
|  3	| 2901.91 |
|  4	| 3455.64 |
|  5  | 3123.77 |
|  6  | 2771.54 |
|  7	| 1593.35 |
|  8  | 1744.65 |
|  9  | 1902.63 |
|  10 | 932.38  |
|  11 | 841.25  |
|  12 | 469.33  |
|  13 | 452.28  |
|  14 | 322.17  | 
|  15 | 140.99  |
|  16 | 118.37  |
|  17 | 84.31   |
|  18 | 93.44   |
|  19 | 77.74   | 
|  20 | 71.05   |

![c82643399914f46d1e20f4a08332d43](https://user-images.githubusercontent.com/36658078/208318623-91333226-f444-43e0-bd1d-8496b32379f9.png)


Observation:  
ResNet50 has a slightly better mAP than MobilNetV3.  
However, it is 3x slower than MobilNetV3 during inference.  
Since there is not a marginal difference of the mAP between `MobilNetV3` and `ResNet50`, we prefer `MobilNetV3`.


## References
[1]	Dhamecha. (2021). A Detailed Explanation of GAN with Implementation Using Tensorflow and Keras. Data Science Blogathon. 
[2]	Wang, Q., Fan, H., Sun, G., Ren, W., & Tang, Y. (2020). Recurrent generative adversarial network for face completion. IEEE Transactions on Multimedia, 23, 429-442.
[3]	Din, N. U., Javed, K., Bae, S., & Yi, J. (2020). A novel GAN-based network for unmasking of masked face. IEEE Access, 8, 44276-44287. 
[4]	Liu, H., Jiang, B., Xiao, Y., & Yang, C. (2019). Coherent semantic attention for image inpainting. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 4170-4179). 
Code reference: https://github.com/YeongHyeon/Context-Encoder

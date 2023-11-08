# DS-301-Final-Project

## Introduction
Global pandemic due to the spread of COVID-19 has post challenges in a new dimension on facial recognition, where people start to wear masks. Under such condition, we consider utilizing machine learning in image inpainting to tackle the problem, by complete the possible face that is originally covered in mask. In particular, `Auto-Encoder` has great potential on retaining important, general features of the image as well as the generative power of the `generative adversarial network (GAN)`. We implement a combination of the two models, context encoders and explain how it combines the power of the two models and train the model with 50,000 images of influencers faces and yields a solid result that still contains space for improvements. 

The Implementation of Auto-Encoder:

![a3ed52ceff3ebd245ee2b5387678529](https://user-images.githubusercontent.com/36658078/208319565-e3a9cb19-6b17-4b33-9bf3-d860188aa68d.png)


## Publication
[GAN-based Algorithm for Efficient Image Inpainting](https://arxiv.org/abs/2309.07293)
## Code structure
```
|   ...
├── datamanager.py # import the data and preprocess the data
├── neuralnet.py # GAN model created in this file
├── run.py # Call all the other files in one integrated file 
├── tf_process.py # train and test process
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
|----- | ------|
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

Sample Testing Output:

![633ba2ff4aa21659ddfc75fb0cd2017](https://user-images.githubusercontent.com/36658078/208319321-1784658a-d4bb-4629-b8b9-5d79bcaa6989.png)


Observation:  
It takes approximately 10 hours to complete 20 epochs with a batch size of 64. While it seems an acceptable on the time scale, consider the fact we are training based on image size 128✕128, training datasets with larger image size, such as 512✕512 for example, may take longer time, possibly on an exponential. It is possible that the model requires less training time once given a proper batch size. Also, from the sample image above, we can find that the edge of inpainting part is somehow inconsistent with the original image, which needs further improvement.


## References

[1] Dhamecha. (2021). A Detailed Explanation of GAN with Implementation Using Tensorflow and Keras. Data
Science Blogathon.

[2] Wang, Q., Fan, H., Sun, G., Ren, W., & Tang, Y. (2020). Recurrent generative adversarial network for face
completion. IEEE Transactions on Multimedia, 23, 429-442.

[3] Din, N. U., Javed, K., Bae, S., & Yi, J. (2020). A novel GAN-based network for unmasking of masked face.
IEEE Access, 8, 44276-44287.

[4] Liu, H., Jiang, B., Xiao, Y., & Yang, C. (2019). Coherent semantic attention for image inpainting.
In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 4170-4179).

[5] Yu, J., Lin, Z., Yang, J., Shen, X., Lu, X., & Huang, T. S. (2018). Generative image inpainting with contextual
attention. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 5505-5514).

[6] Pathak, D., Krahenbuhl, P., Donahue, J., Darrell, T., & Efros, A. A. (2016). Context encoders: Feature learning
by inpainting. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2536-
2544).

[7] Chen, L., Fei, H., Xiao, Y., He, J., & Li, H. (2017, July). Why batch normalization works? a buckling
perspective. In 2017 IEEE International Conference on Information and Automation (ICIA) (pp. 1184-1189).
IEEE.

[8] Wu, S., Li, G., Deng, L., Liu, L., Wu, D., Xie, Y., & Shi, L. (2018). $ L1 $-norm batch normalization for
efficient training of deep neural networks. IEEE transactions on neural networks and learning systems, 30(7),
2043-2051.

[9] Zhou, Y. (2022). The Efficient Implementation of Face Mask Detection Using MobileNet. In Journal of Physics:
Conference Series (Vol. 2181, No. 1, p. 012022). IOP Publishing.

[10]Tran, L., Yin, X., & Liu, X. (2017). Disentangled representation learning gan for pose-invariant face
recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1415-
1424).

Code reference: https://github.com/YeongHyeon/Context-Encoder

# PixColor Pytorch implementation

paper link: [here](https://arxiv.org/abs/1705.07208)

![network architecture](images/model_arch.png)

* There are four main networks included in the architecture

**pix_network_1.py**
1. Conditioning Network:
Pretrain conditioning network on COCO image segmentation

2. Adaptation Network:
Conditioning and adaptation network turn brightness channel Y into a set of features that are used for conditioning the PixelCNN.

3. Coloring Network(pixelCNN):
pixelCNN is optimized alongside conditioning and adaptation network. It predicts a low resolution chrominance of the image


**pix_network_2.py**

4. Refinement Network:  
The low resolution color image made from the previous network is fed into the refinement network, which then produces a full resolution colorization
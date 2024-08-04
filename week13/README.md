# Computer Vision - 4 Augest 2024 - Week 13
## Introduction to Deep Learning for Computer Vision

- What is Deep learning for
computer vision?
- **Key Componenets**
  - Convolutional Neural Networks (CNNs)
  - Popular CNN Architectures 
    - LeNet-5
    - AlexNet
    - VGGNet
    - ResNet
    - Inception (GoogLeNet)
  - Transfer Learning 
  - Generative Adversarial Networks (GANs)
    - Basic
    - Generator
    - Discriminator
    - Application
  - Object Detection and Segmentation 
    - YOLO
    - SSD
    - Faster R-CNN
    - Mask R-CNN
  - Image Segmentation
    - **Semantic Segmentation:** Classifying each pixel in an image into a predefined
category. 
    - **Fully Convolutional Networks (FCNs):** Replace fully connected layers with
convolutional layers to produce dense pixel-wise predictions. 
    - **U-Net:** Combines a contracting path (encoder) with a symmetric expanding path
(decoder) for precise localization. 
    - **DeepLab:** Uses atrous convolution to capture multi-scale context and Conditional
Random Fields (CRFs) for precise object boundaries.
  - Recurrent Neural Networks (RNNs) and LSTMs 
    - **Basics**: RNNs are designed to handle sequential data, where the output
at each time step depends on previous time steps.
    - **LSTMs**: Long Short-Term Memory networks are a type of RNN that
mitigate the vanishing gradient problem by using gating mechanisms to
maintain long-term dependencies
    - **Applications**: Video classification, action recognition, and sequence
generation
  - **3D Vision** 
    - **3D Object Detection and Segmentation:** Using point clouds and depth maps to
detect and segment objects in three dimensions. 
    - **PointNet:** Directly processes point clouds by using a symmetric function to
aggregate information. 
    - **PointNet++:** Extends PointNet by introducing hierarchical feature learning. 
    - **Depth Estimation:** Estimating depth information from 2D images using CNNs
and multi-view stereo techniques.
  - Attention Mechanisms and Transformers
    - **Attention Mechanisms:** Allow the network to focus on relevant parts
of the input, improving performance in tasks like image captioning and
visual question answering (VQA). 
    - **Transformers:** Originally designed for NLP, transformers have been
adapted for vision tasks (e.g., Vision Transformer or ViT) to model
long-range dependencies
  - Self-Supervised and Semi-Supervised Learning

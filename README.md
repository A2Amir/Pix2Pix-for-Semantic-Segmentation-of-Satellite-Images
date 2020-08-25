
## What Is Pix2Pix and How To Use It for Semantic Segmentation of Satellite Images?


<p align="center">
  <img src="https://cdn-images-1.medium.com/max/2000/1*sqSDg7sha53xQMYCK_Ecgw.png" alt="Some Applications of Pix2Pix" width="600" height="400" >
 </p>


## 1. Introduction

**In the last three posts I have explained [Generative Adversarial Network](https://medium.com/analytics-vidhya/generative-adversarial-network-gan-b510d41df2cb), [its problems ](https://medium.com/analytics-vidhya/the-problems-of-generative-adversarial-networks-gans-3d887efa578e)and an extension of the Generative Adversarial Network called [Conditional Generative Adversarial Network](https://medium.com/analytics-vidhya/conditional-generative-adversarial-networks-cgans-46532afbdcc1) to solve the problems in the successful training of the GAN.**

As claimed earlier in [the last post](https://medium.com/analytics-vidhya/conditional-generative-adversarial-networks-cgans-46532afbdcc1), **Image to Image translation** is one of the tasks, which can be done by **[Conditional Generative Adversarial Networks](https://medium.com/analytics-vidhya/conditional-generative-adversarial-networks-cgans-46532afbdcc1)** (**CGANs**) ideally.

In the task of Image to Image translation, an image can be converted into another one by defining a loss function which is extremely complicated. Accordingly, this task has many applications like colorization and making maps by converting aerial photos. Figures above show great example of Image to Image translation.

**Pix2Pix network was developed based on the [CGAN](https://medium.com/analytics-vidhya/conditional-generative-adversarial-networks-cgans-46532afbdcc1).** Some of the applications of this efficient method include object reconstruction from edges, photos synthesis from label maps, and image colorization [[source](https://phillipi.github.io/pix2pix/)].

## 2. The architecture of Pix2Pix Network

As mentioned above, Pix2Pix is based on conditional generative adversarial networks ([CGAN](https://medium.com/analytics-vidhya/conditional-generative-adversarial-networks-cgans-46532afbdcc1)) to learn a mapping function that maps an input image into an output image. **Pix2Pix like [GAN](https://medium.com/analytics-vidhya/generative-adversarial-network-gan-b510d41df2cb), [CGAN ](https://medium.com/analytics-vidhya/conditional-generative-adversarial-networks-cgans-46532afbdcc1)is also made up of two networks, the generator and the discriminator.** Figure below indicates a very high-level view of the Image to Image architecture from the Pix2Pix paper.

![**A very high-level view of the Image-to-Image architecture [[source](http://P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, “Image-to-image translation with conditional adversarial networks,” in Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1125–1134,2017.)]**](https://cdn-images-1.medium.com/max/2028/1*0WR3FWWyLdcHwXduGN7kFQ.png)

### **2.1 The Generator Architecture**

**The generator goal is to take an input image and convert it into the desired image (output or ground truth) by implementing necessary tasks**. **There are two types of the generator, including encoder-decoder and U-Net network.** The latter difference is to have skip connections.

Encoder-decoder networks translate and compress input images into a low-dimensional vector presentation (bottleneck). Then, the process is reversed, and the multitudinous low-level information exchanged between the input and output can be utilized to execute all necessary information across the network. In order to circumvent the bottleneck for information, they added a skip connection between each layer **i** and **n-i** where **i** is the total number of the layers. It should be noted that the shape of the generator with skip connections looks like a U-Net network. Those images are shown below.

![**The architecture of the generators** **[[source](http://P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, “Image-to-image translation with conditional adversarial networks,” in Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1125–1134,2017.)]**](https://cdn-images-1.medium.com/max/2032/1*bh5b5JmdgpZJupqIhVydrg.png)
>  As seen in the U-Net architecture, the information from earlier layers will be integrated into later layers and because of using skip connections, they don’t need any size changes or projections.

### 2.2 The Discriminator Architecture

**The task of the discriminator is to measure the similarity of the input image with an unknown image**. This unknown image either belongs to the dataset (as a target image) or is an output image provided by the generator.

**The PatchGAN discriminator in Pix2Pix network is employed as a unique component to classify individual (N x N) patches within the image as real or fake**.

As the authors claim, since the number of PatchGAN discriminator parameters is very low, the classification of the entire image runs faster. The PatchGAN discriminator’s architecture is shown in figure below.

![**The architecture of the PatchGAN discriminator** **[[source](http://P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, “Image-to-image translation with conditional adversarial networks,” in Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1125–1134,2017.)]**](https://cdn-images-1.medium.com/max/2970/1*Ko9dmOP1cc5q3J8ggLxEbg.png)
>  The reason for calling this architecture as “**PatchGAN**” is that each pixel of the discriminator’s output (30 × 30 image) corresponds to the 70×70 patch of the input image. Also, it’s worth noting that according to the fact the input images size is **256**×**256**, the patches overlay considerably.

## 3. The training Strategy

It is well known that there are many segmentation networks, which can segment objects, but the most important thing that they do not consider is to have segmentation networks that their segmented objects look like ground truths from the perspectives of shapes and edges and contours . To make the topic more clear, an example from one of the own practical projects that was carried out is presented below.

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/2000/1*kMC7Txgk9PFMuipvZyDkwQ.png" alt="An example for building footprint segmentation" width="600" height="200" >
 </p>


As shown in figure above, the right figure represents the result for building footprint segmentation, and the middle figure shows the corresponding ground truth. The model can detect where buildings are located but fails to reflect the boundaries, which are present in the ground truth and edges and contours of the segmented buildings doesn’t match exactly the ground truth.

As known, how buildings look like, depends on how the contours of the building are drawn. The most crucial information of an object is the object boundary and the shape and edges of a segmented building can be controlled or improved with the help of contours (the object boundary). Based on this assumption, the extracted contours of each ground truth were added to the corresponding ground truth again to reinforce the shape and edges of a segmented building for further research.




<p align="center">
  <img src="https://cdn-images-1.medium.com/max/2000/1*4TJwqrRMD0ZF09qzqtSCHw.png" alt="Extracting and adding again contours to objects of the ground truths" >
 </p>



As seen in figures above, contours were added again to objects of the ground truths after extracting them with the OpenCV library.

After adding corrsponding contours again to objects of the ground truths (on the fly in the code), **Satellite images and ground truths with overlaid** **contours **(An example shown in the following) form the dataset used in the training of Pix2Pix network.

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/2000/1*k-ZUBwoWJdOQsqAc4-KUGg.png"  >
 </p>


**After defining th dataset, the weights of Pix2Pix network are adjusted in two steps.**

>  **In the first step, The discriminator** (figure below) takes **the input (Satellite image)/target (ground truths with overlaid contours)** and **then input (Satellite image)/output (generator’ output)** pairs, to estimate how realistic they look like. Then the adjustment of the discriminator’s weights is done according to the classification error of the mentioned pairs.

![**The flowchart of the adjusting weights of the discriminator**](https://cdn-images-1.medium.com/max/2174/1*l5GfHYZWZ9gC2FlWxbErPQ.png)

>  **In the second step, The generator’s weights**(figure below) **adjust by using the discriminator’s output and the difference between the output and target images.**


![**The flowchart of the adjusting weights of the generator**](https://cdn-images-1.medium.com/max/3060/1*PxyDwZUG_RgG7Aro7xYqWg.png)

Based on the paper, the objective function can be expressed as follow:


<p align="center">
  <img src="https://cdn-images-1.medium.com/max/2000/1*YO2XC9AL8DTRm2zIjb1YNA.png"  >
 </p>



The Authors of Pix2Pix used and added the loss function L1 measuring the standard distance between the generated output and the target.

## 4. The training, validation, and test Dataset

In this section, the characteristics of the **[FH Kufstein dataset](https://www.fh-kufstein.ac.at/)** will be explained. This dataset consists of two categories which are shown in Table below.

![**The characteristics of the FH kufstein dataset**](https://cdn-images-1.medium.com/max/2000/1*OBJtodXskiwpIcMoiKWbUQ.png)

As known, all pixels of an image are important when each pixel is used by a segmentation model with a specific size of the input. Therefore, the data (pixels) will be lost from the image and the size of the image will be much smaller if the number of pixels have to reduce to meet this specific size of the input. It can be crucial to retain more information about images when it comes to resizing an image without losing quality.

The images of the **[FH Kufstein dataset](https://www.fh-kufstein.ac.at/)** have to cut into 256×256 because Pix2Pix only takes in 256×256 images. In the first attempt, the size of the satellite and corresponding ground truth images was reduced all at once to 256×256, which leads to losing 158.488.281 pixels. According to the fact that each pixel plays an important role in detecting each edge of a building, many pixels got lost. For this reason, a process was designed to select and cut images:

>  **1. Due to lack of time and processing power, images with a high building density are selected from the entire dataset.**

>  **2. The size (4053×4053 pixels) of each selected image from the dataset (Satellite images and ground truths with overlaid contours) is changed to 4050×4050 pixels (see example blow).**


<p align="center">
  <img src="https://cdn-images-1.medium.com/max/2000/1*gKwU8KLnbzm3V5Bc3ZCKdg.png" alt="From 40537×40537 pixels to 4050×4050 pixels" >
 </p>

>  **3. Images from the last step cropped to the size of 675×675 pixels (see example blow).**

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/2000/1*D6H6ww0yasUmVi4fnZ7kmA.png" >
 </p>



>  **4. The size of all cropped images resized to 256 × 256 pixels, which means that little information is lost (approx. 2.636.718 pixels, which is little compared to 158.488.281 pixels).**

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/2000/1*3UaTHqZ6opnHl_3M2V2gyg.png" >
 </p>


The described process is repeated for all Images (***Satellite images and ground truths with overlaid contours***) to create the training, validation, and test datasets and the properties of each created dataset are indicated in Table below.

![**The characteristics of the training, validation and test datasets derived from the FH dataset**](https://cdn-images-1.medium.com/max/2000/1*37tTSPOydHoBvdTQmHQ_Zw.png)

A validation dataset will be needed to evaluate the model in the duration of training. The validation dataset can be employed for training goals to find and optimize the best model. In this experiment, the validation dataset can be employed for overfitting minimization and hyper-parameter fine-tuning.

## 5. Evaluation Criteria and Results

After training Pix2Pix on the training dataset, as shown below and tested on the validation and test datasets, the generator of Pix2Pix trying to improve the results and makes predictions which look like ground truths from the perspective of contours and edges.

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/2040/1*wgXEu3PKla3aJBnwf3QZ5w.png"  >
 </p>



<p align="center">
  <img src="https://cdn-images-1.medium.com/max/2096/1*ZtCWIwraCNddhpisqIPcHQ.png" alt ="The results of Pix2Pix on the validation dataset">
 </p>


![](https://cdn-images-1.medium.com/max/2084/1*RuThTjRq6HaV6119zD__pw.png)

![**The results of Pix2Pix on the test dataset**](https://cdn-images-1.medium.com/max/2042/1*JzBBQEp8dyP6NIgkjP2Hmw.png)

Evaluating the segmentation quality is very important for image processing, particularly in security cases, including autonomous vehicles. There are many evaluation criteria developed for segmentation evaluation so that researchers can make a choice based on their needs. In [**this post ](https://github.com/A2Amir/Evaluation-Criteria-for-Segmentation-Networks)**on [**my github](https://github.com/A2Amir)** you can read the evaluation criteria of segmentation networks.

## **6. Conclusion**

Since the field of computer vision is significantly influenced by artificial neural intelligence and especially deep learning, many researcher and developers are interested in the implementation of a suitable deep learning architecture for building footprint segmentation. One of the most important issues in the field of segmentation are inefficient and inaccurate segmentation networks which output shapes differing from the shape of ground truth.

The presented experiment aims at using Pix2Pix network to segment the building footprint. **The results of Pix2Pix on the test dataset seem to be good, but there is still much room for improvement of segmentation quality.**
>  [**On my github you will find all the code and data set related to the experiment](https://github.com/A2Amir/Pix2Pix-for-Semantic-Segmentation-of-Satellite-Images).**

## Amir Ziaee
[**A2Amir - Overview**
*Master of software engineering,Master of Web Communication and Information Technology, Data Science ,Deep Learning…*github.com](https://github.com/A2Amir)
[**Amir Ziaee - Medium**
*Read writing from Amir Ziaee on Medium. https://github.com/A2Amir/. Every day, Amir Ziaee and thousands of other voices…*medium.com](https://medium.com/@ziaee.a.a)
[**Amir Z. - Information Technology Teacher - Javan Rayaneh Institute | LinkedIn**
*Hi, I am Amir, who loves analysing big data and building smart products in the field of Machine Learning, Computer…*www.linkedin.com](https://www.linkedin.com/in/Ziaee-A-Amir/)

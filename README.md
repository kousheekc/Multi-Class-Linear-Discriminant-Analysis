# Multi-Class-Linear-Discriminant-Analysis
Python implementation of Multi Class Linear Discriminant Analysis for dimensionality reduction

In this program, I implement Fisher's Linear Discriminant to perform dimensionality reduction on datasets such as the **Iris Flower** dataset and the **Handwritten Digits** dataset. Dimensionality reduction is important because computations on lower dimensions are less computationally intensive and low dimensions also allow visual representations of the datasets which provides a better understanding of the data.  

## Requirement:
The additional libraries that were used in this program are:
* **Numpy**
* **Matplotlib**
* **Scikit-learn** (sklearn was used to simply import some well known datasets)

## Multi-class Linear Discriminant Analysis (LDA)
The primary goal in LDA is to determine suitable **direction vectors** ![\hat{w}](https://render.githubusercontent.com/render/math?math=%5Chat%7Bw%7D) such that when the higher dimension data is projected onto these direction vectors, the seperation between the various classes in maintained and maximized. This process is achieved by generating a critereon function ![J(W)](https://render.githubusercontent.com/render/math?math=J(W)) where the columns of ![W](https://render.githubusercontent.com/render/math?math=W) are the unique direction vectors ![\hat{w}](https://render.githubusercontent.com/render/math?math=%5Chat%7Bw%7D). In this case ![J(W)](https://render.githubusercontent.com/render/math?math=J(W)) is defined as ![J(W) = \frac{|W^TS_BW|}{|W^TS_WW|}](https://render.githubusercontent.com/render/math?math=J(W)%20%3D%20%5Cfrac%7B%7CW%5ETS_BW%7C%7D%7B%7CW%5ETS_WW%7C%7D) where ![S_B](https://render.githubusercontent.com/render/math?math=S_B) and ![S_W](https://render.githubusercontent.com/render/math?math=S_W) are the between scatter matrix and within scatter matrix and therefore ![W](https://render.githubusercontent.com/render/math?math=W) can be determined by differentiating the critereon function with respect to ![W](https://render.githubusercontent.com/render/math?math=W) and setting it to 0, ![\frac{dJ}{dW} = 0](https://render.githubusercontent.com/render/math?math=%5Cfrac%7BdJ%7D%7BdW%7D%20%3D%200). This gives us ![S_w^{-1}S_B\hat{w_i} = \lambda_i\hat{w_i}](https://render.githubusercontent.com/render/math?math=S_w%5E%7B-1%7DS_B%5Chat%7Bw_i%7D%20%3D%20%5Clambda_i%5Chat%7Bw_i%7D) where ![\hat{w_i}](https://render.githubusercontent.com/render/math?math=%5Chat%7Bw_i%7D) and ![\lambda_i](https://render.githubusercontent.com/render/math?math=%5Clambda_i) are the eigenvectors and eigenvalues of the matrix ![S_w^{-1}S_B](https://render.githubusercontent.com/render/math?math=S_w%5E%7B-1%7DS_B).  

Once ![W](https://render.githubusercontent.com/render/math?math=W) is determined, the data can be projected down onto the columns of ![W](https://render.githubusercontent.com/render/math?math=W).
* To project to 1D space, use the first column of ![W](https://render.githubusercontent.com/render/math?math=W)
* To project to 2D space, use the first two column of ![W](https://render.githubusercontent.com/render/math?math=W) and so on...

## Examples
The **Iris Flower dataset** contains 3 classes (varieties of Iris flowers) and each entry contains 4 features. The plot below shows the data reduced to **1D** and **2D**.
<img src="iris flowers.png">

The **Handwritten Digits dataset** contains 10 classes (numbers from 0 to 9) and each entry contains 16 features. The plot below shows the data reduced to **1D**, **2D** and **3D**.
<img src="handwritten digits.png">

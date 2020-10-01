---
layout: post
title: "Classic Video Stabilization"
categories: Bayesian Filtering
image: /thumbnail-mobile.png
author: "Huy Nguyen, Khang Vo, Phu Huynh"
meta: "Springfield"
---

# Video Stabilization

## **Introduction**

Video Stabilization is the process of estimating the undesired camera motion and wrapping the images and compensate for it. For example, the videos taken by hand-held cameras, smartphone cameras are often shaking. Thís can be done with hardware, e.g mordern cameras using OIS (Optical Image Stabilization). Beside that, there are many researchs using algorithm to against undesired vibrations, e.g deep neural network, CNNs ... But in this blog we'll not discuss deep neural net, just focus on classic techniques to solve this problem.

These are example videos before and after stabilized.

Original Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://www.youtube.com/)

Stabilized Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://www.youtube.com/)

## **Main Problem**

The main problem is that, from a arbitrary video with vibrations, how can we reduce the vibrations in that video or in other words how to stabilize it?

There are many researchs about this problem and almost of them follow 4 steps (illustrated in figure 1): motion estimation, motion compensation (or smoothing), image wrapping and finally cropping to remove empty regions at border of frames.

<p align="center">
    <img width="600"  src="/images/Steps_in_Video_stb.png"/>
    <br>
    <i>Figure 1: Steps in video stabilization process</i>
</p>

The first block is [motion estimation](#motion-estimation), used to calculate the motions of features in current frame also a transformation matrix between previous frame to current frame. Because of cameare vibrations, the motions will not smooth. Our purpose in the second block  is to smooth motions through transformation matrices of frames. In order to help you have different views in [motion compensation](#motion-compensation), we present three different approaches: [compositional local smoothing](#compositional-local-smoothing), [local matrix based smoothing](#local-matrix-based-smoothing) & [additive method](#additive-method); brief the advantages and disadvantages of each approach. [The two final steps](#wrapping-and-cropping) are pretty easy, we will brief them as shortly and easily to understand as possible.

**Mathematical notations**

There are some mathematical notations you should know before reading bellow sections.

Image function is notated $$I(x,y)$$. This function have 2 parameters $$(x, y)$$ corresponding to coordinates of pixel on $$x$$ axis and $$y$$ axis. The value function return is the intensity of that pixel.

We denote $$I_{x}$$ and $$I_{y}$$ are gradient of image corresponding in $$x$$ - axis and $$y$$ - axis.

The vectors are denoted by lower case bold Roman letters, such that $$\mathbf{u}$$. And the matrices will be denoted by upper case, e.g $$M$$. One more, we denote $$Id$$ is the identity matrix.

The notation $$H$$ is a tranformation matrix. It can be a translation, affinity, similarity, euclidiean or homography matrix.



## **Motion Estimation**

Vibrations can lead to moving abnormally of features in frames. In order to reduce these movements, we first estimate motion of features in current frame. This step consits of two smaller processes: detect features and expect transformation matrix between two adjacent frames. Lucky for us, both had been already implemented in OpenCV but we must understand both these clearly.

### **Feature Detection**

We could use Harris corner detection or SURF, SHIFT ... to detect features. The SURF and SHIFT algorithm are better than Harris in accuracy and speed, but to make it easy to understand, we'll just brief the main ideal of classic feature detection algorithm, **Harris**.

**A Corner** is a point whose local neighborhood stands in two dominant and different edge directions. In other words, a corner can be interpreted as the junction of two edges, where an edge is a sudden change in image brightness [3]. Corners are the important features in the image, and they are generally termed as interest points which are invariant to translation, rotation and illumination.

The Harris corner detector uses a window to "observe" each pixel and its neighbors, to characterizes "corner" as a point with big variation if shifting in any direction. If variation only happens for "some" directions, then the point should be "edge" and if no variation for any directions, it'll be "flat". You can see in figure 2.

<p align="center">
    <img width="400"  src="/images/corner.png"/>
    <br>
    <i>Figure 2: Intuition of Harris Corner Detector</i>
</p>

The variation can be defined as a sum of square-distance (SSD) as below, $$(u, v)$$ denotes the shift, $$w(x,y)$$ is a window function, $$I(x,y)$$ is image function, at coordinate $$(x, y)$$ image has a certain intensity. We can usually choose a rectangle or gaussian function.

$$ E(u, v) \approx \sum_{(x, y) \, \in \,W} w(x, y) \, [\,I(x + u, y+ v) - I(x, y)\,] \,^ {2} $$

The shifted intensity can be approximated by Taylor expansion:

$$ I (x + u, y + v) \approx I(x, y) + u\,I_{x} (x,y) + v\,I_{y} (x,y) + R_{2}$$

Therefore, the SSD measure is now:

$$ \begin{aligned}
E(u, v) & \approx \sum_{(x, y)\, \in \,W} w(x, y)\,[\,u\,I_{x} (x,y) + v\,I_{y} (x,y)\,]\,^{2} \\
& = \sum_{(x, y)\, \in \,W} w(x, y)\, [u \quad v] 
\left[\begin{array}{cc}
    I_{x}^{\,2}&I_{x}\,I_{y}\\
    I_{x}\,I_{y}&I_{y}^{\,2}
\end{array}\right] 
\left[\begin{array}{c}
    u\\v
\end{array}\right]\\
& =  [u \quad v] \Biggl ( \sum_{(x, y)\, \in \,W} w(x, y)\,
\left[\begin{array}{cc}
    I_{x}^{\,2}&I_{x}\,I_{y}\\
    I_{x}\,I_{y}&I_{y}^{\,2}
\end{array}\right] \Biggr)
\left[\begin{array}{c}
    u\\v
\end{array}\right]\\
    & = \mathbf{u^{T}} \, M \, \mathbf{u}.
\end{aligned}$$

Because $$M$$ is a real and symmetric matrix, it can be decomposed (using SVD) to be:

$$ M = Q \, \Lambda \, Q^{T}. $$

where $$Q$$ is orthogonal matrix and $$\Lambda$$ is diagonal matrix. So the SSD measure can be written again:

$$ \begin{aligned}
    E(\mathbf{u}) & = \mathbf{u^{T}} \, M \, \mathbf{u} = \mathbf{u^{T}} \, Q \, \Lambda \, Q^{T} \, \mathbf{u} = (Q^{\mathbf{T}} \, \mathbf{u})^{\mathbf{T} } \, \Lambda \, (Q^{\mathbf{T}} \, \mathbf{u}) \\
                & = \mathbf{u'^{T}} \, \Lambda \, \mathbf{u'} = \lambda_1 \, ||\mathbf{u'}||_2^{2} + \lambda_2 \, ||\mathbf{u'}||_2^{2}.
\end{aligned} $$

where $$\lambda_1$$ and $$\lambda_2$$ is eigen values of $$M$$. We want $$E(\mathbf{u})$$ to be big for all directions for corner so both $$\lambda_1$$ and $$\lambda_2$$ must be large. If both are small, it will be "flat". For "edge", one large eigen value and one small eigen value.

The Harris detector specially designed a response:

$$\begin{aligned}
    R & = det(M) - k \, trace(M)^{2} \\
      & = \lambda_1 \lambda_2 - k \, (\lambda_1 +\lambda_2)^{2}.
\end{aligned}$$

The paramater k is usually set to 0.04 - 0.06. If $$R$$ is large,it is corner. Otherwise negative $$R$$, it'll be edge; positive $$R$$ but small, the flat region.

<p align = "center">
    <img width="300"  src="/images/harris_region.jpg"/>
    <br>
    <i>Figure 3: Harris Region</i>
</p>


### **Transformation Matrix Expectation**

Phu

## **Motion Compensation**

### **Compositional local smoothing**


### **Local matrix based smoothing**

This method was proposed in [4], we have refered the link below so you can read more. The figure 3 present the main ideal of algorithm.

The method refered above is to accumlate past tranformation matrices to compensate current motions. This also means it generates accumulative error. The **Local matrix based smoothing** retreats this problem by just smoothing displacement from current to the neighboring frames.

There are some below definitions you should consider.

The following formulas define the compositions with the previous transformations form the current frame

$$H_{i,j;j<i} = \prod_{l=j}^{i - 1} H_{l+1,l} = H_{j+1,j}H_{j+2,j+1}...H_{i-1,i-2}H_{i,i-1} \, ,$$

with $$H_{l+1,l} = H_{l,l+1}^{-1}$$, and the compositions with the following frames as

$$H_{i,j;j>i} = \prod_{l=i}^{j} H_{l,l+1} = H_{j-1,j}H_{j-2,j-1}...H_{i-1,i}H_{i,i+1}.$$

in a neigborhood, $$N_{i} = \{ j: i - k \leq j  \leq i + k\}$$ around frame $$i$$ and hyper parameter $$k$$ is smoothing radious.

The algorithm smooths the transformation matrix of the frame by using gaussian function (the red bell curve in fig 4) and neigboring frames ' transformation matrix (e.g homography or affine). 

$$\widehat{H}_{i}(p, q) = \sum_{j = i - k}^{i+k}G_{\sigma}(i - j)H_{i,j}(p,q).$$

with $$H_{i,i}$$ *is Identity matrix*.

The $$\widehat{H}_{i}$$'ll transform the frame $$I_{i}$$ to frame $$\widehat{I}_{i}$$ (the blue frame in fig 4) and we assume that the frame $$\widehat{I}_{i}$$ after transformed is also stabilized frame $$I'_{i}$$.

So easily, we can infer $$H'_{i} = \widehat{H}_{i}^{-1}$$.

<p align = "center">
    <img src="/images/local_matrix_based.png"/>
    <i> Figure 4: Local matrix-based smoothing </i>
</p>

The steps of local matrix-based smoothing are detailed in below.

>$$\begin{aligned}
& Argorithm: \textrm{Local matrix-based} \\
&\quad\mathbf{Input} : \{H\}, \sigma, bc\\
&\quad\mathbf{Output}: \{H^{'}\}\\
&1. \quad radius \leftarrow 3 \sigma \\
&2. \quad \textrm{if } \, radius > N \textrm{ then:} \\
&3.     \qquad radius \leftarrow N \\
&4. \quad \textrm{for } i \leftarrow \textrm{to } N \textrm{ do}:\\
&5.  \qquad t_{1} \leftarrow i - radius \\
&6.  \qquad t_{2} \leftarrow i + radius \\
&7.  \qquad \textrm{if } t_{1} < 1 \textrm{ then:}\\
&8.  \qquad \quad t_{1} \leftarrow 1 \\
&9.  \qquad \textrm{if } t_{2} > N \textrm{ then:} \\
&10. \qquad \quad t_{2} \leftarrow N \\
& \qquad \textrm{// compute backward transformations}\\
&11. \qquad \textrm{if } i > 1 \textrm{ then:}\\
&12. \qquad \quad H_{i-1}^{c} \leftarrow H_{i-1,i}^{-1}\\
&13. \qquad \quad \textrm{for } j \leftarrow i - 2 \textrm{ to } t_{1} \textrm{ do:}\\
&14. \qquad \qquad H_{j}^{c} \leftarrow H_{j,j+1}^{-1}H_{j+1}^{c}\\
& \qquad \textrm{// introduce the indentity matrix}\\
&15. \qquad H_{i}^{c} \leftarrow \textrm{\mathbf{Id}}\\
&16. \qquad \textrm{if } i < N \textrm{ then:}\\
&17. \qquad \quad H_{i+1}^{c} \leftarrow H_{i,i+1}^{-1}\\
&18. \qquad \quad \textrm{for } j \leftarrow i + 2 \textrm{ to } t_{2} \textrm{ do:}\\
&19. \qquad \qquad H_{j}^{c} \leftarrow H_{j-1,j}^{-1}H_{j-1}^{c}\\
&20. \qquad \textrm{call Gaussian convolution}(\{H^{c}\}, \, \widehat{H}_{i}, \, i, \, \sigma, \, bc)\\
&21. \qquad H_{i}^{'} \leftarrow \widehat{H}_{i}^{-1}.
&\end{aligned}$$

**Result**

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://www.youtube.com/)


### **Additive method**

In this section, we will describe a simple strategy named **"Local linear matrix-based smoothing"** that avoid the compositional of matrices. The benefit of this method is that the errors produced by the compositions of matrices are not accumulated.

This method relies on the term *virtual trajectories*, where the information used for estimation the set of smooth transformations is not based on real trajectories, but on the integral of the velocity of a pixel.

* Vitual trajectory of a homography is defined as
 $$\bar{H_{i}} = Id + \sum_{i-1}^{l=1}(H_{l,l+1} - Id)$$
 or
 $$\bar{H}_{i} = \bar{H}_{i-1} + H_{i-1,i}-Id$$
* Smoothed trajectories coeficients of the matrix trajectories for i = 1,..,*N* are calculated using Gaussian smoothing. They are defined by
$$ \tilde{H_{i}}(p,q) = (G_{\sigma}*\{\bar{H_{j}}(p,q)\})_{i}
= \sum_{j=-k}^{j=k}G_{\sigma}(j)\bar{H}_{i-j}(p,q)$$
and then
$$ \hat{H}_{i} = \tilde{H}_{i} - \bar{H}_{i} + Id$$
* Finally, this method appoarch is then obtain by the rectifying transformation $$H_{i}^{'} = \bar{H}_i^{-1}$$.

The concept of virtual trajectory is clearer in the case of point, where the virtual trajectory is not given by the true displacement of the points from frame to frame but by the apparent motion of the point in each position.

The boundary conditions for this method are the same to the compositional strategy.

The step of **Local linear matrix-based smoothing** can be illustrated in the algorithm below.

>$$\begin{aligned}
&\quad\mathbf{Input} : \{H\}, \sigma, bc\\
&\quad\mathbf{Output}: \{H^{'}\}\\
&1.\quad\bar{H}_{1} \leftarrow Id\\
&2.\quad\textrm{for } i \leftarrow 2 \textrm{ to } N \textrm{ do}\\
&3.\quad\quad\bar{H}_{i} \leftarrow \bar{H}_{i-1} + H_{i-1,i}-Id\\
&4.\quad\textrm{for } i \leftarrow 1 \textrm{ to } N \textrm{ do}\\
&5.\quad\quad\textrm{Guassian convolution}(\{\bar{H}\},\{\tilde{H}_{i}\},i,\sigma,bc)\\
&6.\quad\textrm{for } i \leftarrow 1 \textrm{ to } N \textrm{ do}\\
&7. \quad\quad \hat{H}_{i} = \tilde{H}_{i} - \bar{H}_{i} + Id\\
&8. \quad\quad H^{'} = \bar{H}_i^{-1}
&\end{aligned}$$

## **Wrapping and Cropping**

### Crop and zoom strategy

A post-processing is needed to remove the empty region that appear at the border of the images. We will discuss a simple approach - Crop and Zoom. The idea is to find the largest axis-parrallel retangel that does not contain empty regions, and apply the adequate crop to all frames to remove them. 

#### Crop strategy

A fast and simple process for computing the rectangle is to iteratively project the corners of the images using the smoothed homography of each frame and update the limits of the rectangle by each corner.

The algorithm can be describe as follow:

> Current rectangle: $$x_{1}, y_{1}, x_{2}, y_{2}$$\\
At frame $$I$$ do\\
$$ \quad $$ //project and update the top-left corner\\
$$ \quad (x,y,z)^{T} \leftarrow (H_{i}^{'})^{-1}.(0,0,1)^{T}$$\\
$$ \quad$$ If $$x/z>x_{1}$$ then $$x_{1} \leftarrow x/z$$\\
$$ \quad$$ If $$y/z > y_{1}$$ then $$y_{1} \leftarrow y/z$$\\
$$ \quad$$ //project and update the other 3 corners

After iterating to the last frame, we arrive at the rectangle that is close to the largest rectangle that does not contain the empty regions.

#### Zoom strategy

Let $$(x_{m},y_{m}) =(\dfrac{x_{1}+x_{2}}{2},\dfrac{y_{1}+y_{2}}{2})$$ be the center of the rectangle we derived from the earlier step and let $$s = min(\dfrac{x_{1}+x_{2}}{n_{x}},\dfrac{y_{1}+y_{2}}{n_{y}})$$ be the scale factor from the small rectangle to the origin rectangle. We can define the **Crop and Zoom** transformation as

$$\begin{aligned}
T &= \begin{pmatrix}
1&0&x_{n}\\
0&1&y_{n}\\
0&0&1
\end{pmatrix}\begin{pmatrix}
s&0&0\\
0&s&0\\
0&0&1
\end{pmatrix}\begin{pmatrix}
1&0&-n_{x}/2\\
0&1&-n_{y}/2\\
0&0&1
\end{pmatrix}\\
&=\begin{pmatrix}
s&0&x_{m} - sn_{x}/2\\
0&s&y_{m} - sn_{y}/2\\
0&0&1
\end{pmatrix}
\end{aligned}$$

The idea behind this transformation is to move the center of the original frame to $$(x_{m},y_{m})$$ as the same time scaling our frame to match the rectangle's size. 

### Warping
This is the final step in which we will warp each frames base on the smoothed transformation and the Crop&Zoom transformation.

Notice that for a pixel location $$x$$ in the stabilized frame, we have

$$I_i^{'}(x) = I_{i}(H_{i}^{'}Tx)$$

This is because our $$H_{i}^{'}$$ and $$T$$ are used to describe the transformation from the stabilized frame back to the original frame. This relation can be used to determine the color value at each pixel for each new stabilized frame.

## **Conclusion**

In this report, we have descibed a work flow for video stabilization as well as different motion compensation strategies.

The smoothing strategies a divided into compositional and addictive approaches. Compositional approaches tend to suffer more from errors in previous steps because they are accumulated throughout the frame.
 In contrast, addictive approaches tend to suffer less because of it's increment through addiction method. 

The downsides of these appoarchs is that they are not fit to real-time implementation and can also yield bad result when the video quality is low, or the camera is too shaky, loss of perspective ,etc. There is a more power method which can eliminate these downside named [MeshFlow](http://www.liushuaicheng.org/eccv2016/), but we will discuss this in another day.

## **Reference**

[1] <https://www.ipol.im/pub/art/2017/209/>

[2] <https://medium.com/data-breach/introduction-to-harris-corner-detector-32a88850b3f6>

[3]  Konstantinos G. Derpanis (2004). The harris corner detector. York University.

[4] <http://mmlab.ie.cuhk.edu.hk/archive/2006/01634345.pdf>

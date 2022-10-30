---
layout: post
title: "Probabilistic Multiple - View 3D Reconstruction"
tagline: "Variational Methods"
categories: Mathematics
image: /thumbnail-mobile.png
author: "Bao-Huy Nguyen"
meta: "Springfield"
---

## Probabilistic Multiple - View 3D Reconstruction

There are numerous methods for reconstructing a real object into mesh such as voxel carving which processes independently the input images or structure from motion. However, in this blog, we would like to introduce to you another solution for this ill-posed inverse computer vision problem. This method is volumetric approach, where each voxel is assigned two probability values for being in or out of 3D object. Let's start.

![](/images/3DReconstruction/multiple-view-3D-recon.png)

## Preliminaries

We represent our local 3D environment which includes the 3D object needed to reconstruct by a volume $$V$$ which is defined by:

$$V := [v_{11}, v_{12}] \times [v_{21}, v_{22}] \times [v_{31}, v_{32}] \subset \mathbb{R}^3$$

and each $$v_{ij} \in \mathbb{R}$$.

Discretely, our volume $$V$$ is a set of voxels having a resolution $$N_x \times N_y \times N_z$$:

$$V := \left\{ \left( \begin{matrix}
    v_{11} + i \cdot \dfrac{v_{12} - v_{11}}{N_x} \\
    v_{21} + j \cdot \dfrac{v_{22} - v_{21}}{N_y} \\
    v_{31} + k \cdot \dfrac{v_{32} - v_{31}}{N_x} \\
\end{matrix}\right)
\Bigg\vert \quad
\begin{matrix}
    i = 0,..., N_x - 1 \\
    j = 0,..., N_y - 1 \\
    k = 0,..., N_z - 1 \\
\end{matrix}\right\}$$

Let the sequence of input RGB images be:

$$I_1, ..., I_n: \Omega \mapsto \mathbb{R}^3$$

and the known model view project matrices corresponding to RGB images respectively be:

$$\pi_1, ..., \pi_n : V \mapsto \Omega$$

where $$\Omega \subset \mathbb{R}^2$$.

## Generative Model

Let's have a look at the below graphical model below:

![](/images/3DReconstruction/3DRecon_graphical_model.png)

where:

* $$R$$: foreground or background region
* $$\textbf{c}$$: color of a pixel
* $$\textbf{v}$$: voxel
* $$u$$: shape of 3D object
* $$n$$: the index of a frame.

This graphical model may be biased to the heuristic of designers but in some point it is still valid. The random variable $$\textbf{v}$$ in $$V$$ depends on not only the shape of 3D object (obviously) but also foreground or background region $$R$$ that its projection to image planes belongs to. Whereas, the variable color $$\textbf{c}$$ undoubtedly depends on the region $$R$$.

## Maximizing A Posteriori

Our goal is to **maximize a posteriori** $$P(u \mid \{I_1, ..., I_n\})$$ or in other words $$P(u \mid \textbf{v}, \textbf{c}_{1...n})$$.

To derive optimization equation for that, first let's start with the joint distribution for a given voxel $$P(u, \textbf{v}, R_{1...n}, \textbf{c}_{1...n})$$:

$$P(u, \textbf{v}, R_{1...n}, \textbf{c}_{1...n}) = P(\textbf{v} \mid u, \textbf{v}, R_{1...n}) P(\textbf{c}_{1...n} \mid R_{1...n}) P(R_{1...n}) P(u)$$

We divide both sides by $$P(\textbf{c}_{1...n}) = \underset{i\in{\{f, b\}}}{\sum}P(\textbf{c}_{1...n} \mid R_{i, 1...n}) P(R_{i, 1...n})$$ to get:

$$P(u, \textbf{v}, R_{1...n} \mid \textbf{c}_{1...n}) = P(\textbf{v} \mid u, \textbf{v}, R_{1...n}) P(R_{1...n} \mid \textbf{c}_{1...n}) P(u)$$

Next, we marginalize over $$P(R_{1...n})$$:

$$P(u, \textbf{v} \mid \textbf{c}_{1...n}) = \underset{i \in \{ j, b\}}{\sum} P(\textbf{v} \mid u, \textbf{v}, R_{i, 1...n}) P(R_{i, 1...n} \mid \textbf{c}_{1...n}) P(u)$$

With assumption that $$P(\textbf{v})$$ is constant, we can omit the random variable $$\textbf{v}$$ by divide $$P(u, \textbf{v} \mid \textbf{c}_{1...n})$$ by $$P(\textbf{v})$$:

$$P(u \mid \textbf{v}, \textbf{c}_{1...n}) \propto \underset{i \in \{ j, b\}}{\sum} P(\textbf{v} \mid u, \textbf{v}, R_{i, 1...n}) P(R_{i, 1...n} \mid \textbf{c}_{1...n}) P(u)$$

The prior $$P(u)$$ is also eliminated for the sake of simplicity. finally, the posterior over the volume becomes likelihood:

$$\begin{aligned}
    P(u \mid \Omega_3) &\propto \underset{\textbf{v} \in \Omega_3}{\prod} P(u \mid \textbf{v}, \textbf{c}_{1...n}) \\
    &\propto \underset{\textbf{v} \in \Omega_3}{\prod}  \left\{ \sum P(\textbf{v} \mid u, \textbf{v}, R_{i, 1...n}) P(R_{i, 1...n} \mid \textbf{c}_{1...n}) \right\}
\end{aligned}$$

where:

$$P(R_{i, 1...n} \mid \textbf{c}_{1...n}) = \dfrac{P(\textbf{c}_{1...n} \mid R_{i, 1...n}) P(R_{i,1...n})}{\underset{j \in \{f, b\}}{\sum}P(\textbf{c}_{1...n} \mid R_{j, 1...n}) P(R_{j,1...n})}$$

and:

$$\begin{aligned}
    P(\textbf{v} \mid u, R_{f, 1...n}) &= \dfrac{u}{\zeta_f} \\
    P(\textbf{v} \mid u, R_{b, 1...n}) &= \dfrac{1 - u}{\zeta_b} 
\end{aligned}$$

with:

$$u(\textbf{v}) = \begin{cases}
    1 \qquad \textbf{v} \in \text{3D object} \\
    0 \qquad \text{otherwise}
\end{cases}$$

and $$\zeta_f, \zeta_b$$ being the average number of voxels (over n views) that project to a foreground pixel (with $$P(\textbf{c} \mid R_f) \gt P(\textbf{c} \mid R_b)$$ and  $$\textbf{c} = I_m(\pi_m(\textbf{v}))$$) and a background pixel, respectively.

Now, one question has come is that how can we compute two probabilities $$P(\textbf{c}_{1...n} \mid \textbf{R}_{i,1...n})$$ with $$i \in \{ f, b\}$$. A straightforward way is to treat $$\{P(\textbf{c}_k \mid R_{i, k})\}$$ independently. Based on visibility, the probability of a voxel in foreground is equivalent to the probability that all cameras observe this voxel in foreground, whereas the probability of a voxel being a part of background is that at least one camera sees it as background. This way is pretty similar to voxel carving when a voxel is considered as background if its project on background.

$$\begin{aligned}
    P(\textbf{c}_{1...n} \mid R_{f, 1...n}) &= \prod_{k=1}^nP(\textbf{c}_k \mid R_{f, k}) \\
    P(\textbf{c}_{1...n} \mid R_{b, 1...n}) &= 1 - \prod_{k=1}^n\{1 - P(\textbf{c}_k \mid R_{b, k}) \}
\end{aligned}$$

We can describe $$P(\textbf{c}_k \mid R_{i, k})$$ by Gaussian distribution or simply with a histogram.

However, these joint probabilities are not calculated easily since their products tend to be very small. To overcome this, they are rewritten:

$$\begin{aligned}
    P(\textbf{c}_{1...n} \mid R_{f, 1...n}) &= \operatorname{exp} \left( \dfrac{\sum_{k = 1}^n\operatorname{log}P(\textbf{c}_k \mid R_{f, k})}{n} \right) \\

    P(\textbf{c}_{1...n} \mid R_{b, 1...n}) &= 1 - \operatorname{exp} \left( \dfrac{\sum_{k = 1}^n1 - \operatorname{log}( 1 - P(\textbf{c}_k \mid R_{b, k}))}{n} \right) \\
\end{aligned}$$

The probability of foreground and background region over
n views becomes $$P(R_{f, 1...n}) = \dfrac{\bar{\eta_f}}{\eta}$$ with hf being the average value of foreground area $$\eta_f$$ over the n views. $$P(R_{b, 1...n})$$ follows analogously ($$\eta$$ is the whole image area).

So the final posterior equation is:

$$E = P(u \mid \Omega_3) = \prod_{\textbf{v} \in \Omega_3}\{ uP_i + (1 - u)P_o\}$$

There are many numerical optimization methods to maximize this posteriori (basically it is likelihood) such as taking logarithm and using gradient descent, projected gradient descent or Gauss Newton Strategy. However for globally solvable formulation, Victor et al [[1]](#1) replaced logarithmic opinion pool by a linear opinion pool to fuse pixel - wise posteriors and added weighted surface regularization term:

$$E = \sum_{\textbf{v} \in \Omega_3}\{uP_i + (1 - u) P_o + \alpha |\nabla u| \}$$

where $$\alpha$$ is a tunable parameter.

For fast and global convergence, Victor et al [[1]](#1)  used continuos min cut / max flow [[3]](#3):

$$E = \underset{p_t, p_s, p}{\operatorname{max}} \,\, \underset{u}{\operatorname{min}}\sum_\textbf{v}\{ u\cdot p_t + (1 - u) \cdot p_s + u \cdot \operatorname{div}p\}$$

such that:

$$\begin{aligned}
    u &\in [0, 1] \\
    p_s(\textbf{v}) &\lt P_i(\textbf{v}) \\
    p_t(\textbf{v}) &\lt P_o(\textbf{v}) \\
    \mid p(\textbf{v}) \mid &\lt \alpha
\end{aligned}$$

To know the details of the algorithm, visit [[3]](#3) and [[4]](#4)

## Results

Some may ask how can we know $$\{\pi_k\}$$ in the real environment?

To answer this, we use **ARCore** which simultaneously localizes the position of the camera in 3D world. The setting in mobile phone is straightforward.

* First, users will scan floors or flatten areas in order to detect the plane and estimate the camera poses.
* Next step is to put a large cube (volume) on the detected plane and put the object needed to scan inside the cube.
* Finally, to achieve the best result, users have to go around object to observe all the perspectives of it.

[![3D Reconstruction on Android](https://res.cloudinary.com/marcomontalbano/image/upload/v1667125203/video_to_markdown/images/youtube--gPBLQ9BkSnI-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://www.youtube.com/watch?v=gPBLQ9BkSnI "3D Reconstruction on Android")

To comprehend the whole system, we recommend you read [[1]](#1), [[2]](#2), [[3]](#3), [[4]](#4).

## References

<a id="1">[1]</a> Prisacariu, Victor Adrian, et al. "Real-time 3d tracking and reconstruction on mobile phones." IEEE transactions on visualization and computer graphics 21.5 (2014): 557-570.

<a id="2">[2]</a> Kolev, Kalin, Thomas Brox, and Daniel Cremers. "Fast joint estimation of silhouettes and dense 3D geometry from multiple images." IEEE transactions on pattern analysis and machine intelligence 34.3 (2012): 493-505.

<a id="3">[3]</a> Yuan, Jing, Egil Bae, and Xue-Cheng Tai. "A study on continuous max-flow and min-cut approaches." 2010 ieee computer society conference on computer vision and pattern recognition. IEEE, 2010.

<a id="4">[4]</a> Chambolle, Antonin. "An algorithm for total variation minimization and applications." Journal of Mathematical imaging and vision 20.1 (2004): 89-97.

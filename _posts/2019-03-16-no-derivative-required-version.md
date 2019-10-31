---
layout: post
title: "Kalman filter, Extended Kalman Filter, and Unscented Kalman Filter - Part 3"
tagline: "Hidden charm of Gaussian distribution"
categories: Bayesian Filtering
image: /thumbnail-mobile.png
author: "Phuong Hoang"
meta: "Springfield"
---

In Extended Kalman Filter (EKF) part, we have discovered the beauty of Gaussian distribution by linearly approximating the non linear transforms $$x_k = f(x_{k-1})$$ and $$y_k = g(x_k)$$. However, EKF requires Jacobina matrices $$F_x$$ and $$H_x$$, respectively. Meanwhile, many real applications neither have a closed form for functions $$f(.)$$ and $$g(.)$$ nor Jacobian matrices.

It's time for Unscented Kalman Filter (UKF). First, we wil have a look at Unscented Transform to answer several questions such as how to choose sigma points, and weights for the points. Then, we will apply the transform in the context of filtering problem.

Form 3 sigma points as follows:

$$ X_0 = \mu $$
$$ X_1 = \mu + \sigma $$
$$ X_2 = \mu - \sigma $$

Select weights so that

$$\sum_i W_i = 1 $$

$$\mu = \sum W_i X_i$$

$$\sigma^2 = \sum_i W_i (X_i - \mu)^2$$

For random variable in a vector form $$x \sim N(m,P)$$ the standard deviation $$\sigma$$ is the Cholesky factor $$L = \sqrt P$$ or $$P = LL^T$$

Then the sigma points can be formed using columns of $$L$$ with c is a selectively positive constant

$$X_0 = m$$
$$X_i = m+ cL_i$$
$$X_{n+i} = m - cL_i$$

For transformation $$y = g(x)$$ the approximation of distribution of $$y$$ can be done as follows:

$$\mu_y = \sum_i W_i g(X_i)$$
$$\sigma_y = \sum_i W_i (g(X_i) - \mu_y)(g(X_i) - \mu_y)^T$$

Assume that the filtering distribution of previous step is Gaussian

$$p(x_{k-1} \mid y_{1:k-1}) \approx N(x_{k-1} \mid m_{k-1}, P_{k-1})$$

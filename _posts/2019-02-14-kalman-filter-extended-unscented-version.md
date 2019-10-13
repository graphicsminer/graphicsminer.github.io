---
layout: post
title: "Kalman filter, Extended Kalman Filter, and Unscented Kalman Filter"
tagline: "Hidden charm of Gaussian"
categories: Bayesian Filtering
image: /thumbnail-mobile.png
author: "Phuong Hoang"
meta: "Springfield"
---

# Kalman Filter, Extendted version and No derivative required version
Before jumping into Kalman filter, we should recall Gaussian distribution and its effectiveness when digital computers not ready yet.

Gaussian pdf function for a random variable

**x** ~ $$ N $$ ( $$ m $$, $$P$$)

$$N( x | m, P) = \frac{1}{(2\pi)^{n/2} P^{1/2}} \exp{(-\frac{1}{2} (x - m)^T P^{-1} (x- m))} $$

Let **x** and **y** have the Gaussian densities and have linear relationship

$$ P($$ **x** $$) $$  $$ = N($$ **x** $$|$$ **m**, **P** $$)$$

$$ P($$ **y** $$|x $$ $$) $$  $$ = N($$ **y** $$|$$ **Hx**, **R** $$)$$


The joint and marginal distribution of **x** and **y** are given as follows:

$$\begin{pmatrix} x \\\ y \end{pmatrix}$$  ~  $$N \left( \begin{pmatrix} m\\\ Hm\end{pmatrix}, \left( \begin{pmatrix} P && PH^{T}\\\ HP && HPH^T + R\end{pmatrix}  \right)$$

**y** ~ $$N(Hm, HPH^T + R) $$

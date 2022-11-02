---
layout: post
title: "Linear Algebra (Part 1)"
tagline: "Linear Algebra"
categories: Mathematics
image: /thumbnail-mobile.png
author: "Bao-Huy Nguyen"
meta: "Springfield"
---

## Basic Definitions in Linear Algebra

### Vector

In $2$-dimension space, we learned that each vector $\textbf{v}$ has two components $x$ and $y$, is denoted $\textbf{v} = [x, y]^T \in \mathbb{R}^2$.

In $3$-dimension space, a 3D vector $\textbf{v}$ includes 3 components $\textbf{v} = [x, y, z]^T \in \mathbb{R}^3$.

Expanding this idea, in $n$-dimension space, a vector $\textbf{v}$ will have $n$ components $\textbf{v} = [v_1, v_2, ..., v_n]^T \in \mathbb{R}^n$

### Independence

<u>Definition</u>:

A set of vectors $\textbf{v}_1, \textbf{v}_2, ..., \textbf{v}_n \in \mathbb{R}^n$ are independent if no combination gives zero vector $\textbf{0}$ (except the zero combination - all $c_i = 0$)

$$c_1\textbf{v}_1 + c_2\textbf{v}_2+ ... + c_n\textbf{v}_n \neq \textbf{0}$$

(except $c_1 = c_2 = ... = c_n = 0$)


Independent             |  Not Independent
:-----------------------:|:-------------------------:
![](/images/Linear_Algebra_Basics/independence.png)  |  ![](/images/Linear_Algebra_Basics/dependence.png)

### Space

<u> Definition </u>:

Vectors $\textbf{v}_1, \textbf{v}_2, ..., \textbf{v}_l$ **span** a space means the space consists of all the linear combinations of these vectors.

<u>Example</u>:

Two vectors $\textbf{v}_1 = [1, 1, 0]$ and $\textbf{v}_2 = [2, 1, 1]$ span a space and vector $\textbf{v}_3 = 2\textbf{v}_1 + \textbf{v}_2 = [4, 3, 1]$ is a part of the space.

### Basics

<u> Definition </u>:

**Basics** of a space is a sequence of vectors $\textbf{v}_1, \textbf{v}_2, ..., \textbf{v}_d$ with 2 properties:

1. They are independent.
2. They span the space.

<u> Example </u>:

Space $\mathbb{R}^3$ have:

* One basics is $\left[\begin{matrix}
    1 \\
    0 \\
    0
\end{matrix}\right]$, $\left[\begin{matrix}
    0 \\
    1 \\
    0
\end{matrix}\right]$, $\left[\begin{matrix}
    0 \\
    0 \\
    1
\end{matrix}\right]$.

* Another basics is $\left[\begin{matrix}
    2 \\
    0 \\
    0
\end{matrix}\right]$, $\left[\begin{matrix}
    0 \\
    3 \\
    0
\end{matrix}\right]$, $\left[\begin{matrix}
    0 \\
    0 \\
    4
\end{matrix}\right]$.

* But not this $\left[\begin{matrix}
    2 \\
    2 \\
    5
\end{matrix}\right]$, $\left[\begin{matrix}
    1 \\
    1 \\
    3
\end{matrix}\right]$ since they do not span $\mathbb{R}^3$. Even though we add another vector $\left[\begin{matrix}
    3 \\
    3 \\
    8
\end{matrix}\right]$, it does not help since they are still dependent. The solution in here is to put a vector that is not in the plane created by these two vectors.

### Dimension of a space

<u> Definition </u>:

The number of vectors in basics for the space is called DIMENSION of THE SPACE.

<u> Example </u>:

* The space spanned by column vectors of $A = \left[\begin{matrix}
    1 & 2 & 3 & 1 \\
    1 & 1 & 2 & 1 \\
    1 & 2 & 3 & 1
\end{matrix}\right]$ has 2 dimensions (which is also its rank).

* The space spanned by column vector of $A = \left[\begin{matrix}
    -1 \\
    -1 \\
    1 \\
    0
\end{matrix}\right]$ has 1 dimensions (which is also its rank).
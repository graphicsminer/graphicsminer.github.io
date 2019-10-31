---
layout: post
title: "Robotics : Kinematics and Dynamics from Optimization Perspectives"
tagline: "Inverse Kinematics and Inverse Dynamics"
categories: Robotics
image: /thumbnail-mobile.png
author: "Phuong Hoang"
meta: "Springfield"
---

##Forward Kinematics
Forward kinematics describes the mapping between joint coordinates **q** and the end-effector configuration $$X_e$$ of the robot.

$$X_e = X_e(q)$$

The mapping function can be obtained through the evaluation of transformation from the base frame of the robot to the end-effector frame.

$$T(q) = T_{01} \prod_{k=1}^{n_j} T_{k-1,k}(q_k) T_{n_j E}$$

In real practices, when talking about fixed base robots, the base frame of the robot is not moving w.r.t an inertial frame - $$T_{01} $$ and $$T_{n_j E}$$ are constant. In contrast, the floating based robots, the base frame dynamically moves.

We can express the forward kinematics problem as follows:
Given $$q$$ configuration of robot joints, we need to estimate the end-effector configuration. However, in real practices, we are looking for solving the inverse kinematics problem discussed later in this post.

Before talking about inverse problems, we should understand about Jacobian matrix - a bit mathematically speaking parts of this post.
We are normally interested in local changes of some variables based on other parameters (i.e  stock prices  or bitcoin values based on Trump tweets). So we do in robotics. We do care about how end-effector configuration changes based on the alternation of the joint configuration.
To formulate it mathematically, a common approach is to linearize the forwards kinematics.

$$X_e + \partial {X_e} = X_e (q+\partial q) = X_e (q) + \frac{X_e(q)}{\partial q} \partial q + O(\partial q^2)$$

so

$$ \partial {X_e} \approx \frac{X_e(q)}{\partial q}\partial q = J_{eA}(q) \partial q$$

where

$$ J_{eA}(q) = \begin{pmatrix} \frac{\partial x_1}{\partial q_1} \cdots \frac{\partial x_1}{\partial q_{nj}} \\\ \vdots \\\ \frac{\partial x_m}{\partial q_m} \cdots \frac{\partial x_m}{\partial q_{nj}} \end{pmatrix}$$

The Jacobian is useful in both kinematics and dynamics of robotic systems. It relates the differences from joint to end-effector space.

##Analytical and Kinematic Jacobian

### Analytical Jacobian
$$  \dot{X_e} = J_{eA}(q) \dot{q}$$

It relates time-derivatives of config parameters to generalized velocities. It depends on the parameterization selection in 3D. For example, using the cartesian coordinates for representation of positions is different from using the cylindrical coordinates. It results in different analytical Jacobian matrices.

###Geometric Jacobian

$$ w_e = \begin{pmatrix} v_e \\\ w_e \end{pmatrix} = J_{e0}(q)\dot{q}$$

It relates the end-effector velocity to generalized velocities including linear and angular velocities.


## Inverse Differential Kinematics vs Inverse Kinematics

###Inverse Kinematics
The goal of inverse kinematics is to find a mapping function from the desired end-effector configuration $$X_e^\star$$ to joint configuration $$q$$:

$$q = q(X_e^\star)$$

given $$ \partial {X_e} \approx J_{eA}(q) \partial q$$

###Inverse Differential Kinematics
As we have known, the geometric Jacobian matrix  $$J_e0{q}$$ performs a simple mapping between $$\dot{q}$$ and the end-effector velocity $$w_e$$. However, in real practices, we can decide the end-effector velocity manually or automatically, the robot should automatically estimate the joint velocity $$\dot q$$ to control your robot.

If we can compute the pseudo-inverse of the Jacobian, we can calculate the joint velocity as follows:
$$\dot q = J_{e0}^{+}w_e^{\star}$$

By taking The Moore-Penrose pseudo inverse, the above solution minimizes the least square error
$$|| w_e^{\star} - J_{e0} \dot{q} ||^2$$

In the case $$J_{e0} $$ close to singularities, we can use a damped version of the Moore-Penrose pseudo-inverse which is similar to minimize the error $$|| w_e^{\star} - J_{e0} \dot{q} ||^2 + \lambda ||\dot{q}||^2$$

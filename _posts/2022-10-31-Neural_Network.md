---
layout: post
title: "Neural Networks"
---

# Neural Networks 
## Why neural network?
### A recap on perceptron
In classic perceptron decision function, the decision function is not differentiable. $$y=sign(w^Tx)$$
To solve this problem, we use logisitic function to make perceptron differentiable.$$y=\frac{1}{1+e^{-w^Tx}}$$
So why do we want to make perceptron differentiable in the first place?  
This is because we will be able to use gradient descent to solve the logistic regression.
### Problem with Linear Model
Let's take a simple example. In XOR, a linear model doesn't work because it's not possible to seperate the 1's and 0's by a line. But does it mean that it's not going to work in another space? The answer is NO. We are able to transform the data from the original space to a feature space by feature transform. Think of how contour map works. The data in a contour map seems to be not linearly seperable, but if it's transformed into a higehr dimension, it's possible to seperate the data in the feature space.  
This is where neural networks come in. In the next sections, we will explore how neural networks is applied to learn the good features, and also talk a bit about deep neural networks.

## Transform to feature space
In linear regression, remember that we have this equation $$h_j=\sum_{i=1}^{d}w_{ij}x_i$$

To make it non-linear, our first attemp is to simply multiply a matrix, so that $$y=w_y^tw_hx$$
However, we observe that this approach doesn't work out because $$w_y^tw_h$$ ends up being another linear model, and our goal of tranforming to a nonlinear model is not achieved. To achieve our goal, we can add a non-linear function $$h_j=g(\sum_{i=1}^{d}w_{ij}x_i)$$
There are many functions to apply, some popular ones are tanh, sigmoid, and relu. We will be using sigmoid in the next section.  
A side notes, 'Deep' Learning simply means more layers, which means that there are many feature transformations taken place to compute y.

## Gradient descent for neural network training
We will be using the sigmoid as our activation function here, $$sigmoid(x)\ =\ \frac{1}{1+e^{-x}}$$To use gradient descent, we need the derivative of sigmoid by the use of quotient rule. We get $$\frac{d\ sigmoid(x)}{dx}=sigmoid(x)(1-sigmoid(x))$$
Since gradient descent is used to minize the error, we need to find the deritivative of error. To do so, let's list out the error function first, which is also called the L2 norm. t is the actual value, and y is the estimated value. 1/2 is added here to make result more elegant.
$$E=\frac{1}{2}(t-y)^2$$
The partial fraction of error with regard to one weight w_k is 
$$\frac{dE}{dy}\frac{dy}{ds}\frac{ds}{dw_k}$$
By doing simple derivatives on Error, activation function(sigmoid), and the linear combiation of weights s, we get 
$$-(t-y)\ y(1-y)\ h_k$$
Now with this equation, we are able to compute every single weight. For every single weight, gradient descent will tell us how to chanage the weight to remove that error. The gradient descent equation based on the derivative of error is $$w_k = w_k + \alpha(t-y)y(1-y)h_k$$
Alpha here is the learning rate, and we will be moving in the opposite direction of the derivative of the error to minimize the error through gradient descent.

## Conclusion
We need neural networks so that we can have a nonlinear model that can classify more complex data sets, or even simple data sets that can't be seperated by linear model such as XOR. We learned that in a differentiable perceptron model, we are able to use gradient descent to find out the weights that can minimize the errors. In real-life, most data sets are nonlinear and complex, so neural networks have a lot of applications in real-life, and it's a very important machine learning method.

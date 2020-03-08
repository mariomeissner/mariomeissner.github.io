---
layout: single
title: "A Gentle Introduction to Backpropagation in Neural Networks"
categories: technical
---

Backpropagation is the keystone to neural networks. Without them, we wouldn't be where we are today. Yet, it is one of the concepts that usually takes more time to grasp and fully understand.

A full two layer neural network written in `numpy` can be as short as the following:

```python
for i in range(10000):

  # forward pass
  # (batch,features) @ (features,hidden) + (1,hidden) = (batch,hidden)
  zh    = X.dot(w1) + b1
  h     = sigmoid(zh)
  # (batch,hidden) @ (hidden,output) + (1,output) = (batch,output)
  zpred = h.dot(w2) + b2
  pred  = softmax(zpred)

  # loss
  loss = cross_entropy(pred, y)

  # Print loss every now and then
  if i % 500 == 0:
      print(f"Iteration {i:0>4}, loss: {loss:6.3f}")

  # backpropagation
  grad_zpred = pred - y
  grad_h     = grad_zpred.dot(w2.T)
  grad_w2    = h.T.dot(grad_zpred)
  grad_b2    = grad_zpred.sum(axis=0)
  grad_zh    = grad_h * sigmoid(h, deriv=True)
  grad_w1    = X.T.dot(grad_zh)
  grad_b1    = grad_zh.sum(axis=0)

  # update weights
  w1 -= lr * grad_w1
  b1 -= lr * grad_b1
  w2 -= lr * grad_w2
  b2 -= lr * grad_b2
```

Most courses proptly explain to you how the forward pass works and how we use gradient descent to update the weights of the network until we reach an optimum. However, the formulas needed to calculate those weights are oftentimes left in the dark. In this post I will attempt to explain where they come from in a way that I believe most people will easily understand. I do assume some level of understanding with regards to tensor operations though.

Courses like Stanford's [CS231N](http://cs231n.stanford.edu/syllabus.html) cover this topic quite in detail, so be sure to have a look at their video lectures for in depth explanations. However, I have found that even after watching the lecture on backpropagation, I was still a little confused on how exactly you end up obtaining the python code that I show above. The last step was still missing for me. I will attempt to cover everything, including the last mile.

In order to tackle this issue, lets first understand the concept of _computational graphs_. They are [DAGS](https://en.wikipedia.org/wiki/Directed_acyclic_graph), with inputs and an output, and each node is an operation.

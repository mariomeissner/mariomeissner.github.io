# Bias vs Variance clearly explained

As most things in this blog will be, this entry attempts to explain bias vs variance in a way similar to the way it made click for me in my brain. Several sources will explain it in several ways, and rarely will one alone help you understand something the way you need it. Although I will try to distill what I learned into a single blog post, you might want to consider consulting other sources to complement it.

# Intuition

Let's start by building an intuition and seeing the relationship between bias/variance with over/underfitting. Machine learning models have an underlying complexity level that allow them to be more or less flexible while trying to represent the dataset. Let's imagine a regression model trying to predict a dataset with just one feature. X are the features and Y is the real valued label that we are trying to teach to our model. 

(insert pic: simple_complex.png)

For the sake of clarity, let's assume the red points are a sample of points from a ground truth that looks more like this: 

(insert pic with more points and a curve) Neither the simple nor the complex model are very correct, since one of them generalizes too much while the other is too precise.

The simple model is linear, it can only fit data by drawing a line. It's precision on the training data is rather low. If this model could talk, it would tell us that the data has a lot of noise, and the underlying ground truth is in reality the line. It assumes that if we take a different sample, with different points (both being representative of the same ground trouth) the location of the points would be quite different, but nontheless the model could explain it by using the same or a very similar line. It means there is a low variance in the model if we take a different sample of points to train it. It also means the general error the model is making is quite high, to be able afford this low variance. This means the bias of the model is high. 

(insert gif of sample changing)

The complex model is fitting a very high degree polinomial on the data, perfectly explaining each point. If this model could talk, it would be saying that the ground truth is a quite complex polinomial.The error on this sample is very low, but what would happen if we train it by using a different sample? The model would need to change completely to be able to fit the new data. It would give us a very high error on a different sample. Once trained, the error will be low again. This means it's bias is low. It's not biased towards some specific distribution or shape, it will do whatever it takes to be unbiased and train itself with maximum precision. However, the variance of this model is very high. It needs to change a lot to be able to adapt to different samples.

The source that helped me most was this excellent post by Scott Fortmann: http://scott.fortmann-roe.com/docs/BiasVariance.html
## Maximum Likelihood Estimation

![Maximum Likelihood (ML) vs. REML. Linear Mixed Model via Restricted… | by  Nikolay Oskolkov | Towards Data Science](https://miro.medium.com/v2/resize:fit:1058/1*RkrBD8WJsc43q9ndDhiWXg.png)

[TOC]

### Summary

最大似然估计是一种通过找到参数值使得观察到的数据概率最大化的统计估计方法。

Maximum Likelihood Estimation is a statistical estimation method that seeks parameter values to maximize the probability of observed data.

### Key Takeaways

1. 最大似然估计是一种统计学中常用的参数估计方法，用于估计概率分布模型中的未知参数。
2. 它通过找到使得给定观察到的数据的概率最大化的参数值，从而得到最优的参数估计结果。
3. 最大似然估计假设样本数据是独立同分布的，因此可以将联合概率密度函数简化为各个样本数据的概率密度函数的乘积。
4. 通常为了方便计算，会对似然函数取对数，转换成求和的形式。
5. 最大似然估计寻找满足对数似然函数的导数等于零的参数值，这些值是似然函数的极值点，即最大似然估计的结果。
6. 在实际应用中，可以根据具体问题选择合适的概率分布模型和参数形式，然后使用最大似然估计来估计模型参数。
7. 最大似然估计在统计学和机器学习中广泛应用，用于估计回归模型、分类模型以及其他概率模型的参数。
8. 在样本量较大的情况下，最大似然估计通常能够得到较准确的参数估计。
9. 理论上，最大似然估计的估计值具有一致性和渐近正态性等良好性质。
10. 尽管最大似然估计是一种常用的参数估计方法，但在某些情况下，可能会存在偏差较大或者估计值不稳定的问题，因此在实际应用中也需要综合考虑其他估计方法。

---

1. Maximum Likelihood Estimation is a widely used statistical method for estimating unknown parameters in probability distribution models.
2. It seeks the parameter values that maximize the probability of the observed data, providing the optimal parameter estimates.
3. MLE assumes that the sample data is independently and identically distributed, allowing for the simplification of the joint probability density function into the product of individual data probabilities.
4. To simplify calculations, logarithms are often taken to transform the likelihood function into a summation form.
5. MLE finds parameter values by setting the derivative of the logarithm of the likelihood function to zero, yielding the maximum likelihood estimates as the solutions.
6. In practical applications, suitable probability distribution models and parameter forms are chosen, and MLE is then used to estimate the model parameters.
7. MLE is extensively applied in statistics and machine learning for estimating parameters in regression models, classification models, and other probabilistic models.
8. With a sufficiently large sample size, MLE typically provides accurate parameter estimates.
9. Theoretically, MLE estimators possess desirable properties such as consistency and asymptotic normality.
10. Despite being a common estimation method, MLE may suffer from biases or instability in some cases, necessitating the consideration of alternative estimation methods in practical applications.

### Interview Questions

#### 1. MLE基础

##### 1.1 什么是最大似然估计（MLE）？它在统计学中的作用是什么？

答案：MLE是一种用于估计概率模型中未知参数的方法。它通过寻找使得给定观察数据的概率最大化的参数值，得到最优的参数估计结果。在统计学中，MLE是一种重要的参数估计方法，广泛应用于回归分析、分类模型、概率分布参数估计等问题。

##### 1.2 MLE和最小二乘估计（OLS）之间有什么区别？在什么情况下应该使用哪种方法？

答案：MLE和OLS都是常用的参数估计方法，但适用的情况和假设不同。MLE假设数据是来自某个概率分布，通过最大化似然函数来估计参数。OLS则是在回归分析中用于估计线性模型的参数，假设误差项服从独立同分布的正态分布。

通常情况下，当数据满足正态分布假设且模型是线性的，可以使用OLS。当数据满足其他概率分布假设或者模型非线性时，可以考虑使用MLE。

#### 2. 似然函数

##### 2.1 解释似然函数（Likelihood Function）和对数似然函数（Log-Likelihood Function）之间的关系。

答案：似然函数是关于模型参数的函数，表示给定参数下观察数据的概率。对数似然函数是似然函数取对数后得到的函数。这样做的目的是简化计算，因为对数函数可以将概率的乘积转换为对数的求和，便于求导和计算。

对数似然函数和似然函数在最大化过程中得到相同的极值点，因此在求解MLE时通常使用对数似然函数。

#### 3. 一致性与渐进性

##### 3.1 什么是MLE的一致性和渐进正态性？

答案：一致性是指当样本量趋于无穷时，MLE的估计值会收敛到真实参数值。换句话说，随着数据量的增加，MLE的估计结果越来越接近真实参数值。

渐进正态性是一种性质，它表明当样本量趋于无穷时，MLE的估计值的分布会趋近于一个正态分布。这个性质在很多情况下方便了对MLE估计结果的推断和置信区间的计算。

#### 4. 常见应用

##### 4.1 MLE在逻辑回归中是如何应用的？

答案：在逻辑回归中，我们希望通过最大似然估计来估计回归系数。假设我们有二分类数据，对于每个样本，逻辑回归模型预测其属于某个类别的概率。然后，通过最大化似然函数来找到使观察到的数据在模型下概率最大化的回归系数值。一般使用迭代优化算法（如梯度下降）来求解最大似然估计问题。

#### 5. MLE的优缺点

##### 5.1 最大似然估计的优点和缺点是什么？

答案：最大似然估计是一种有很多优点的估计方法。它是渐进无偏的，当样本量充分大时，可以获得较准确的估计结果。此外，MLE有很好的统计性质，如一致性和渐进正态性，使得在大样本下可以进行统计推断。

然而，MLE也有一些缺点。首先，当样本量较小时，可能会导致估计结果不稳定。其次，MLE对于选择不同的概率模型或者参数化形式，可能会导致不同的估计结果。此外，MLE对于异常值较为敏感，可能会导致估计偏离真实参数值。因此，在实际应用中，需要谨慎地选择适合的概率模型和参数形式，并对结果进行验证和敏感性分析。
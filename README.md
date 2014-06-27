OPERA-lib
========

Operalib is a structured learning and prediction library for [Python](https://www.python.org/) utilising operator-valued kernels (OVKs). OVKs are an extension of scalar kernels into matrix-valued kernels, allowing prediction of several targets simultaneously while encoding the output structure with the operator-valued kernel.

The library implements structured OVK regression and classification with various sparsity constraints (L2, L1, elastic net, group lasso, sparse group lasso) using proximal algorithms.

We aim at providing an easy-to-use standard implementation of operator-valued kernel methods. Operalib is designed for close compatilibity to [Scikit-learn](http://scikit-learn.org/) interface and conventions. It utilises [Numpy](http://www.numpy.org/), [Scipy](http://www.scipy.org/) and [Matplotlib](http://matplotlib.org/) as underlying libraries.

For structured learning using max-margin methods, conditional random fields or structured SVM's check the excellent [PyStruct](https://pystruct.github.io) library.

The project is developed by the [AROBAS](https://www.ibisc.univ-evry.fr/arobas) group of the [IBISC laboratory](https://www.ibisc.univ-evry.fr/en/start) of the University of Evry, France. 

The library is based on publications

* Néhémy Lim, Florence d'Alché-Buc, Cédric Auliac, George Michailidis (2014): Operator-valued Kernel-based Vector Autoregressive Models for Network Inference, (in revision)
* Lim, Senbabaoglu, Michalidis and d'Alche-Buc (2013): OKVAR-Boost: a novel boosting algorithm to infer nonlinear dynamics and interactions in gene regulatory networks. Bioinformatics 29 (11):1416-1423.
* Brouard, d'Alché-Buc and Szafranski (2011): Semi-Supervized Penalized Output Kernel Regression for Link Prediction. In ICML 2011.

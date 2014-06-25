OPERA-lib
========

Operalib is a structured learning and prediction library for [Python](https://www.python.org/) utilising operator-valued kernels (OVKs). OVKs are an extension of scalar kernels into matrix-valued kernels, allowing prediction of several targets simultaneously while encoding the output structure with the operator-valued kernel.

The module implements structured OVK regression and classification with various sparsity constraints (L2, L1, elastic net, group lasso, sparse group lasso) using proximal algorithms.

We aim at providing a standard implementation of the most relevant operator-valued kernel methods in an easy-to-use library. Operalib is designed for close compatilibity to Scikit-learn interface and conventions. It utilises [Numpy](http://www.numpy.org/), [Scipy](http://www.scipy.org/) and [Matplotlib](http://matplotlib.org/) as underlying libraries.

The project is maintained by the [AROBAS group](https://www.ibisc.univ-evry.fr/arobas) at University of Evry, France.

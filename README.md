OPERA-lib
========

Operalib is a structured learning and prediction library for [Python](https://www.python.org/) utilising operator-valued kernels (OVKs). OVKs are an extension of scalar kernels into matrix-valued kernels, allowing prediction of several targets simultaneously while encoding the output structure with the operator-valued kernel.

The library implements structured OVK regression and classification with various sparsity constraints (L2, L1, elastic net, group lasso, sparse group lasso) using proximal algorithms.

We aim at providing an easy-to-use standard implementation of operator-valued kernel methods. Operalib is designed for close compatilibity to [Scikit-learn](http://scikit-learn.org/) interface and conventions. It utilises [Numpy](http://www.numpy.org/), [Scipy](http://www.scipy.org/) and [Matplotlib](http://matplotlib.org/) as underlying libraries.

For structured learning using max-margin methods, conditional random fields or structured SVM's check the excellent [PyStruct](https://pystruct.github.io) library.

The project is maintained by the [AROBAS](https://www.ibisc.univ-evry.fr/arobas) group at University of Evry, France.

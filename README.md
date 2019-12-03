# StructEdit: Learning Structural Shape Variations

![Overview](https://github.com/daerduoCarey/structedit/blob/master/images/teaser.png)

**Figure 1.** Edit generation and transfer with StructEdit. We present StructEdit, a method that learns a distribution of shape differences between structured objects that can be used to generate a large variety of edits (first row); and accurately transfer edits between different objects and across different modalities (second row). Edits can be both geometric and topological.

## Introduction
We learn local shape edits (shape deltas) space that captures both discrete structural changes and continuous variations. Our approach is based on a conditional variational autoencoder (cVAE) for encoding and decoding shape deltas, conditioned on a source shape. The learned shape delta spaces support shape edit suggestions, shape analogy, and shape edit transfer, much better than StructureNet, on the PartNet dataset.

## About the paper

Our team: 
[Kaichun Mo](https://cs.stanford.edu/~kaichun),
[Paul Guerrero](http://paulguerrero.net/),
[Li Yi](https://cs.stanford.edu/~ericyi/),
[Hao Su](http://cseweb.ucsd.edu/~haosu/),
[Peter Wonka](http://peterwonka.net/),
[Niloy Mitra](http://www0.cs.ucl.ac.uk/staff/n.mitra/),
and [Leonidas J. Guibas](https://geometry.stanford.edu/member/guibas/) 
from 
Stanford University, University College London (UCL), University of California San Diego (UCSD), King Abdullah University of Science and Technology (KAUST), Adobe Research, Google Research and Facebook AI Research.

Arxiv Version: https://arxiv.org/abs/1911.11098

Project Page: https://cs.stanford.edu/~kaichun/structedit/

## Citations

    @article{Mo19StructEdit,
        Author = {Mo, Kaichun and Guerrero, Paul and Yi, Li and Su, Hao and Wonka, Peter and Mitra, Niloy and Guibas, Leonidas},
        Title = {{StructEdit}: Learning Structural Shape Variations},
        Year = {2019},
        Eprint = {arXiv:1911.11098},
    }

## About this repository

This repository provides data and code as follows.


```
    data/                   # contains data, models, results, logs
    code/                   # contains code and scripts
         # please follow `code/README.md` to run the code
    stats/                  # contains helper statistics
```

The code is developed with Python 3.6, PyTorch 1.1.0, and CUDA 9.0.

## Questions

Please post issues for questions and more helps on this Github repo page. We encourage using Github issues instead of sending us emails since your questions may benefit others.

## License

MIT Licence

## Updates

* [Nov 25, 2019] Data and Code released.


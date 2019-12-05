Please fill in [this form](https://docs.google.com/forms/d/e/1FAIpQLSc9g2XEGMY-etdlCcy4p6ZQ4nNStaERV-ivehGYzn-FLhvBpg/viewform?usp=sf_link) to download the SynShapes synthetic dataset.

## About this repository

```
    syn_chair/                          # hierarchy of graphs data for synthetic chairs
            [anno_id].json              # storing the tree structure, 
                                        # and all part oriented bounding box parameters)
            [train/test].txt            # train/test split
            test200.txt                 # the anno_id for testing shapes we used for reporting numbers in paper
    syn_stool/
    syn_sofa/
``` 

## Cite

Please cite [StructEdit](https://cs.stanford.edu/~kaichun/structedit/) if you use this data.

    @article{Mo19StructEdit,
        Author = {Mo, Kaichun and Guerrero, Paul and Yi, Li and Su, Hao and Wonka, Peter and Mitra, Niloy and Guibas, Leonidas},
        Title = {{StructEdit}: Learning Structural Shape Variations},
        Year = {2019},
        Eprint = {arXiv:1911.11098},
    }


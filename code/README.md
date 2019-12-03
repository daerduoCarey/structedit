# StructEdit Experiments
This folder includes the StructEdit experiments for AE edit reconstruction, VAE edit generation, and VAE edit transfer using both box-shape on both PartNet/StructureNet data and Synshapes synthetic dataset.

## Before start
To train the models, please first go to `data/partnetdata/` folder and download the training data. 
To test over the pretrained models, please go to `data/models/` folder and download the pretrained checkpoints.
To download the pre-generated results, please go to `data/results/` folder and download the data.
All the resources to download share the same [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSc9g2XEGMY-etdlCcy4p6ZQ4nNStaERV-ivehGYzn-FLhvBpg/viewform?usp=sf_link).

## Dependencies
This code has been tested on Ubuntu 16.04 with Cuda 9.0, GCC 5.4.0, Python 3.6.5, PyTorch 1.1.0, Jupyter IPython Notebook 5.7.8. 

Please run
    
    pip3 install -r requirements.txt

to install the other dependencies.

Then, install https://github.com/rusty1s/pytorch_scatter by running

    pip3 install torch-scatter

## Compute and Visualize the Shape Neighborhoods
You can visualize the computed shape neighbors using `vis_neighbors.ipynb`.

## AE Edit Reconstruction
To train the network from scratch, for example, run

    bash scripts/train_ae_chair_cd.sh

to train a model on chair box-shapes using chamfer-distance neighborhoods.

To test the model, run

    bash scripts/eval_recon_ae_chair_cd.sh

You can use `vis_recon.ipynb` to visualize the AE edit reconstruction results.

Notice that the evaluation code, by default, requires the baseline method of StructureNet result directory to output the quantitative comparison.
Put the pre-generated results under `../data/results` folder.

To train the StructureNet baseline from scratch, run

    bash scripts/train_structurenet_ae_chair.sh

Notice that for fair comparison, we use the no-edge version StructureNet in the StructEdit experiments.

To test the pre-trained model, run

    bash scripts/eval_recon_structurenet_ae_chair.sh

The evaluation code may overwrite the downloaded pre-generated results.

You can use `vis_recon.ipynb` to visualize the results in Python Jupyter Notebook.

## Run for Table/StorageFurniture, or Structure-distance Neighborhood
Simple replace the `Chair` to `Table` or `StorageFurniture` in the training or evaluation scripts, and replace `cd` with `sd` for using structure distance neighborhood.

## Normalization Factors to Obtain Table Numbers in Paper
To obtain the numbers shown in paper, please divide the reconstruction errors by the normalization factors defined in `../stats/norm_params.txt`.
The normalization factors are computed and defined to be the average distance of neighbors from the source shape.

These numbers are just used to make the quantitative numbers have reasonable and similar scales for chamfer and structure metrics.
They are not affecting the results at all.

We use the shared normalization factors when measuring chamfer distances, while use separate ones when measuring structure distances regarding quality and converage scores.

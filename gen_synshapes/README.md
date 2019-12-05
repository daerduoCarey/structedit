First, run 

    python random_gen_chair.py [output_dir] [num_gen]

to generate [num\_gen] synthetic chair shapes, each with 96 variants, to [output\_dir] folder.

Replace `chair` with `sofa` or `stool` to generate sofa and stool synthetic shapes.

Then, run

    python convert_to_structedit_format.py

to generate the StructEdit json format.

You may use `../code/vis_synshapes.ipynb` to visualize the SynShapes data on Jupyter Notebook.

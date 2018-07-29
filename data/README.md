# Notes about the `data` directory structure

Each subdirectory contains files for a specific dataset (e.g. `CUB`). <br>
`xian2017` subdirectory contains the zero-shot data and splits provided by Xian (CVPR 2017).<br>
The `meta` subdirectory contains additional meta data that is required for LAGO but is not provided by Xian.<br>

**For example:**

`CUB/meta/attribute_names_with_semantic_group.txt` contains the attribute names preceeded by their semantic group name, in the format `<id> groupname::attributename` (e.g. `174 has_under_tail_color::yellow`).

`CUB/meta/class_descriptions_by_attributes.txt` is a text file containing the matrix values 
for the mean class-description, provided with the dataset. Xian provides a L2 normalized version of this file. But L2 normalization removes the probabilistic meaning
 of the attribute-class description, which is a key ingredient of LAGO.

`CUB/meta/classes.txt` is a text file containing class ids and class names according to their order in `class_descriptions_by_attributes.txt`.




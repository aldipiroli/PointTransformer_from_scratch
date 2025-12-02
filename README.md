# Point Transformer from scratch 
Implementing from scratch the paper "Point Transformer" ICCV 2021.

### Clone and install dependencies
``` 
git clone https://github.com/aldipiroli/PointTransformer_from_scratch
pip install -r requirements.txt
``` 

### Training and Evaluation
``` 
python train.py point_transformer/config/config_cls.yaml
``` 

### Results
>Part Semantic Segmentation results on the ShapeNet Core validation set after ~20 epochs.

![](assets/output_part_semseg.gif)

> Classification results on the ShapeNet Core validation set after ~30 epochs.

![](assets/output_cls.gif)

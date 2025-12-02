# Point Transformer from scratch 
Implementing from scratch the paper "Point Transformer" ICCV 2021.

### Clone and install dependencies
``` 
git clone https://github.com/aldipiroli/PointTransformer_from_scratch
pip install -r requirements.txt
``` 

### Training and Evaluation
Classification:
``` 
python train.py point_transformer/config/config_cls.yaml
``` 
Segmentation:
``` 
python train.py point_transformer/config/config_segm.yaml
``` 

### Results
>Part Semantic Segmentation results on the ShapeNet Core validation set after ~20 epochs. Top: Predictions, Bottom: Ground Truth.

![](assets/output_part_semseg.gif)

> Classification results on the ShapeNet Core validation set after ~30 epochs. Format: Prediction/Ground Truth.

![](assets/output_cls.gif)

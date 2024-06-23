# Bi-Classifier with Neighborhood Aggregation for Unsupervised Domain Adaptation
##Basic setting
###File path
>The file folder path under the  LUHP\examples\domain_adaptation\image_classification\


### Requirements
	pytorch 1.7.1
	numpy 1.21.2
	torchvision 0.8.2
	tqdm 4.62.3
	timm 0.4.12
	scikit-learn 1.0.2
---

for ImageCLEF:
```
CUDA_VISIBLE_DEVICES=0  python bcna_ImageCLEF.py data/image_CLEF -d Office31 -s i -t p -a resnet50 --epochs 10 -i 1000 --seed 0 --log logs/bcna/ImageCLEF_I2P
```
or
```
bash run_ImageCLEF.sh
```


for OfficeHome:
```
CUDA_VISIBLE_DEVICES=0  python bcna_officehome.py data/office-home -d OfficeHome -s A -t C -a resnet50 --epochs 20 -i 1000 --seed 1 --trade-off 0.01 --log logs/bcna/OfficeHome_Ar2Cl
```
or
```
bash run_officehome.sh
```

### Acknowledgement
Our implementation is based on the Transfer Learning Library.

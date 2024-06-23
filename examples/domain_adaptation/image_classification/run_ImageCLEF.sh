python bcna_ImageCLEF.py data/image_CLEF -d Office31 -s c -t i -a resnet50 --epochs 10 -i 1000 --seed 1 --log logs/bcna/ImageCLEF_C2I
python bcna_ImageCLEF.py data/image_CLEF -d Office31 -s i -t p -a resnet50 --epochs 10 -i 1000 --seed 1 --log logs/bcna/ImageCLEF_I2P
python bcna_ImageCLEF.py data/image_CLEF -d Office31 -s i -t c -a resnet50 --epochs 10 -i 1000 --seed 1 --log logs/bcna/ImageCLEF_I2C
python bcna_ImageCLEF.py data/image_CLEF -d Office31 -s p -t i -a resnet50 --epochs 10 -i 1000 --seed 1 --log logs/bcna/ImageCLEF_P2I
python bcna_ImageCLEF.py data/image_CLEF -d Office31 -s c -t p -a resnet50 --epochs 10 -i 1000 --seed 1 --log logs/bcna/ImageCLEF_C2P
python bcna_ImageCLEF.py data/image_CLEF -d Office31 -s p -t c -a resnet50 --epochs 10 -i 1000 --seed 1 --log logs/bcna/ImageCLEF_P2C


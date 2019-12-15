# To download the dataset:

# To process the dataset:

# To train:

Run command.
```bash 
python train.py --batch_size 128 --gpus 1 --epochs (20 default) --lr (3e-4 default)
```
This code was run on 1 GeForce RTX 2080Ti with the data stored on hard drive.
The total train time was 6 hours for 20 epochs 

# To test:

python test.py --model <model path> --batch_size 128 --gpus 1

Takes about 10 minutes to test.

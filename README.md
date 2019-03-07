# Apnea Detection
a deep learning implement about apnea detection by pytorch framework with 1d cnn(1d convolution)
 
## Environment
- Python 3.7
- Pytorch 1.0.1
- pandas 0.24.1
- numpy 1.15.4
- scipy 1.2.1

## dataset
The data set was collected from a smart mattress based on piezoelectric ceramics at zhoupu hospital 
through yueyang medical technology co., LTD. 

apnea events are derived from PSG

Each sample consists of a 10-second 50hz signal and a label

`train.csv` has 3808 samples and `test.csv` has 960 samples, The number of positive and negative samples is the same

columns 1 to 500 represent 10-second 50hz signal

column 501 represent label. 1 for apnea sample; 0 for normal sample

## Model
see *reference paper* 

## performance
something wrong with my model, and I will update my model later.

Now, with`epoch_num=2`and`batch_size=32`, the test accuracy is 49.5% and total loss is 1.051

## TODO
- modify the model
    - add BN layer between every convolution layers
    - try some others optimizer
    - adjust parameters
- fix dataset read bug
- adjust code structure to make it more readable

## reference
Urtnasan E , Park J U , Joo E Y , et al. 
Automated Detection of Obstructive Sleep Apnea Events from a Single-Lead Electrocardiogram Using a Convolutional Neural Network[J]. 
Journal of Medical Systems, 2018, 42(6):104.
# Deep Transfer Learning of CNN using Time-Frequency Representation of Current Signals 
In this repository, the codes related to the paper 'Deep Transfer Learning Approach Using Filtered Time-Frequency Representations of Current Signals for Bearing Fault Detection in Induction Machines' (doi: 10.13140/RG.2.2.35815.15521) are given. The paper is currently under submission and review in a journal.  

## Dataset
The data supporting this study's findings have been obtained thanks to a collaboration with Tallinn University of Technology. Due to privacy and ethical restrictions, the data are not publicly available. However, any current signal measurements from a drive tested with and without bearing faults should be sufficient for fine-tuning. In the present case, the data have been acquired at 20kHz.

The experimental setup includes a tested induction motor (IM), a load machine, and a torque transducer with an encoder. The test motor is a 7.5 kW machine operating at a grid voltage of 50 Hz. An ABB ACS600 industrial drive controls the load motor, allowing for the application and regulation of variable load torque to the test IM. Further details can be find in the paper. 

The extraction method of the time-frequency representation (TFR) is also detailed in the paper. Particularly, we extracted minimum 1888 TFR as training samples to fine-tune different CNN. 

## Algorithms 
In this folder, the following code can be found: 
- image_generation.m (MATLAB): this file is used to generate TFR of current signals using a band-stop filter to filter out the fundamental as performed in the paper. 
- pipeline.py: PipelineTorch has been built as Python class and implements a deep learning training and inference pipeline using PyTorch.
- train_run_model.ipynb: A Jupyter notebook which (1) splits the dataset into training, test and validation sets, (2) fine-tunes the CNN of choice detailed in pipeline.py, (3) saves the trained models into a .pth format, (4) print the accuracy of the model on the test set. 

The codes have been run using an in-house university cluster of Aalto University, using a single GPU. 

## Installation
To set up the required dependencies, run:
```
pip install -r requirements.txt
```

## Contact
The codes were majorily written by Alireza Nemat Saberi and further extended by Nada El Bouharrouti (nada.elbouharrouti@aalto.fi). 

## Citation 
```
N. E. Bouharrouti, A. N. Saberi, K. D. H. Khan, K. Kudelina, M. U. Naseer, and A. Belahcen, 
"Deep Transfer Learning Approach Using Filtered Time-Frequency Representations of Current Signals for Bearing Fault Detection in Induction Machines," 
(preprint), 2025. doi: 10.13140/RG.2.2.35815.15521.
```

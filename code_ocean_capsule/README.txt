This is a toy version of what we did the main paper with open source environments and models.


System requirements: a general-purpose machine (e.g., AWS r5d.4xlarge) with 128 GB of memory and 16 CPUs will run this code with no issues.

Installation guide: See the environment/postinstall file with the libraries needed, we list them here for completeness also.

```
#!/usr/bin/env bash
set -e
apt-get update
apt-get install -y swig

pip3 install --upgrade pip
pip install toml
pip install numpy
pip3 install torch torchvision torchaudio
pip install gym'[box2d]'
pip install tqdm
pip install gym==0.24.0
pip install gym-notices==0.0.7
pip install scikit-learn
pip install numpy==1.26.4
pip install matplotlib
'''

Demo
* Run
* Will output a gif in the results which shows the car driving with our CW-Net and the concept explanations printed on the title of the images. You will also get the loss plots, accuracy plot, and some reward data and error data. The error data is the difference between CW-Net and the original black box in Mean squared error, which should be close to zero. The reward is the enviornment reward achieved by CW-Net, we expect this to be around 220 mean to be as good as the black box original policy.
* Runtime: a few mintues

Instructions for use NA




======= Meta Instructions =====

This is a simple toy domain of a self-driving car to illustrate the paper's algorithm on open source environments. There are three concepts, "drive straight", "turn left", "turn right"

python train_cwnet.py

Note that when you run this code the training can sometimes get stuck in local minima, you want the losses to look like this around epoch 50. You may need to run it a few times to get this result (on average we found it works on around > 50% of the runs)

Epoch 50/50, Loss CE: 0.3304, Loss MSE: 0.2487, Accuracy: 87.97%

The important number is the Loss MSE

The results will show a gif which nicely summarises the algorithm working. You will also find a confusion matrix for classifying the three concepts 0/1/2, which are "straight road", "left turn", "right turn". Lastly, there are losses for the concept predictions and MSE for mimicking the original black-box labels. We also included numpy arrays for the average rewards of CW-Net and its error compared to the original black box as mean squared error between the two.
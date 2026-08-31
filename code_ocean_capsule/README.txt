CW-Net toy demo (Code Ocean capsule)
====================================

This is a toy version of what we did in the main paper, using open-source
environments and models. It is a simple toy domain of a self-driving car
(OpenAI Gym CarRacing) that illustrates the paper's algorithm. There are
three concepts: "drive straight", "turn left", "turn right", of which we
manually labelled 15,000+ examples.

Data
----
The training data lives in data/ (scenario0.npy ... scenario19.npy,
real_actions.pkl, X_train.pkl). The data ships with the Code Ocean capsule.
It is NOT included in the git repository (see .gitignore); to run this demo
locally, download the capsule from Code Ocean, which includes data/.

System requirements
-------------------
Any general-purpose machine will run this code; the published capsule was
configured with 16 CPUs and 128 GB of memory, but far less is required.
No GPU is needed.

Installation
------------
The capsule environment is built by environment/Dockerfile +
environment/postInstall (see REPRODUCING.md for the Docker instructions).

To set up locally instead:

```
bash code/setup.sh          # creates and populates a virtualenv
```

or install the dependencies directly:

```
sudo apt-get install swig   # needed to build box2d
pip install -r code/requirements.txt
```

Demo
----
* Run:

```
cd code
bash run                    # equivalent to: python -u main.py
```

* Output (written to ../results/): a GIF showing the car driving with our
  CW-Net and the concept explanations printed in the title of the images;
  loss plots; an accuracy plot; a confusion matrix for classifying the three
  concepts 0/1/2 ("straight road", "left turn", "right turn"); and numpy
  arrays with the average rewards of CW-Net and its error compared to the
  original black box (mean squared error between the two, which should be
  close to zero).
* Runtime: a few minutes.

Training notes
--------------
When you run this code, training can sometimes get stuck in a local minimum.
You want the losses to look like this around epoch 50:

```
Epoch 50/50, Loss CE: 0.3304, Loss MSE: 0.2487, Accuracy: 87.97%
```

The important number is the Loss MSE: if it is around 0.25 the run was
successful. You may need to run the code a few times to get this result (on
average we found it works on more than 50% of runs). A successful run
achieves a mean simulation reward above 200 (typically around 220), matching
the original black-box policy; main.py prints "Successful training!" or
"Failed training..." accordingly.

Attribution
-----------
The PPO CarRacing agent (ppo.py, memory.py, games/carracing.py) and the
pre-trained agent weights (weights/agent_weights.pt) are adapted from
Jinay Jain's MIT-licensed "deep-racing" project:
https://github.com/JinayJain/deep-racing
See THIRD_PARTY_LICENSES.txt.

License
-------
Apache License 2.0 (see LICENSE), except for the third-party components
listed above.

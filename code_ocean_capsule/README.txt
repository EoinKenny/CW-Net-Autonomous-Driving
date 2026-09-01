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
real_actions.pkl, X_train.pkl, and the original obs_train.pkl export). The
complete data export is included in this git repository and in the Code Ocean
capsule. It is released under CC0 1.0 Universal; see data/LICENSE.

System requirements
-------------------
Any general-purpose machine will run this code; the published capsule was
configured with 16 CPUs and 128 GB of memory, but far less is required.
No GPU is needed.

Installation
------------
The capsule environment is built by environment/Dockerfile +
environment/postInstall (see REPRODUCING.md for the Docker instructions).
The published image and the tested local setup both use Python 3.9.

For a local macOS or Linux setup, install Conda and run the following commands
from the repository root:

```
bash code_ocean_capsule/code/setup.sh
conda activate "$PWD/.venv-codeocean"
```

The setup script creates an isolated Python 3.9 environment, installs SWIG
from conda-forge for the Box2D build, and installs the pinned dependencies in
code/requirements.txt. The environment is local-only and ignored by git.
Set CWNET_ENV_PREFIX before invoking setup.sh to use a different location.

pip 22.1.2 is intentional: newer pip versions reject the legacy metadata in
Gym 0.24.0. Do not upgrade pip inside this environment.

Demo
----
* Run from the repository root:

```
conda activate "$PWD/.venv-codeocean"
cd code_ocean_capsule/code
MPLBACKEND=Agg SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 bash run
```

The environment variables make plotting and CarRacing work on headless
machines. `bash run` is equivalent to `python -u main.py`.

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

A complete local verification with the pinned environment produced a final
training MSE of 0.2501, simulation rewards of 232.3, 206.3, and 212.5, and a
mean reward of 217.03.

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

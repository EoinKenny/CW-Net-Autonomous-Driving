"""Original CW-Net model definitions that ran on the autonomous vehicle.

These are the serial and parallel CW-Net architectures described in the paper
(Fig. 2 / Extended Data Fig. 1), excerpted from the on-vehicle code with only
light edits (the imports below, this docstring, and a spelling fix), so the
file can be imported and inspected. "PW-Net" (prototype-wrapper network) was the
development-time name of CW-Net; the class names are kept as they ran on the
car. The models are not runnable standalone: they wrap a frozen black-box IRL
planner whose 128-dimensional latent state is the input `x`, and the planner
cannot be released for intellectual-property reasons (see the paper's Code
Availability statement).

Concept vector layout (num_concepts = 8):
  [:3]  steering (Left / Right / Straight)  - softmax over 3 classes
  [3:5] speed (Stop / Slow)                 - softmax over 2 classes
  [5:]  ASV, Intersection, Close            - independent sigmoids

Note that in the serial model the ranker consumes the RAW concept logits;
the softmax/sigmoid activations are applied in place afterwards, only for the
returned concept predictions.
"""

import torch
import torch.nn as nn


class PWNet_Serial_Balanced_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_size = 128
        self.num_concepts = 8
        self.expand = 4

        # Classifier layer (multi-label)
        self.classifier = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size),
            nn.BatchNorm1d(self.latent_size),
            nn.ReLU(),
            nn.Linear(self.latent_size, self.num_concepts),
        )

        # Ranker layer
        self.ranker = nn.Sequential(
            nn.Linear(self.num_concepts, self.num_concepts * self.expand),
            nn.BatchNorm1d(self.num_concepts * self.expand),
            nn.ReLU(),
            nn.Linear(self.num_concepts * self.expand, 1),
        )

    def forward(self, x):
        concept_logits = self.classifier(x)
        pwnet_rankings = self.ranker(concept_logits)

        concept_logits[:, :3] = torch.softmax(concept_logits[:, :3], dim=1)
        concept_logits[:, 3:5] = torch.softmax(concept_logits[:, 3:5], dim=1)
        concept_logits[:, 5:] = torch.sigmoid(concept_logits[:, 5:]) 

        return pwnet_rankings, concept_logits




class PWNet_Parallel(nn.Module):
    def __init__(self):

        super().__init__()
        self.latent_size = 128
        self.num_concepts = 8

        # left right straight
        self.steering_head = nn.Sequential(
            nn.Linear(self.latent_size, (self.latent_size // 2)),
            nn.BatchNorm1d((self.latent_size // 2)),
            nn.ReLU(),
            nn.Linear((self.latent_size // 2), 3),
        )
        # stop slow
        self.speed_head = nn.Sequential(
            nn.Linear(self.latent_size, (self.latent_size // 2)),
            nn.BatchNorm1d((self.latent_size // 2)),
            nn.ReLU(),
            nn.Linear((self.latent_size // 2), 2),
        )

        self.asv_head = nn.Sequential(
            nn.Linear(self.latent_size, (self.latent_size // 2)),
            nn.BatchNorm1d((self.latent_size // 2)),
            nn.ReLU(),
            nn.Linear((self.latent_size // 2), 1),
        )

        self.intersection_head = nn.Sequential(
            nn.Linear(self.latent_size, (self.latent_size // 2)),
            nn.BatchNorm1d((self.latent_size // 2)),
            nn.ReLU(),
            nn.Linear((self.latent_size // 2), 1),
        )

        self.close_head = nn.Sequential(
            nn.Linear(self.latent_size, (self.latent_size // 2)),
            nn.BatchNorm1d((self.latent_size // 2)),
            nn.ReLU(),
            nn.Linear((self.latent_size // 2), 1),
        )

    def forward(self, x):
        # 500k dataset concepts
        steering_logits = self.steering_head(x)
        speed_logits = self.speed_head(x)
        asv_logits = self.asv_head(x)
        intersection_logits = self.intersection_head(x)
        close_logits = self.close_head(x)

        concept_predictions = torch.cat(
            (steering_logits, speed_logits, asv_logits, intersection_logits, close_logits), dim=1
        )

        concept_predictions[:, :3] = torch.softmax(concept_predictions[:, :3], dim=1)
        concept_predictions[:, 3:5] = torch.softmax(concept_predictions[:, 3:5], dim=1)
        concept_predictions[:, 5:] = torch.sigmoid(concept_predictions[:, 5:])

        dummy_rankings = concept_predictions[:, :1]
        return dummy_rankings, concept_predictions

"""Toy self-driving demo of the CW-Net algorithm (Code Ocean capsule entrypoint).

Trains a CW-Net (concept classifier + ranker) on latent states of a pre-trained
PPO CarRacing agent (weights/agent_weights.pt, from JinayJain/deep-racing), then
runs the wrapped agent in simulation and saves losses, a concept confusion
matrix, rewards, and a GIF of the first simulation to ../results/.

Inputs (../data/): scenario{0..19}.npy (concept labels), real_actions.pkl,
X_train.pkl (latent states). The complete data export ships with both this git
repository and the Code Ocean capsule; see code_ocean_capsule/README.txt.

Success criterion: mean simulation reward > 200 (typically ~220, matching the
original black-box policy) and final train MSE loss around 0.25. Training can
land in a local minimum; see README.txt.
"""

import pickle
import os
# Pin CPU: ppo.py binds its device at import, while CW-Net below stays on CPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import toml
import random
import numpy as np
import torch.nn as nn

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, TensorDataset
from games.carracing import RacingNet, CarRacing
from ppo import PPO
from torch.distributions import Beta
from tqdm import tqdm


def main():

    set_seed(0)

    # All outputs (plots, arrays, GIF) go to ../results; on Code Ocean the
    # directory pre-exists, on a fresh local checkout it must be created first.
    os.makedirs("../results", exist_ok=True)

    # Load and preprocess data
    labels = []
    for scenario in range(20):
        y = np.load(f"../data/scenario{scenario}.npy")
        labels.append(y.tolist())
    with open('../data/real_actions.pkl', 'rb') as fp:
        actions = pickle.load(fp)
    with open('../data/X_train.pkl', 'rb') as fp:
        X = pickle.load(fp)
    actions = actions[:len(labels)]
    X = X[:len(labels)]


    # Prepare data tensors
    action_labels = []
    concept_labels = []
    latent_data = []
    for i in range(len(actions)):
        for j in range(len(actions[i])):
            action_labels.append(actions[i][j])
            concept_labels.append(labels[i][j])
            latent_data.append(X[i][j])
    action_labels = torch.tensor(action_labels, dtype=torch.float32)
    concept_labels = torch.tensor(concept_labels, dtype=torch.long)
    latent_data = torch.tensor(latent_data, dtype=torch.float32)
    print('Data size:', action_labels.shape, concept_labels.shape, latent_data.shape)



    # Split data into training and testing sets
    latent_train, latent_test, concept_train, concept_test, action_train, action_test = train_test_split(
        latent_data, concept_labels, action_labels, test_size=0.2, random_state=42
    )
    # Create DataLoaders for batching
    train_dataset = TensorDataset(latent_train, concept_train, action_train)
    test_dataset = TensorDataset(latent_test, concept_test, action_test)

    g = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=g)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)    


    # Define CW-Net architecture
    class CWNet(nn.Module):
        
        def __init__(self, input_dim, concept_dim, action_dim):
            super(CWNet, self).__init__()
            self.latent_size = input_dim
            self.num_concepts = 3  # concept_dim
            self.expansion = 0.1
            self.output_size = action_dim
            self.tanh = nn.Tanh()
            self.relu = nn.ReLU()
            
            # Concept Classifier layer
            self.classifier = nn.Sequential(
                nn.Linear(self.latent_size, int(self.latent_size*self.expansion)),
                nn.BatchNorm1d(int(self.latent_size*self.expansion)),
                nn.ReLU(),
                nn.Linear(int(self.latent_size*self.expansion), self.num_concepts),
            )

            # Ranker layer
            self.ranker = nn.Sequential(
                nn.Linear(self.num_concepts, int(self.latent_size*self.expansion)),
                nn.BatchNorm1d(int(self.latent_size*self.expansion)),
                nn.ReLU(),
                nn.Linear(int(self.latent_size*self.expansion), self.output_size),
            )

            
        def output_act_func(self, p_acts):    
            """
            Use appropriate activation functions for the problem at hand
            Here, tanh and relu make the most sense as they bin the possible output
            ranges to be what the car is capable of doing.
            """
            p_acts.T[0] = self.tanh(p_acts.T[0])  # steering between -1 -> +1
            p_acts.T[1] = self.relu(p_acts.T[1])  # acc > 0
            p_acts.T[2] = self.relu(p_acts.T[2])  # brake > 0
            return p_acts

        
        def forward(self, x):
            concept_logits = self.classifier(x)        
            cwnet_rankings = self.ranker(concept_logits)        
            return self.output_act_func(cwnet_rankings), concept_logits
        
        

    # Initialize model, loss functions, and optimizer
    input_dim = latent_data.shape[1]
    concept_dim = len(torch.unique(concept_labels))
    action_dim = action_labels.shape[1] if len(action_labels.shape) > 1 else 1
    model = CWNet(input_dim, concept_dim, action_dim)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    # criterion_mse = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)



    # Training loop
    num_epochs = 50
    train_losses_ce = []
    train_losses_mse = []
    test_losses_ce = []
    test_losses_mse = []
    accuracies = []



    for epoch in range(num_epochs):
        model.train()
        train_loss_ce = 0.0
        train_loss_mse = 0.0
        correct = 0
        total = 0
        
        for latent_batch, concept_batch, action_batch in train_loader:
            
            optimizer.zero_grad()
            cwnet_rankings, cwnet_logits = model(latent_batch)
            
            loss_ce = criterion_ce(cwnet_logits, concept_batch)
            loss_mse = criterion_mse(cwnet_rankings, action_batch) * 100
            loss = loss_mse + loss_ce
                    
            loss.backward()
            optimizer.step()
            train_loss_ce += loss_ce.item()
            train_loss_mse += loss_mse.item()

            # Calculate accuracy
            _, predicted = torch.max(cwnet_logits, 1)
            total += concept_batch.size(0)
            correct += (predicted == concept_batch).sum().item()

        train_losses_ce.append(train_loss_ce / len(train_loader))
        train_losses_mse.append(train_loss_mse / len(train_loader))
        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss CE: {train_loss_ce/len(train_loader):.4f}, Loss MSE: {train_loss_mse/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")
        scheduler.step()

        
    # Evaluation loop
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        test_loss_ce = 0.0
        test_loss_mse = 0.0
        correct = 0
        total = 0
        for latent_batch, concept_batch, action_batch in test_loader:
            
            cwnet_rankings, cwnet_logits = model(latent_batch)
            loss_ce = criterion_ce(cwnet_logits, concept_batch)
            loss_mse = criterion_mse(cwnet_rankings, action_batch)
            
            test_loss_ce  += loss_ce.item()
            test_loss_mse += loss_mse.item()

            # Calculate accuracy
            _, predicted = torch.max(cwnet_logits, 1)
            total += concept_batch.size(0)
            correct += (predicted == concept_batch).sum().item()

            # Collect predictions and labels for confusion matrix
            y_true.extend(concept_batch.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    test_losses_ce.append(test_loss_ce / len(test_loader))
    test_losses_mse.append(test_loss_mse / len(test_loader))
    accuracy = 100 * correct / total
    print(f"Test Loss CE: {test_loss_ce/len(test_loader):.4f}, Loss MSE: {test_loss_mse/len(test_loader):.4f}, Accuracy: {accuracy:.2f}%")

    # Confusion matrix
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred)
    cmd = ConfusionMatrixDisplay(cm, display_labels=np.arange(concept_dim))
    cmd.plot(cmap='Blues', xticks_rotation='vertical')
    plt.title('Confusion Matrix')
    plt.savefig('../results/confusion_matrix_concept_accuracy.pdf')
    plt.close()

    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses_ce, label='Train Loss (CE)')
    plt.plot(train_losses_mse, label='Train Loss (MSE)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss Curve')
    plt.savefig('../results/losses.pdf')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(accuracies, label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.savefig('../results/accuracy.pdf')
    plt.close()



    # ## Test in simulation
    def load_config():
        with open(CONFIG_FILE, "r") as f:
            config = toml.load(f)
        return config




    CONFIG_FILE = "config.toml"




    cfg = load_config()
    env = CarRacing(frame_skip=0, frame_stack=4,)
    net = RacingNet(env.observation_space.shape, env.action_space.shape)
    ppo = PPO(
        env,
        net,
        lr=cfg["lr"],
        gamma=cfg["gamma"],
        batch_size=cfg["batch_size"],
        gae_lambda=cfg["gae_lambda"],
        clip=cfg["clip"],
        value_coef=cfg["value_coef"],
        entropy_coef=cfg["entropy_coef"],
        epochs_per_step=cfg["epochs_per_step"],
        num_steps=cfg["num_steps"],
        horizon=cfg["horizon"],
        save_dir=cfg["save_dir"],
        save_interval=cfg["save_interval"],
    )
    ppo.load("weights/agent_weights.pt")


    # Mapping for concept predictions
    concept_mapping = {0: "Straight", 1: "Left", 2: "Right"}

    # Updated simulation loop with GIF saving for the first simulation
    reward_arr = []
    all_errors = list()
    data_rewards = list()
    data_errors = list()
    mse_loss = nn.MSELoss()
    num_simulation = 3

    frames = []  # List to store images for GIF


    # Directory path
    temp_frames_dir = "../results/temp_frames/"

    # Check if the directory exists
    if not os.path.exists(temp_frames_dir):
        os.makedirs(temp_frames_dir)
        print(f"Directory created: {temp_frames_dir}")
    else:
        print(f"Directory already exists: {temp_frames_dir}")



    for i in tqdm(range(num_simulation)):
        state = ppo._to_tensor(env.reset())
        count = 0
        rew = 0
        model.eval()

        for t in range(10000):
            # Get black box action
            value, alpha, beta, latent_x = ppo.net(state)
            value, alpha, beta = value.squeeze(0), alpha.squeeze(0), beta.squeeze(0)
            policy = Beta(alpha, beta)
            input_action = policy.mean.detach()
            bb_action = ppo.env.preprocess(input_action)

            action, concept_logits = model(latent_x)
            concept_pred = torch.argmax(concept_logits[0]).item()
            concept_word = concept_mapping.get(concept_pred, "Unknown")  # Map to word

            all_errors.append(mse_loss(torch.tensor(bb_action), action[0]).detach().item())

            # Save images for the first simulation
            if i == 0:
                plt.figure(figsize=(6, 6))
                plt.imshow(state[0][0], cmap="gray")
                plt.title(concept_word)
                plt.axis("off")
                plt.tight_layout()

                # Save the current frame as an image in memory
                frame_path = f"../results/temp_frames/frame_{t}.png"  # Save each frame to a file
                plt.savefig(frame_path, dpi=100, format="png")
                plt.close()
                frames.append(Image.open(frame_path))  # Open the saved file with Pillow

            state, reward, done, _, _ = ppo.env.step(action[0].detach().numpy(), real_action=True)
            state = ppo._to_tensor(state)
            rew += reward
            count += 1

            if done:
                break

        reward_arr.append(rew)

    data_errors.append(all_errors)
    data_errors = np.array(data_errors)
    data_rewards = np.array(reward_arr)

    print(" ")
    print("===== Data MAE:")
    print("Mean difference to original black-box:", data_errors.mean())
    print(" ")
    print("===== Data Reward:")
    print("Rewards:", data_rewards)
    print("Mean (should be > 200):", data_rewards.mean())

    if data_rewards.mean() > 200:
        print('Successful training!')
    else:
        print('Failed training...')

    np.save('../results/error.npy', data_errors)
    np.save('../results/rewards.npy', data_rewards)

    # Save GIF from frames
    if frames:
        gif_path = "../results/simulation.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            duration=100,  # Set duration between frames in ms
            loop=0
        )
        print(f"Simulation GIF saved at {gif_path}")
    else:
        print("No frames captured for GIF.")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


if __name__ =='__main__':
    main()

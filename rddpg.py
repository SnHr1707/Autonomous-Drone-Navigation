import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from per import PrioritizedReplayBuffer

class Actor(nn.Module):
    def __init__(self, action_size, action_high):
        super(Actor, self).__init__()
        self.action_high = action_high
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 16, 3, stride=2, padding=1), nn.ELU(),
            nn.Flatten()
        )
        self.gru = nn.GRU(16 * 5 * 8, 48, batch_first=True)
        self.state_fc = nn.Sequential(nn.Linear(6, 48), nn.Tanh())
        
        self.fc1 = nn.Sequential(nn.Linear(48 + 48, 64), nn.ELU())
        self.fc2 = nn.Sequential(nn.Linear(64, 32), nn.ELU())
        self.out = nn.Linear(32, action_size)

    def forward(self, image_seq, state):
        B, Seq, C, H, W = image_seq.shape
        x = image_seq.view(B * Seq, C, H, W)
        x = self.cnn(x)
        x = x.view(B, Seq, -1)
        _, gru_out = self.gru(x)
        gru_out = gru_out.squeeze(0) 
        
        state_out = self.state_fc(state) 
        combined = torch.cat([gru_out, state_out], dim=1)
        
        out = self.fc1(combined)
        out = self.fc2(out)
        return torch.tanh(self.out(out)) * self.action_high

class Critic(nn.Module):
    def __init__(self, action_size):
        super(Critic, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 16, 3, stride=2, padding=1), nn.ELU(),
            nn.Flatten()
        )
        self.gru = nn.GRU(16 * 5 * 8, 48, batch_first=True)
        self.state_fc = nn.Sequential(nn.Linear(6, 48), nn.Tanh())
        self.action_fc = nn.Sequential(nn.Linear(action_size, 48), nn.Tanh())
        
        self.fc1 = nn.Sequential(nn.Linear(48 + 48 + 48, 64), nn.ELU())
        self.fc2 = nn.Sequential(nn.Linear(64, 32), nn.ELU())
        self.out = nn.Linear(32, 1)

    def forward(self, image_seq, state, action):
        B, Seq, C, H, W = image_seq.shape
        x = image_seq.view(B * Seq, C, H, W)
        x = self.cnn(x)
        x = x.view(B, Seq, -1)
        _, gru_out = self.gru(x)
        gru_out = gru_out.squeeze(0)
        
        state_out = self.state_fc(state)
        action_out = self.action_fc(action)
        
        combined = torch.cat([gru_out, state_out, action_out], dim=1)
        out = self.fc1(combined)
        out = self.fc2(out)
        return self.out(out)

class RDDPGAgent:
    def __init__(self, action_size=3, max_action=5.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = Actor(action_size, max_action).to(self.device)
        self.actor_target = Actor(action_size, max_action).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        
        self.critic = Critic(action_size).to(self.device)
        self.critic_target = Critic(action_size).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=5e-4)
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.memory = PrioritizedReplayBuffer(50000)
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 64
        
    def get_action(self, image_seq, state, noise_scale=0.0):
        self.actor.eval()
        with torch.no_grad():
            img_tensor = torch.FloatTensor(image_seq).unsqueeze(0).to(self.device)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(img_tensor, state_tensor).cpu().data.numpy().flatten()
        self.actor.train()
        
        noise = np.random.normal(0, noise_scale, size=action.shape)
        return np.clip(action + noise, -self.actor.action_high, self.actor.action_high)
        
    def train(self, beta=0.4):
        if len(self.memory) < self.batch_size:
            return 0, 0
            
        batch, idxs, is_weights = self.memory.sample(self.batch_size, beta)
        
        img_seqs = torch.FloatTensor(np.array([t[0] for t in batch])).to(self.device)
        states = torch.FloatTensor(np.array([t[1] for t in batch])).to(self.device)
        actions = torch.FloatTensor(np.array([t[2] for t in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([t[3] for t in batch])).unsqueeze(1).to(self.device)
        next_img_seqs = torch.FloatTensor(np.array([t[4] for t in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([t[5] for t in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([t[6] for t in batch])).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(is_weights).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_actions = self.actor_target(next_img_seqs, next_states)
            target_Q = self.critic_target(next_img_seqs, next_states, next_actions)
            target_Q = rewards + (1 - dones) * self.gamma * target_Q

        current_Q = self.critic(img_seqs, states, actions)
        td_errors = target_Q - current_Q
        critic_loss = (weights * (td_errors ** 2)).mean()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0) # <--- ADD THIS
        self.critic_optimizer.step()

        for i, idx in enumerate(idxs):
            self.memory.update(idx, abs(td_errors[i].item()))

        actor_loss = -self.critic(img_seqs, states, self.actor(img_seqs, states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0) # <--- ADD THIS
        self.actor_optimizer.step()

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss.item(), critic_loss.item()
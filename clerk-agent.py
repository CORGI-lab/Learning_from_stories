import numpy as np

import gym
import textworld
import textworld.gym
from textworld import EnvInfos

from transformers import BertTokenizer, BertConfig, BertForSequenceClassification

import torch
import re
from typing import List, Mapping, Any, Optional
from collections import defaultdict

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import os
from glob import glob
import csv

import torch as t
#tw-extract -v entities /mnt/c/users/spenc/desktop/lfs/tw_games/cg.ulx
#tw-extract -v vocab /mnt/c/users/spenc/desktop/lfs/tw_games/cg.ulx
#tw-play /mnt/c/users/spenc/desktop/lfs/tw_games/cg.ulx
#.....   avg. steps:  73.7; avg. score:  2.7 / 3.
#Trained in 282.13 secs
# 
t.device("cuda:0" )
torch.manual_seed(22)
torch.device("cpu")
t.manual_seed(22)
t.cuda.set_device(0)
state_dict = t.load('./models/pytorch_model.bin')
#ggmodel.load_state_dict(state_dict)
tokenizer = BertTokenizer('./models/vocab.txt', do_lower_case=True)

GGCLASSES = ['negative','positive']
ggmodel = BertForSequenceClassification.from_pretrained('./models/', cache_dir=None, from_tf=False, state_dict=state_dict).to("cuda:0")

#config = BertConfig.from_json_file('./models/gg.json')
#ggmodel = BertForSequenceClassification('./models/gg.bin')

# input_ids = torch.tensor(tokenizer.encode("He always cleans his room", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
# labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
# outputs = ggmodel(input_ids, labels=labels)
# loss, logits = outputs[:2]
# print("BERT TEST OUT: {},{}".format(loss,logits))
# classification_index = max(range(len(logits[0])), key=logits[0].__getitem__)
# print(GGCLASSES[classification_index])

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#print(torch.cuda.is_available(), device)

#

def play(agent, path, max_step=100, nb_episodes=500, verbose=True):
    infos_to_request = agent.infos_to_request
    infos_to_request.max_score = True  # Needed to normalize the scores.
    
    gamefiles = [path]
    if os.path.isdir(path):
        gamefiles = glob(os.path.join(path, "*.ulx"))
        
    env_id = textworld.gym.register_games(gamefiles,
                                          request_infos=infos_to_request,
                                          max_episode_steps=max_step)

    env = gym.make(env_id)  # Create a Gym environment to play the text game.
    if verbose:
        if os.path.isdir(path):
            print(os.path.dirname(path), end="")
        else:
            print(os.path.basename(path), end="")
        
    # Collect some statistics: nb_steps, final reward.
    avg_moves, avg_scores, avg_norm_scores = [], [], []
    for no_episode in range(nb_episodes):
        obs, infos = env.reset()  # Start new episode.

        score = 0
        done = False
        nb_moves = 0
        while not done:
            command = agent.act(obs, score, done, nb_moves, infos)
            #print(command)
            obs, score, done, infos = env.step(command)
            nb_moves += 1
        
        agent.act(obs, score, done, nb_moves, infos)  # Let the agent know the game is done.
                
        if verbose:
            print(".", end="")
        avg_moves.append(nb_moves)
        avg_scores.append(score)
        avg_norm_scores.append(score / infos["max_score"])

    env.close()
    msg = "  \tavg. steps: {:5.1f}; avg. score: {:4.1f} / {}."
    if verbose:
        if os.path.isdir(path):
            print(msg.format(np.mean(avg_moves), np.mean(avg_norm_scores), 1))
        else:
            print(msg.format(np.mean(avg_moves), np.mean(avg_scores), infos["max_score"]))

class CommandScorer(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CommandScorer, self).__init__()
        device = torch.device("cpu")
        #torch.manual_seed(42)  # For reproducibility
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #print(torch.cuda.is_available())
        #print(device)
        self.embedding    = torch.nn.Embedding(input_size, hidden_size).to("cpu")
        self.encoder_gru  = torch.nn.GRU(hidden_size, hidden_size).to("cpu")
        self.cmd_encoder_gru  = torch.nn.GRU(hidden_size, hidden_size).to("cpu")
        self.state_gru    = torch.nn.GRU(hidden_size, hidden_size).to("cpu")
        self.hidden_size  = hidden_size
        self.state_hidden = torch.zeros(1, 1, hidden_size).to("cpu")
        self.critic       = torch.nn.Linear(hidden_size, 1).to("cpu")
        self.att_cmd      = torch.nn.Linear(hidden_size * 2, 1).to("cpu")

    def forward(self, obs, commands, **kwargs):
        input_length = obs.size(0)
        batch_size = obs.size(1)
        nb_cmds = commands.size(1)

        embedded = self.embedding(obs).to("cpu")
        encoder_output, encoder_hidden = self.encoder_gru(embedded)
        encoder_hidden.to("cpu")
        state_output, state_hidden = self.state_gru(encoder_hidden, self.state_hidden.to("cpu"))
        self.state_hidden = state_hidden
        value = self.critic(state_output).to("cpu")

        # Attention network over the commands.
        cmds_embedding = self.embedding.forward(commands).to("cpu")
        _, cmds_encoding_last_states = self.cmd_encoder_gru.forward(cmds_embedding) # 1 x cmds x hidden

        # Same observed state for all commands.
        cmd_selector_input = torch.stack([state_hidden] * nb_cmds, 2).to("cpu")  # 1 x batch x cmds x hidden

        # Same command choices for the whole batch.
        cmds_encoding_last_states = torch.stack([cmds_encoding_last_states] * batch_size, 1).to("cpu")  # 1 x batch x cmds x hidden

        # Concatenate the observed state and command encodings.
        cmd_selector_input = torch.cat([cmd_selector_input, cmds_encoding_last_states], dim=-1).to("cpu")

        # Compute one score per command.
        scores = torch.nn.functional.relu(self.att_cmd(cmd_selector_input)).squeeze(-1).to("cpu")  # 1 x Batch x cmds

        probs = torch.nn.functional.softmax(scores, dim=2).to("cpu")  # 1 x Batch x cmds
        index = probs[0].multinomial(num_samples=1).unsqueeze(0).to("cpu") # 1 x batch x indx
        return scores, index, value

    def reset_hidden(self, batch_size):
        self.state_hidden = torch.zeros(1, batch_size, self.hidden_size).to("cpu")


class NeuralAgent:
    """ Simple Neural Agent for playing TextWorld games. """
    MAX_VOCAB_SIZE = 150
    UPDATE_FREQUENCY = 10
    LOG_FREQUENCY = 1000
    GAMMA = 0.9
    
    def __init__(self) -> None:
        self._initialized = False
        self._epsiode_has_started = False
        self.id2word = ["<PAD>", "<UNK>"]
        self.word2id = {w: i for i, w in enumerate(self.id2word)}
        
        self.model = CommandScorer(input_size=self.MAX_VOCAB_SIZE, hidden_size=128)
        self.optimizer = optim.Adam(self.model.parameters(), 0.00003)
        
        self.mode = "test"
    
    def train(self):
        self.mode = "train"
        self.stats = {"max": defaultdict(list), "mean": defaultdict(list)}
        self.transitions = []
        self.model.reset_hidden(1)
        self.last_score = 0
        self.no_train_step = 0
    
    def test(self):
        self.mode = "test"
        self.model.reset_hidden(1)
        
    @property
    def infos_to_request(self) -> EnvInfos:
        return EnvInfos(description=True, inventory=False, admissible_commands=True,
                        won=True, lost=True)
    
    def _get_word_id(self, word):
        if word not in self.word2id:
            if len(self.word2id) >= self.MAX_VOCAB_SIZE:
                return self.word2id["<UNK>"]
            
            self.id2word.append(word)
            self.word2id[word] = len(self.word2id)
            
        return self.word2id[word]
            
    def _tokenize(self, text):
        # Simple tokenizer: strip out all non-alphabetic characters.
        text = re.sub("[^a-zA-Z0-9\- ]", " ", text)
        word_ids = list(map(self._get_word_id, text.split()))
        return word_ids

    def _process(self, texts):
        texts = list(map(self._tokenize, texts))
        max_len = max(len(l) for l in texts)
        padded = np.ones((len(texts), max_len)) * self.word2id["<PAD>"]

        for i, text in enumerate(texts):
            padded[i, :len(text)] = text

        padded_tensor = torch.from_numpy(padded).type(torch.long).to("cpu")
        padded_tensor = padded_tensor.permute(1, 0).to("cpu") # Batch x Seq => Seq x Batch
        return padded_tensor
      
    def _discount_rewards(self, last_values):
        returns, advantages = [], []
        R = last_values.data
        for t in reversed(range(len(self.transitions))):
            rewards, _, _, values = self.transitions[t]
            R = rewards + self.GAMMA * R
            adv = R - values
            returns.append(R)
            advantages.append(adv)
            
        return returns[::-1], advantages[::-1]

    def act(self, obs: str, score: int, done: bool, moves: int, infos: Mapping[str, Any]) -> Optional[str]:
        #device = torch.device("cpu")
        #torch.cuda.set_device(0)
        # Build agent's observation: feedback + look + inventory.
        #input_ = "{}\n{}\n{}".format(obs, infos["description"], infos["inventory"])
        input_ = "{}\n{}".format(obs, infos["description"])
        #print(infos["admissible_commands"])
        # Tokenize and pad the input and the commands to chose from.
        input_tensor = self._process([input_])
        commands_tensor = self._process(infos["admissible_commands"])
        
        # Get our next action and value prediction.
        outputs, indexes, values = self.model(input_tensor, commands_tensor)
        action = infos["admissible_commands"][indexes[0]]
        #print(action)
        ginput_ids = t.tensor(tokenizer.encode(infos["description"]+',so he '+action, add_special_tokens=True)).unsqueeze(0) # Batch size 1
        glabels = t.tensor([1]).unsqueeze(0).cuda()  # Batch size 1
        goutputs = ggmodel(ginput_ids, labels=glabels)
        gloss, glogits = goutputs[:2]
        #print("BERT TEST OUT: {},{}".format(loss,logits))
        classification_index = max(range(len(glogits[0])), key=glogits[0].__getitem__)
        #print(GGCLASSES[classification_index])
        BERT_reward = 0
        if GGCLASSES[classification_index] == 'negative':
            BERT_reward = glogits[0][0] * -100
        else:
            BERT_reward = glogits[0][1] * 100

        if self.mode == "test":
            if done:
                self.model.reset_hidden(1)
            return action
        
        self.no_train_step += 1
        
        if self.transitions:
            reward = (score * 100 - self.last_score) - moves + BERT_reward  # Reward is the gain/loss in score.
            #A2C-base, mixed binary, mixed diff, pos only, neg only
            #reward = (score * 100) - moves
            #print(reward)
            self.last_score = score
            if infos["won"]:
                print('won')
                reward += 1000
                reward -= moves
                reward += BERT_reward
            if infos["lost"]:
                reward -= 1000
                reward -= moves
                reward += BERT_reward
                
            self.transitions[-1][0] = reward  # Update reward information.
        
        self.stats["max"]["score"].append(score)
        if self.no_train_step % self.UPDATE_FREQUENCY == 0:
            # Update model
            returns, advantages = self._discount_rewards(values)
            #A2C-base, mixed binary, mixed diff, pos only, neg only
            with open('gg-confidence-mix-binary-a2c.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                lx = 0.0000001
                loss = Variable(torch.tensor(lx), requires_grad = False).to("cpu")
                #oss = 0
                for transition, ret, advantage in zip(self.transitions, returns, advantages):
                    reward, indexes_, outputs_, values_ = transition
                    
                    advantage        = advantage.detach()# Block gradients flow here.
                    probs            = F.softmax(outputs_, dim=2)
                    log_probs        = torch.log(probs)
                    log_action_probs = log_probs.gather(2, indexes_)
                    policy_loss      = (-log_action_probs * advantage).sum()
                    value_loss       = (.5 * (values_ - ret) ** 2.).sum()
                    entropy     = (-probs * log_probs).sum()
                    loss += policy_loss + 0.5 * value_loss.long() - 0.1 * entropy
                    
                    self.stats["mean"]["reward"].append(reward)
                    self.stats["mean"]["policy"].append(policy_loss.item())
                    self.stats["mean"]["value"].append(value_loss.item())
                    self.stats["mean"]["entropy"].append(entropy.item())
                    self.stats["mean"]["confidence"].append(torch.exp(log_action_probs).item())
                    #episode,reward,policy,value,entropy,confidence,score,vocabsize
                    if self.no_train_step % 1000 == 0:
                        writer.writerow([self.no_train_step,reward,policy_loss.item(),value_loss.item(),entropy.item(),torch.exp(log_action_probs).item(),score,len(self.id2word)])
                if self.no_train_step % self.LOG_FREQUENCY == 0:
                    msg = "{}. ".format(self.no_train_step)
                    msg += "  ".join("{}: {:.3f}".format(k, np.mean(v)) for k, v in self.stats["mean"].items())
                    msg += "  " + "  ".join("{}: {}".format(k, np.max(v)) for k, v in self.stats["max"].items())
                    msg += "  vocab: {}".format(len(self.id2word))
                    print(msg)
                    self.stats = {"max": defaultdict(list), "mean": defaultdict(list)}
                
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 40)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
                self.transitions = []
                self.model.reset_hidden(1)
        else:
            # Keep information about transitions for Truncated Backpropagation Through Time.
            self.transitions.append([None, indexes, outputs, values])  # Reward will be set on the next call
        
        if done:
            self.last_score = 0  # Will be starting a new episode. Reset the last score.
        
        return action

agent = NeuralAgent()

from time import time
agent = NeuralAgent()

print("Training")
agent.train()  # Tell the agent it should update its parameters.
starttime = time()
play(agent, "tw_games/cg.ulx", max_step=50, nb_episodes=50000, verbose=True)  # Dense rewards game.
print("Trained in {:.2f} secs".format(time() - starttime))
agent.test()
#play(agent, "tw_games/cg.ulx")  # Dense rewards game.
#play(agent, "tw_games/cg.ulx")
# Register a text-based game as a new Gym's environment.
# env_id = textworld.gym.register_game("tw_games/clerk_game.ulx",
#                                      max_episode_steps=50)

# env = gym.make(env_id)  # Start the environment.

# obs, infos = env.reset()  # Start new episode.
# env.render()

# score, moves, done = 0, 0, False
# while not done:
#     command = input("> ")
#     obs, score, done, infos = env.step(command)
#     env.render()
#     moves += 1

# env.close()
# print("moves: {}; score: {}".format(moves, score))
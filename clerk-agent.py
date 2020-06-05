import numpy as np

import gym
import textworld
import textworld.gym
from textworld import EnvInfos

from transformers import BertTokenizer, BertConfig, BertForSequenceClassification

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

#tw-extract -v entities /mnt/c/users/spenc/desktop/lfs/tw_games/cg.ulx
#tw-extract -v vocab /mnt/c/users/spenc/desktop/lfs/tw_games/cg.ulx
#tw-play /mnt/c/users/spenc/desktop/lfs/tw_games/cg.ulx
#.....   avg. steps:  73.7; avg. score:  2.7 / 3.
#Trained in 282.13 secs
# 
torch.manual_seed(22)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#state_dict = t.load('./models/pytorch_model.bin')
#ggmodel.load_state_dict(state_dict)
tokenizer = BertTokenizer('./models/vocab.txt', do_lower_case=True)

GGCLASSES = ['negative','positive']
ggmodel = BertForSequenceClassification.from_pretrained('./models/', cache_dir=None, from_tf=False, state_dict=None).to("cuda:0")

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

modelmode = "a2c"
#mode = "gg-mix"
#mode = "gg-neg"
#mode = "gg-pos"

current_vloss = 0
current_ploss = 0
current_entropy = 0
current_confidence = 0
total_gg_pos = 0
total_gg_neg = 0
current_reward = 0
bert_agreement = []
bert_loss = 0
bertrews = []
baserews = []

lobbyDesc = "There is a form you need to stamp in your mailbox. There is a customer on the floor. There are some donuts on the table. There is a clerk window and an employee door to your south."
counterDesc = "There are doors to the office and storage room. There is a form you need to stamp on a table. Your coworker looks stressed and there is a customer who looks angry."
officeDesc = "This is a store room. There are many objects on the floor. There is a stack of papers that need to be stamped on the shelf. There is a door to your east."
storageDesc = "This is an office. There is an important form on the desk. There is food on the desk. There is a door to the east."
#

baseSentences = {'aid':["The clerk helps the person."], 'ask':["He wants to be helpful so he asks what is wrong"], 'drop':['Thinking hes done, he drops the object'], 'take':["Seeing the thing on the floor, he takes it"], 'go':["He heads in the direction of the goal"],'look':["He looks at the strange situation unfolding in the room."], 'stamp':["He stamps the ticket because that is his job"], 'wait':["The man taps his foot waiting for something"], 'eat':["He eats the food quickly but carefully"]}
lobbyCrowdsourced = {''}

def play(agent, path, max_step=100, nb_episodes=500, verbose=True, agentType="a2c", runNumber=0, pronoun="He"):
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
    with open('action-phrases-vs-not.csv', 'a', newline='\n') as file:
        writer2 = csv.writer(file)
        writer2.writerow(['runNumber','agentType','no_episode','nb_moves','win','pronoun','current_reward','current_ploss','current_vloss','current_confidence','current_entropy','total_gg_pos','total_gg_neg','avg_bert_reward','avg_base_reward'])  
        for no_episode in range(nb_episodes):
            obs, infos = env.reset()  # Start new episode.
            bert_agreement = []
            score = 0
            done = False
            nb_moves = 0
            while not done:
                command = agent.act(agentType, obs, score, done, nb_moves, pronoun, infos)
                #print(command)
                obs, score, done, infos = env.step(command)
                nb_moves += 1
                #print(nb_moves)
            win = 0
            #if score == 15:
            #    win = 1
            ba = 0
            if len(bertrews) != 0:
                if sum(bertrews) != 0:
                    ba = sum(bertrews)/len(bertrews)
                print("ep agreement: {}".format(ba))
            oa = 0
            if len(baserews) != 0:
                if sum(baserews) != 0:
                    oa = sum(baserews)/len(baserews)
                print("avg rew: {}".format(oa))
            agent.act(agentType,obs, score, done, nb_moves, pronoun, infos)  # Let the agent know the game is done.
            writer2.writerow([runNumber,agentType,no_episode,nb_moves,win,pronoun,current_reward,current_ploss,current_vloss,current_confidence,current_entropy,total_gg_pos,total_gg_neg,ba,oa])        
            if verbose:
                print(".", end="")
            avg_moves.append(nb_moves)
            avg_scores.append(score)
            avg_norm_scores.append(score / infos["max_score"])

    env.close()
    msg = agentType+"  \tavg. steps: {:5.1f}; avg. score: {:4.1f} / {}."
    print(msg)
    if verbose:
        if os.path.isdir(path):
            print(msg.format(np.mean(avg_moves), np.mean(avg_norm_scores), 1))
        else:
            print(msg.format(np.mean(avg_moves), np.mean(avg_scores), infos["max_score"]))

class CommandScorer(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CommandScorer, self).__init__()
        #device = torch.device("cpu")
        #torch.manual_seed(42)  # For reproducibility
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #print(torch.cuda.is_available())
        #print(device)
        self.embedding    = torch.nn.Embedding(input_size, hidden_size).to(device)
        self.encoder_gru  = torch.nn.GRU(hidden_size, hidden_size).to(device)
        self.cmd_encoder_gru  = torch.nn.GRU(hidden_size, hidden_size).to(device)
        self.state_gru    = torch.nn.GRU(hidden_size, hidden_size).to(device)
        self.hidden_size  = hidden_size
        self.state_hidden = torch.zeros(1, 1, hidden_size).to(device)
        self.critic       = torch.nn.Linear(hidden_size, 1).to(device)
        self.att_cmd      = torch.nn.Linear(hidden_size * 2, 1).to(device)

    def forward(self, obs, commands, **kwargs):
        input_length = obs.size(0)
        batch_size = obs.size(1)
        nb_cmds = commands.size(1)

        embedded = self.embedding(obs).to(device)
        encoder_output, encoder_hidden = self.encoder_gru(embedded)
        encoder_hidden.to(device)
        state_output, state_hidden = self.state_gru(encoder_hidden, self.state_hidden.to(device))
        self.state_hidden = state_hidden
        value = self.critic(state_output).to(device)

        # Attention network over the commands.
        cmds_embedding = self.embedding.forward(commands).to(device)
        _, cmds_encoding_last_states = self.cmd_encoder_gru.forward(cmds_embedding) # 1 x cmds x hidden

        # Same observed state for all commands.
        cmd_selector_input = torch.stack([state_hidden] * nb_cmds, 2).to(device)  # 1 x batch x cmds x hidden


        cmds_encoding_last_states = torch.stack([cmds_encoding_last_states] * batch_size, 1).to(device)  # 1 x batch x cmds x hidden

        # Concatenate the observed state and command encodings.
        cmd_selector_input = torch.cat([cmd_selector_input, cmds_encoding_last_states], dim=-1).to(device)

        # Compute one score per command.
        scores = torch.nn.functional.relu(self.att_cmd(cmd_selector_input)).squeeze(-1).to(device)  # 1 x Batch x cmds

        probs = torch.nn.functional.softmax(scores, dim=2).to(device)  # 1 x Batch x cmds
        index = probs[0].multinomial(num_samples=1).unsqueeze(0).to(device) # 1 x batch x indx
        return scores, index, value

    def reset_hidden(self, batch_size):
        self.state_hidden = torch.zeros(1, batch_size, self.hidden_size).to(device)


class NeuralAgent:
    """ Simple Neural Agent for playing TextWorld games. """
    MAX_VOCAB_SIZE = 500
    UPDATE_FREQUENCY = 10
    LOG_FREQUENCY = 1000
    GAMMA = 0.9
    
    def __init__(self) -> None:
        self._initialized = False
        self._epsiode_has_started = False
        self.id2word = ["<PAD>", "<UNK>"]
        self.word2id = {w: i for i, w in enumerate(self.id2word)}
        
        self.model = CommandScorer(input_size=self.MAX_VOCAB_SIZE, hidden_size=512)
        self.optimizer = optim.Adam(self.model.parameters(), 0.0003)
        
        self.mode = "test"
    
    def train(self):
        self.mode = "train"
        self.stats = {"max": defaultdict(list), "mean": defaultdict(list)}
        self.transitions = []
        self.ggtransitions = []
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

        padded_tensor = torch.from_numpy(padded).type(torch.long).to(device)
        padded_tensor = padded_tensor.permute(1, 0).to(device) # Batch x Seq => Seq x Batch
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

    def act(self, agentType: str, obs: str, score: int, done: bool, moves: int, pronoun: str, infos: Mapping[str, Any]) -> Optional[str]:
        global current_ploss
        global current_vloss
        global current_entropy
        global current_confidence
        global total_gg_pos
        global total_gg_neg
        global current_reward
        global bert_agreement
        global bert_loss
        global bertrews
        global baserews
        global descpairs

        global officeDesc
        global lobbyDesc
        global counterDesc
        global storageDesc
        #device = torch.device("cpu")
        #torch.cuda.set_device(0)
        # Build agent's observation: feedback + look + inventory.
        desc = infos["description"]

        if "office" in infos["description"]:
            desc = officeDesc
        if "storage" in infos["description"]:
            desc = storageDesc
        if "lobby" in infos["description"]:
            desc = lobbyDesc
        if "counter" in infos["description"]:
            desc = counterDesc
        input_ = "{}\n{}".format(obs, desc)
        #input_ = "{}\n{}".format(obs, infos["description"])
        #print(infos["admissible_commands"])
        # Tokenize and pad the input and the commands to chose from.
        input_tensor = self._process([input_])
        commands_tensor = self._process(infos["admissible_commands"])

        #SJF OUTPUTTING STATE/ACTION PAIRS: DESCRIPTION AS KEY, ADMISSABLE AS VALUE IN DICT, RUN AGENT FOR A FEW MINUTES
        #SJF: for each, format a string and add to a file

        # Get our next action and value prediction.
        outputs, indexes, values = self.model(input_tensor, commands_tensor)
        action = infos["admissible_commands"][indexes[0]]
        # print(infos["admissible_commands"][0])
        # print(infos["admissible_commands"][3])
        # print(infos["admissible_commands"][indexes[0]])
        #print("\nold action: {}".format(action))
        old_action = action
        #print("------")
        #print(outputs[0][0]) #outputs = scores
        #print("-------")
        #sum_of_values = []
        max_value = -9999
        max_value_index = -1


        BERT_pos_reward = 0
        BERT_neg_reward = 0

        if self.mode == "test":
            if done:
                self.model.reset_hidden(1)
            return action
        
        self.no_train_step += 1
        
        if self.transitions:
            if "gg-ps" in agentType:
                for idx, val in enumerate(outputs[0][0]):
                    with torch.no_grad():
                        if pronoun == "crowdsourced":
                            matched = False
                            d = {'aid':["The clerk helps the person."], 'ask':["He wants to be helpful so he asks what is wrong"], 'drop':['Thinking hes done, he drops the object'], 'take':["Seeing the thing on the floor, he takes it"], 'go':["He heads in the direction of the goal"],'look':["He looks at the strange situation unfolding in the room."], 'stamp':["He stamps the ticket because that is his job"], 'wait':["The man taps his foot waiting for something"], 'eat':["He eats the food quickly but carefully"]}
                            for key in d:
                                if key in desc:
                                    matched = True
                                    sentence = d[key]
                                    ginput_ids = torch.tensor(tokenizer.encode(sentence, add_special_tokens=True)).unsqueeze(0).cuda()
                            if matched == False:
                                ginput_ids = torch.tensor(tokenizer.encode(desc+'. He '+infos["admissible_commands"][idx], add_special_tokens=True)).unsqueeze(0).cuda()
                        else:
                            ginput_ids = torch.tensor(tokenizer.encode(desc+'. He '+infos["admissible_commands"][idx], add_special_tokens=True)).unsqueeze(0).cuda() # Batch size 1
                        glabels = torch.tensor([1]).unsqueeze(0).cuda()  # Batch size 1
                        goutputs = ggmodel(ginput_ids, labels=glabels)
                        gloss, glogits = goutputs[:2]
                        #print("BERT TEST OUT: {},{}".format(loss,logits))
                        classification_index = max(range(len(glogits[0])), key=glogits[0].__getitem__)
                        #print(GGCLASSES[classification_index])
                        BERT_neg_reward = glogits[0][0].item()
                        #neg_values_per_command.append(BERT_neg_reward)
                        BERT_pos_reward = glogits[0][1].item()

                        mod = 1.0
                        if agentType == "gg-ps0.5":
                            mod = 0.5
                        if agentType == "gg-ps0.1":
                            mod = 0.1
                        #pos_values_per_command.append(BERT_pos_reward)
                        newVal = val.item() * (BERT_pos_reward - BERT_neg_reward) / mod
                        #print(newVal)
                        if newVal > max_value:
                            max_value = newVal
                            max_value_index = idx
                action = infos["admissible_commands"][max_value_index]

            BERT_reward = 0

            if agentType != "a2c" and agentType != "gg-loss" and ("gg-ps" not in agentType):
                with torch.no_grad(): 
                    if pronoun == "crowdsourced":
                        matched = False
                        d = {'aid':["The clerk helps the person."], 'ask':["He wants to be helpful so he asks what is wrong"], 'drop':['Thinking hes done, he drops the object'], 'take':["Seeing the thing on the floor, he takes it"], 'go':["He heads in the direction of the goal"],'look':["He looks at the strange situation unfolding in the room."], 'stamp':["He stamps the ticket because that is his job"], 'wait':["The man taps his foot waiting for something"], 'eat':["He eats the food quickly but carefully"]}
                        for key in d:
                            if key in desc:
                                matched = True
                                sentence = d[key]
                                ginput_ids = torch.tensor(tokenizer.encode(sentence, add_special_tokens=True)).unsqueeze(0).cuda()
                        if matched == False:
                            ginput_ids = torch.tensor(tokenizer.encode(desc+'. He '+infos["admissible_commands"][idx], add_special_tokens=True)).unsqueeze(0).cuda()
                    else:
                        ginput_ids = torch.tensor(tokenizer.encode(desc'. '+'He'+' '+action, add_special_tokens=True)).unsqueeze(0).cuda() # Batch size 1
                        glabels = torch.tensor([1]).unsqueeze(0).cuda()  # Batch size 1
                        goutputs = ggmodel(ginput_ids, labels=glabels)
                        gloss, glogits = goutputs[:2]
                        #print("BERT TEST OUT: {},{}".format(loss,logits))
                        classification_index = max(range(len(glogits[0])), key=glogits[0].__getitem__)
                        #print(GGCLASSES[classification_index])
                        
                        if GGCLASSES[classification_index] == 'negative':
                            BERT_neg_reward = glogits[0][0].item() * -1
                            total_gg_neg = total_gg_neg + glogits[0][0].item() * -1
                            if agentType == "gg-mix" or agentType == "gg-mix-multi" or agentType == "gg-neg":
                                BERT_reward = BERT_neg_reward * -1
                        else:
                            BERT_pos_reward = glogits[0][1].item()
                            total_gg_pos = total_gg_pos + glogits[0][0].item() 
                            if agentType == "gg-mix" or agentType == "gg-mix-multi" or agentType == "gg-pos":
                                BERT_reward = BERT_pos_reward

            #reward = ((score) * 100) - (moves) + int(BERT_reward)
            reward = score + BERT_reward
            if agentType == "gg-mix-multi":
                reward = score * BERT_reward
            #reward = ((self.last_score - score) * 100) - moves + int(BERT_reward)  # Reward is the gain/loss in score.
            current_reward = reward
            baserews.append(reward)
            #A2C-base, mixed binary, mixed diff, pos only, neg only
            #reward = (score * 100) - moves
            #print(reward)
            self.last_score = score
            if infos["won"]:
                print('won')
                #reward += 1000
            if infos["lost"]:
                print('lost')
                #reward -= 1000
                
            self.transitions[-1][0] = reward  # Update reward information.
            self.ggtransitions.append(BERT_pos_reward + BERT_neg_reward)
            bertrews.append(BERT_pos_reward + BERT_neg_reward)
        self.stats["max"]["score"].append(score)
        if self.no_train_step % self.UPDATE_FREQUENCY == 0:
            # Update model
            returns, advantages = self._discount_rewards(values)
            #A2C-base, mixed binary, mixed diff, pos only, neg only
            lx = 0.0000001
            loss = 0
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
                bert_loss = 0
                if agentType == "gg-loss":
                    bert_loss = (sum(self.ggtransitions)+0.000001)
                loss += policy_loss + (0.5 * value_loss.long()) - (0.1 * entropy) - (bert_loss)
                #try altering the gradient based on the bert_loss... change the rate at which weights change               
                #print(bert_loss)

                current_ploss = policy_loss.item()
                current_vloss = value_loss.item()
                current_entropy = entropy.item()
                current_confidence = torch.exp(log_action_probs).item()
                self.stats["mean"]["reward"].append(reward)
                self.stats["mean"]["policy"].append(policy_loss.item())
                self.stats["mean"]["value"].append(value_loss.item())
                self.stats["mean"]["entropy"].append(entropy.item())
                self.stats["mean"]["confidence"].append(torch.exp(log_action_probs).item())
                #episode,reward,policy,value,entropy,confidence,score,vocabsize
                #if self.no_train_step % 1000 == 0:
                #    writer.writerow([self.no_train_step,reward,policy_loss.item(),value_loss.item(),entropy.item(),torch.exp(log_action_probs).item(),score,len(self.id2word),moves])
            if self.no_train_step % self.LOG_FREQUENCY == 0:
                msg = "{}. ".format(self.no_train_step)
                msg += "  ".join("{}: {:.3f}".format(k, np.mean(v)) for k, v in self.stats["mean"].items())
                msg += "  " + "  ".join("{}: {}".format(k, np.max(v)) for k, v in self.stats["max"].items())
                msg += "  vocab: {}".format(len(self.id2word))
                #print(msg)
                self.stats = {"max": defaultdict(list), "mean": defaultdict(list)}
            
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 40)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
            self.transitions = []
            self.ggtransitions = []
            self.model.reset_hidden(1)
        else:
            # Keep information about transitions for Truncated Backpropagation Through Time.
            self.transitions.append([None, indexes, outputs, values])  # Reward will be set on the next call
        
        if done:
            bertrews = []
            baserews = []
            self.last_score = 0  # Will be starting a new episode. Reset the last score.
        
        return action

#agent = NeuralAgent()
modzlist = ["a2c","gg-mix","gg-ps0.5"]
#modzlist = ["a2c","gg-mix","gg-mix-multi","gg-pos","gg-neg","gg-loss","gg-ps1.0","gg-ps0.5","gg-ps0.1"] #policy shaping across all values, gg output as a scalar positive or negative, base a2c
pronouns = ["He", "She", "They"]

p = "He"

for a in range(5):
    for modz in modzlist:
        from time import time
        agent = NeuralAgent()

        print("Training {} with pronoun {} run {}".format(modz,p,a))
        agent.train()  # Tell the agent it should update its parameters.
        starttime = time()
        play(agent, "tw_games/cg.ulx", max_step=50, nb_episodes=1000, verbose=True, agentType=modz, runNumber=a, pronoun=p)  # Dense rewards game.
        print("Trained in {:.2f} secs".format(time() - starttime))
        agent.test()
    for modz in modzlist:
        from time import time
        agent = NeuralAgent()

        print("Training {} with pronoun {} run {}".format(modz,p,a))
        agent.train()  # Tell the agent it should update its parameters.
        starttime = time()
        play(agent, "tw_games/cg.ulx", max_step=50, nb_episodes=1000, verbose=True, agentType=modz, runNumber=a+5, pronoun='crowdsourced')  # Dense rewards game.
        print("Trained in {:.2f} secs".format(time() - starttime))
        agent.test()

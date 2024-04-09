import os
import json
import numpy
import re
import torch
import torch_ac
import gym

import utils


def get_obss_preprocessor(obs_space,evaluation=False):
    # Check if obs_space is an image space
    if isinstance(obs_space, gym.spaces.Box):
        obs_space = {"image": obs_space.shape}

        def preprocess_obss(obss, device=None):
            return torch_ac.DictList({
                "image": preprocess_images(obss, device=device)
            })

    # Check if it is a MiniGrid observation space
    elif isinstance(obs_space, gym.spaces.Dict) and list(obs_space.spaces.keys()) == ["image"]:
        obs_space = {"image": obs_space.spaces["image"].shape, "text": 100}

        vocab = Vocabulary(obs_space["text"])
        
        # define preprocess function
        def preprocess_obss(obss, device=None):
            # print('obss:',obss.keys())
            # print('obss state:', obss["image"])
            # print('obss mission:', obss["mission"])
            # print(len(obss))
            # for obs in obss:
            #     print('obs:',obs)

            if evaluation:
                _dict = torch_ac.DictList({
                    "image": preprocess_images([obss["image"]], device=device),
                    "text": preprocess_texts([obss["mission"]], vocab, device=device)
                })
            else:
                _dict = torch_ac.DictList({
                    "image": preprocess_images([obs["image"] for obs in obss], device=device),
                    "text": preprocess_texts([obs["mission"] for obs in obss], vocab, device=device)
                })
            return _dict
        
        preprocess_obss.vocab = vocab

    else:
        raise ValueError("Unknown observation space: " + str(obs_space))

    return obs_space, preprocess_obss


def preprocess_images(images, device=None):
    # Bug of Pytorch: very slow if not first converted to numpy array
    images = numpy.array(images)
    return torch.tensor(images, device=device, dtype=torch.float)


def preprocess_texts(texts, vocab, device=None):
    var_indexed_texts = []
    max_text_len = 0

    for text in texts:
        tokens = re.findall("([a-z]+)", text.lower())
        var_indexed_text = numpy.array([vocab[token] for token in tokens])
        var_indexed_texts.append(var_indexed_text)
        max_text_len = max(len(var_indexed_text), max_text_len)

    indexed_texts = numpy.zeros((len(texts), max_text_len))

    for i, indexed_text in enumerate(var_indexed_texts):
        indexed_texts[i, :len(indexed_text)] = indexed_text

    return torch.tensor(indexed_texts, device=device, dtype=torch.long)


class Vocabulary:
    """A mapping from tokens to ids with a capacity of `max_size` words.
    It can be saved in a `vocab.json` file."""

    def __init__(self, max_size):
        self.max_size = max_size
        self.vocab = {}

    def load_vocab(self, vocab):
        self.vocab = vocab

    def __getitem__(self, token):
        if not token in self.vocab.keys():
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import evaluate, datasets
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn, optim
from transformers import (AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM,
                          LlamaConfig, LogitsProcessor, LogitsProcessorList)

from dataset import get_dataset, get_ngram_model_path, Dataset, InputExample, ModelOutput, Completion

from parse import parse_args

class NgramModelNeural(nn.Module):
    def __init__(self, vocab_size, context_size, n_dim):
        super(NgramModel, self).__init__()
        self.vocab_size = vocab_size
        self.context_size = context_size  
        self.embeddings = nn.Embedding(vocab_size, n_dim)
        # LSTM layer where context_size is considered as the number of layers
        self.lstm = nn.LSTM(input_size=n_dim, hidden_size=n_dim, num_layers=3, batch_first=True)
        # Linear layer to map the outputs to vocabulary size
        self.linear = nn.Linear(n_dim, vocab_size)

    def forward(self, input_ids):
        # Embedding the input
        embeds = self.embeddings(input_ids)
        # LSTM output
        lstm_out, _ = self.lstm(embeds)
        # We take the last LSTM outputs for classification purposes
        final_output = lstm_out[:, -1, :]
        # Pass the output of the last time step to the linear layer
        logits = self.linear(final_output)
        log_prob = F.log_softmax(logits, dim=-1)
        return log_prob

    def get_logits(self, input_ids):
        # Embedding the input
        embeds = self.embeddings(input_ids)
        # LSTM output
        lstm_out, _ = self.lstm(embeds)
        # We take the last LSTM outputs
        final_output = lstm_out[:, -1, :]
        # Pass the output of the last time step to the linear layer
        logits = self.linear(final_output)
        return logits

class NgramModel(nn.Module):
    def __init__(self, vocab_size, context_size, n_dim):
        super(NgramModel, self).__init__()
        self.context_size = context_size
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, n_dim)
        self.linear1 = nn.Linear(context_size*n_dim, n_dim)
        self.linear2 = nn.Linear(n_dim, vocab_size)

    def forward(self, input_ids):
        assert input_ids.shape[1] == self.context_size
        embeds = self.embeddings(input_ids)
        out = self.linear1(embeds.view(-1, self.context_size*embeds.shape[-1]))
        out = F.relu(out)
        logits = self.linear2(out)
        log_prob = F.log_softmax(logits, dim=-1)
        return log_prob

    def get_logits(self, input_ids):
        # print("input_ids.shape", input_ids.shape)
        if input_ids.shape[1] < self.context_size:
            return torch.zeros(input_ids.shape[0], self.vocab_size).to(self.linear1.weight.device)
        self.eval()
        embeds = self.embeddings(input_ids)
        out = self.linear1(embeds.view(-1, self.context_size*embeds.shape[-1]))
        out = F.relu(out)
        logits = self.linear2(out)
        return logits

from collections import defaultdict
import random

import inspect


class NgramModel:
    def __init__(self, n, tokenizer, dataset_name):
        print("N of ngram model is [{}]".format(n))
        self.context_size = n - 1
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.n = n
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name


    def remove_special_tokens(self, input_ids):
        tokenizer = self.tokenizer
        tab_token_id = tokenizer.convert_tokens_to_ids('\t')
        newline_token_id = tokenizer.convert_tokens_to_ids('\n')
        carriage_return_token_id = tokenizer.convert_tokens_to_ids('\r')
        special_tokens = {tab_token_id, newline_token_id, carriage_return_token_id}
        new_input_ids = [token_id for token_id in input_ids if token_id not in special_tokens]
        if isinstance(input_ids, torch.Tensor):
            return torch.tensor(new_input_ids)
        return new_input_ids

    def train(self, tokenized_corpus):
        tokenizer = self.tokenizer
        # tokenized_corpus should be a list of lists of token IDs
        for input_ids in tqdm(tokenized_corpus, desc="Training ngram model"):
            # print(input_ids.shape)
            input_ids = input_ids.flatten().tolist()
            input_ids = self.remove_special_tokens(input_ids)
            # Prepare the input_ids with start and end tokens
            # Here we assume <s> and </s> are represented by some specific IDs assigned
            sentence = [tokenizer.pad_token_id] * (self.n - 1) + input_ids + [
                tokenizer.eos_token_id]  # Assume 0 is <s> and 1 is </s>
            for i in range(len(sentence) - self.n + 1):
                context = tuple(sentence[i:i + self.n - 1])
                token = sentence[i + self.n - 1]
                self.ngram_counts[tuple(context + (token,))] += 1
                self.context_counts[context] += 1

    def get_prob(self, context, tokens_list):
        # context and token should be tuples of token IDs
        if isinstance(context, torch.Tensor):
            context = context.tolist()
        if isinstance(tokens_list, torch.Tensor):
            tokens_list = tokens_list.tolist()
        if len(context) < self.n - 1:
            return [0] * len(tokens_list)
        elif len(context) > self.n - 1:
            context = context[-(self.n - 1):]
        context = tuple(context)
        if self.context_counts[context] == 0:
            return [0] * len(tokens_list)
        return [self.ngram_counts[context + (token,)] / self.context_counts[context] for token in tokens_list]

    @property
    def vocab(self):
        return set(token for context_plus_token in self.ngram_counts for token in context_plus_token)


def train_ngram_model(tokenizer, device, dataset, context_size, ngram_epoch, batch_size=512):
    ngram_model = NgramModel(context_size + 1, tokenizer, dataset.name)
    encoded_ids = []
    for example in tqdm(dataset, desc="Tokenizing dataset"):
        data = example["data"].content
        test_ids = tokenizer(data, return_tensors='pt').input_ids
        encoded_ids.append(test_ids)
    print("Training ngram model")
    ngram_model.train(encoded_ids)
    print("Training ngram model done")

    return ngram_model


def train_ngram_model_neural(tokenizer, device, dataset, context_size, ngram_epoch, batch_size=512):
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    ngram_model = NgramModel(tokenizer.vocab_size, context_size, 100).to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(ngram_model.parameters(), lr=2e-3)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    for epoch in tqdm(range(ngram_epoch)):
        running_loss = 0
        batch_ngrams = []
        batch_labels = []

        for example in dataset:
            data = example["data"].content
            test_ids = tokenizer.encode(data, return_tensors='pt')[0].data
            batch_ngrams.extend([test_ids[i:i + context_size] for i in range(len(test_ids) - context_size)])
            batch_labels.extend([test_ids[i + context_size] for i in range(len(test_ids) - context_size)])

        batch_ngrams = torch.stack(batch_ngrams, dim=0).to(device)
        batch_labels = torch.tensor(batch_labels, device=device)

        num_batches = len(batch_ngrams) // batch_size
        for batch in range(num_batches):
            start = batch * batch_size
            end = start + batch_size
            ngram_batch = batch_ngrams[start:end]
            label_batch = batch_labels[start:end]

            out = ngram_model(ngram_batch)
            loss = criterion(out, label_batch)
            running_loss += loss.item() * len(ngram_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step(running_loss)
        print(f'epoch: {(epoch + 1)}, Loss: {(running_loss / len(batch_ngrams)):.6f}')

    return ngram_model


class NGramLogitsProcessor(LogitsProcessor):
    def __init__(
            self,
            ngram_model: NgramModel,
            is_vllm: bool = False,
            tokenizer=None,
    ):
        self.ngram_model = ngram_model
        self.alpha = 0.1  # coefficient
        # debug
        self.tokenizer = tokenizer
        self.ngram_hits_count = 0
        self.is_vllm = is_vllm
        self.prompt_token_ids = 'None'
        self.reset_ngram_hits()
        # self.ngram_hits_tokens = []

    def __repr__(self):
        return f"NGramLogitsProcessor()"


    def reset_ngram_hits(self):
        self.ngram_hits_count = 0
        self.ngram_hits_tokens = []

    @torch.no_grad()
    def log_ngram_hits(self, ngram_values: torch.Tensor):
        hits = ngram_values > 1e-6
        if torch.any(hits):
            self.ngram_hits_count += 1
            return True
        return False

    def process_last_token_logits(self, input_ids, logits):
        for i in range(logits.shape[0]):
            processed_logits = self.process_last_token_logits_no_batch(input_ids[i].unsqueeze(0),
                                                                       logits[i].unsqueeze(0))
            logits[i] = processed_logits
        return logits

    def process_last_token_logits_no_batch(self, input_ids, processed_logits):
        possible_tokens = processed_logits.topk(k=10, dim=-1).indices
        input_ids = input_ids.flatten()
        new_input_ids = self.ngram_model.remove_special_tokens(input_ids)

        ngram_logits_for_possible_tokens = self.ngram_model.get_prob(new_input_ids, possible_tokens.flatten())
        ngram_logits = torch.tensor(ngram_logits_for_possible_tokens).to(processed_logits.device).to(
            processed_logits.dtype)

        self.log_ngram_hits(ngram_logits)
        processed_logits = processed_logits.view(-1)
        possible_tokens = possible_tokens.view(-1)

        alpha = self.alpha
        processed_logits[possible_tokens] = processed_logits[possible_tokens] * (1 - ngram_logits)

        processed_logits = processed_logits.view(1, -1)
        return processed_logits

    def set_prompt(self, prompts=None):
        if prompts is None:
            self.prompt_token_ids = None
        elif isinstance(prompts, list):
            self.prompt_token_ids = self.tokenizer.batch_encode_plus(prompts)
        else:
            print("Setting prompt to [{}]".format(prompts))
            self.prompt_token_ids = self.tokenizer.encode(prompts)
            print("Prompt token IDs:", self.prompt_token_ids)

    def __call__(self, input_ids, logits):
        if self.is_vllm:
            assert isinstance(input_ids, list)

            frame = inspect.currentframe()
            caller_frame = frame.f_back
            caller_locals = caller_frame.f_locals
            seq_group = caller_locals.get('seq_group')
            seq_id = caller_locals.get('seq_id')
            input_ids = [seq_group.seq_data[seq_id].prompt_token_ids + input_ids]
            logits = logits.unsqueeze(0)
            input_ids = torch.tensor(input_ids)
        sequence_length = len(input_ids[0])

        processed_logits = logits.clone()
        if logits.dim() == 2:
            processed_logits = self.process_last_token_logits(input_ids, processed_logits)
        elif logits.dim() == 3:
            for end_idx in range(self.ngram_model.context_size, sequence_length):
                start_idx = end_idx - self.ngram_model.context_size
                processed_logits[:, end_idx - 1] = self.process_last_token_logits(input_ids[:, start_idx:end_idx],
                                                                                  processed_logits[:, end_idx - 1])

        if self.is_vllm:
            processed_logits = processed_logits.squeeze(0)
        return processed_logits


def load_or_train_ngram_model(tokenizer, device, args, dataset: Dataset):
    ngram_model_path = get_ngram_model_path(dataset.name)
    if (not args.retrain_ngram) and os.path.exists(ngram_model_path):
        # ngram_model = torch.load(ngram_model_path).to(device)
        ngram_model = torch.load(ngram_model_path)
    else:
        os.system('rm -rf ' + ngram_model_path)
        print("Training new ngram model")
        ngram_model = train_ngram_model(tokenizer, device, dataset, args.ngram_context_size, args.ngram_epoch)
        print(f"Saving ngram model to {ngram_model_path}")
        torch.save(ngram_model, ngram_model_path)
    ngram_model.context_size = args.ngram_context_size
    return ngram_model


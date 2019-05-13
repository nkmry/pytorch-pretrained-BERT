from typing import List, Dict
import subprocess
import argparse
import os
import logging
import random
from pathlib import Path
from tqdm import tqdm, trange

import numpy as np
import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

import sentencepiece as spm

from allennlp.data import Instance, Vocabulary
from allennlp.data.fields import TextField
from allennlp.data.dataset_readers import DatasetReader, LanguageModelingReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import WordSplitter
from allennlp.data.iterators import BucketIterator

from allennlp.models import Model
from allennlp.modules import TokenEmbedder, Embedding
from allennlp.modules.text_field_embedders import (
    BasicTextFieldEmbedder, TextFieldEmbedder)
from allennlp.nn.util import get_text_field_mask
from allennlp.training import Trainer
from allennlp.training.metrics import CategoricalAccuracy

from pytorch_pretrained_bert import (
    OpenAIAdam, GPT2Config, GPT2LMHead, GPT2Model, GPT2SpTokenizer
)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class SentensepieceSplitter(WordSplitter):
    def __init__(self, pretrained_model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(pretrained_model_path)

    def split_words(self, sentence: str):
        return [e for e in self.sp.EncodeAsPieces(sentence) if e != "_"]


class GPT2LMHeadModel(Model):
    def __init__(self, config: GPT2Config, vocab: Vocabulary):
        super().__init__(vocab)
        self.transformer = GPT2Model(config)
        self.lm_head = GPT2LMHead(self.transformer.wte.weight, config)
        self.accuracy = CategoricalAccuracy()

    def forward(self,
                input_tokens: Dict[str, Tensor],
                output_tokens: Dict[str, Tensor] = None) -> Dict[str, Tensor]:
        mask = get_text_field_mask(input_tokens)
        if output_tokens is not None:
            loss = self.lm(input_ids=input_tokens['tokens'],
                           lm_labels=output_tokens['tokens'])
        else:
            pass
        hidden_states, presents = self.transformer(
            input_ids=input_tokens['tokens'])
        lm_logits = self.lm_head(hidden_states)
        output = {'lm_logits': lm_logits}
        if output_tokens is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[:, :-1].contiguous()
            shift_labels = output_tokens['tokens'][:, 1:].contiguous()
            self.accuracy(shift_logits, shift_labels, mask)

            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            output['loss'] = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
        return output




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sp_model', type=str, required=True,
                        help='a path to Sentencepeice model')
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--train_dataset', type=str, default='')
    parser.add_argument('--eval_dataset', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)
    parser.add_argument('--warmup_proportion', type=float, default=0.002)
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lm_coef', type=float, default=0.9)
    parser.add_argument('--n_valid', type=int, default=374)
    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        # use all GPUs
        cuda_device = [i for i in range(n_gpu)]
    else:
        cuda_device = -1
    logger.info("device: {}, n_gpu {}".format(device, n_gpu))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load tokenizer and model
    # This loading functions also add new tokens and embeddings called `special tokens`
    # These new embeddings will be fine-tuned on the RocStories dataset
    special_tokens = ['_start_', '_delimiter_', '_classify_']
    tokenizer = GPT2SpTokenizer(args.sp_model, special_tokens)
    special_tokens_ids = list(tokenizer.convert_tokens_to_ids(token) for token in special_tokens)
    model = GPT2LMHeadModel(GPT2Config(**{
        "initializer_range": 0.02,
        "layer_norm_epsilon": 1e-05,
        "n_ctx": 1024,
        "n_embd": 768,
        "n_head": 12,
        "n_layer": 12,
        "n_positions": 1024,
        "vocab_size_or_config_json_file": len(tokenizer)
    }))
    model.to(device)

    num_sentences = int(subprocess.check_output(
        ['wc', '-l', args.train_dataset], encoding='utf-8').split()[0])
    reader = LanguageModelingReader(tokenizer=WordTokenizer(
        word_splitter=SentensepieceSplitter(args.sp_model)), lazy=True)
    train_dataset = reader.read(args.train_dataset)
    vocab = Vocabulary.from_instances(train_dataset)
    iterator = BucketIterator(sorting_keys=[("input_tokens", "num_tokens")],
                              batch_size=args.train_batch_size,
                              instances_per_epoch=num_sentences)
    iterator.index_with(vocab)
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    num_train_optimization_steps =\
        len(train_dataset) * args.num_train_epochs // args.train_batch_size
    optimizer = OpenAIAdam(optimizer_grouped_parameters,
                           lr=args.learning_rate,
                           warmup=args.warmup_proportion,
                           max_grad_norm=args.max_grad_norm,
                           weight_decay=args.weight_decay,
                           t_total=num_train_optimization_steps)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      patience=1,
                      num_epochs=20,
                      cuda_device=cuda_device,
                      serialization_dir=args.output_dir)
    trainer.train()

    #save
    with (Path(args.output_dir) / "model.th").open('wb') as f:
        torch.save(model.state_dict(), f)
    vocab.save_to_files(args.output_dir)


if __name__ == "__main__":
    main()

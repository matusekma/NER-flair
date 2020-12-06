from flair.data import Corpus
from flair.datasets import ColumnCorpus

# this is the folder in which train, test and dev files reside
data_folder = "./data"

# define columns
columns = {0: 'text', 1: 'pos', 2: 'ner'}

# init a corpus using column format, data folder and the names of the train, dev and test files
corpus: Corpus = ColumnCorpus(data_folder, columns,
                              encoding="latin1",
                              train_file='train.txt',
                              test_file='test.txt',
                              dev_file='dev.txt')

from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from typing import List

import flair, torch
flair.device = torch.device('cuda:0') 

tag_type = 'ner'

# dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# initialize embeddings
embedding_types: List[TokenEmbeddings] = [

    # GloVe embeddings
    WordEmbeddings('glove'),

    # contextual string embeddings, forward
    FlairEmbeddings('news-forward'),

    # contextual string embeddings, backward
    FlairEmbeddings('news-backward'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type)

# initialize trainer
from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

# train
trainer.train('resources/taggers/ner',
              train_with_dev=True,  
              max_epochs=150,
                )

# Manual testing

from flair.models import SequenceTagger
from flair.data import Sentence
model = SequenceTagger.load('final-model.pt')

# create example sentence
sentence = Sentence("He was sworn in as vice president just three weeks ago under a peace accord reached earlier this year.")

# predict tags and print
model.predict(sentence)

print(sentence.to_tagged_string())
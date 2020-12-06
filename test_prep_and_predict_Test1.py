import pandas as pd

df = pd.read_csv("./data/Test1NER.csv", sep=";", names=["Sentence #", "Word", "Pos"], encoding = "latin1")
df.fillna('', inplace=True)

def fill_sentence_numbers(df):
    sentences = df["Sentence #"].values
    current = sentences[0]
    new_sentence_nums = []
    for s in sentences:
        if s != '':
            current = s
        new_sentence_nums.append(int(current.replace("Sentence: ", "")))
    df["Sentence_num"] = new_sentence_nums
        
fill_sentence_numbers(df)

sentences = df.groupby('Sentence_num').apply(lambda row: " ".join(row["Word"]))

from flair.models import SequenceTagger
from flair.data import Sentence

model = SequenceTagger.load('final-model.pt')

tagged_sentences = []
# create example sentence
for sentence_string in sentences:
    sentence = Sentence(text=sentence_string, use_tokenizer=False)

    # predict
    model.predict(sentence)
    tagged_sentences.append(sentence.to_tagged_string())

import re
ner_regex = re.compile('^<[B|I]-.+>')

# set all predictions to O
df["Predicted"] = "O"

row_index = 0
for tagged_sentence in tagged_sentences:
    sentence_tokens = tagged_sentence.split(" ")
    for token in sentence_tokens:
        if ner_regex.match(token) is not None:
            # set previous word tag
            df.iat[row_index - 1, 4] = token[1:-1]
        else:
            row_index += 1
print(row_index)
print(len(df))

df.to_csv("Test1NER_Prediction.csv", sep=";", index=False, columns=["Sentence #", "Word", "Pos", "Predicted"], header=["Sentences", "Word", "POS", "Predicted"])
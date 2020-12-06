import pandas as pd

df = pd.read_csv("./data/TrainNER.csv", sep=";", encoding = "ISO-8859-1")
df.fillna('', inplace=True)


def fill_sentence_numbers(df):
    sentences = df["Sentence #"].values
    current = sentences[0]
    new_sentence_nums = []
    for s in sentences:
        if s != '':
            current = s
        new_sentence_nums.append(int(current.replace("Sentence: ", "")))
    df["Sentence #"] = new_sentence_nums
        
fill_sentence_numbers(df)


df = df[df["Word"] != ""]
df

train = df[df["Sentence #"] <= 6000]
dev = df[(df["Sentence #"] > 6000) & (df["Sentence #"] >= 7500)]
test = df[df["Sentence #"] > 7500]

def write_in_proper_format(df, filename):
    prev_sentence_num = df.iloc[0]["Sentence #"]
    
    for i in range(len(df)):
        row = df.iloc[i]
        current_sentence_num = row["Sentence #"]
        
        with open(filename, "a") as file:
            if current_sentence_num != prev_sentence_num:
                file.write("\n")
            file.write(row["Word"])
            file.write(" ")
            file.write(row["POS"])
            file.write(" ")
            file.write(row["Tag"])
            file.write("\n")
        prev_sentence_num = current_sentence_num

output_csv_train = "./data/train.txt"
output_csv_dev = "./data/dev.txt"
output_csv_test = "./data/test.txt"
write_in_proper_format(train, output_csv_train)
write_in_proper_format(dev, output_csv_dev)
write_in_proper_format(test, output_csv_test)
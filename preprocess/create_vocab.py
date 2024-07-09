import pandas as pd
from tokenizers import ByteLevelBPETokenizer

REPORTS = '/home/dasith/Documents/Personal/Academics/chestXpert/Datasets/indiana/cleaned_df.csv'

reports = pd.read_csv(REPORTS)
reports = list(reports['findings'].values)

with open('indiana.txt', 'w') as f:
    for item in reports:
        f.write("%s\n" % item)

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(files='indiana.txt', vocab_size=20000, min_frequency=2, special_tokens=[
    '<pad>',
    '<s>',
    '</s>',
    '<unk>',
    '<mask>',
])

tokenizer.save('./indiana.json', pretty=True)
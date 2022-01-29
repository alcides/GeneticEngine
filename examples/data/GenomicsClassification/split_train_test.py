import pandas as pd
from sklearn.model_selection import train_test_split

data=pd.read_csv('SRSF1.tsv', sep="\t")

# Drop exon strength column as it is a feature correlated with the class
# data = data.drop('average_cassette_strength', axis=1)

# Drop irrelevant columns
data = data.drop(['target_coordinates'], axis=1)

# Define numeric class
data['exon_group'] = data.exon_group.replace({'CTRL': 0, 'KD': 1})

# Set the class column as the last in the data
y = data.pop("exon_group")
data = pd.concat([data, y], axis=1)

# Split in a stratified faction
x_train, x_test, _, _ = train_test_split(data, data.exon_group, stratify=data.exon_group, test_size=0.2)

x_train.to_csv("Train.tsv", sep="\t", index=False)
x_test.to_csv("Test.tsv", sep="\t", index=False)

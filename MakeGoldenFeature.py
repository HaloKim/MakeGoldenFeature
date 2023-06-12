import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import itertools
from tqdm import tqdm
import json

parser = argparse.ArgumentParser(description='Find GolenFeature')
parser.add_argument("-t", "--task", dest="task", required=True, help='MLtask(If regression, reg)')
parser.add_argument("-l", "--label", dest="label", required=True, help='Label feature name')
parser.add_argument("-i", "--input", dest="input", required=True, help='Input file path')
parser.add_argument("-o", "--output", dest="output", default='GoldenFeatureScore.json', help='Output file path')
args = parser.parse_args()

task = args.task

# Dataload
data = pd.read_csv(args.input)
y = data[args.label]

# Check Data Task
label_mean = np.mean(y)
label_std = np.std(y)

# isRegression
if label_std > 0:  
    task = 'reg'

# Preprocess
def apply_ordinal_encoder(df):
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    for column in df.columns:
        if df[column].dtype == 'object' or df[column].dtype == 'str':
            encoded_data = encoder.fit_transform(df[[column]])
            df[column] = encoded_data
    return df


def scaler(df):
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df)
    return pd.DataFrame(scaled_df, columns=df.columns).astype('float32')


X = apply_ordinal_encoder(data.drop(['가격'], axis=1))
X = scaler(X)


# SET eval model
def evaluate_model(X_train, X_test, y_train, y_test, task):
    if task == 'reg':
        model = DecisionTreeRegressor(random_state=42)
    else:
        model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return score


# Make Golden Feature
def create_new_features(df):
    new_features = pd.DataFrame()
    
    # Feature Combinations
    combinations = list(itertools.combinations(df.columns, 2))

    # plus, multiply, devide, minus
    for col1, col2 in combinations:
        new_features.loc[:, f'{col1}_plus_{col2}'] = df[col1] + df[col2]
        new_features.loc[:, f'{col1}_multiply_{col2}'] = df[col1] * df[col2]
        new_features.loc[:, f'{col1}_divide_{col2}'] = df[col1] / df[col2]
        new_features.loc[:, f'{col1}_minus_{col2}'] = df[col1] - df[col2]

    return new_features

X_train_new = X.copy()
new_features = create_new_features(X_train_new)

# Each Golden Feature Scoring
scores = []
for new_feature in tqdm(new_features.columns):
    new_df = pd.concat([X_train_new, new_features[[new_feature]]], axis=1)

    # Test
    if task == 'reg':
        X_train, X_test, y_train, y_test = train_test_split(new_df, y, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(new_df, y, stratify=y, random_state=42)
    score = evaluate_model(X_train, X_test, y_train, y_test, task)
    scores.append((new_feature, score))

# Sorting Score
sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

# Make Dict
results = []
for feature, score in sorted_scores:
    result = {'Feature': feature, 'Score': score}
    results.append(result)

# Write JSON File
filename = args.output
with open(filename, "w") as outfile:
    json.dump(results, outfile)
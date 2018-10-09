from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle

import warnings
warnings.filterwarnings("ignore")

tf.logging.set_verbosity(tf.logging.DEBUG) # FATAL, WARN, INFO, DEBUG

# Load the dataset
DATA_ROOT = 'C:/data/ccdata/'

# Load the dataset
cc_train_data = pd.read_csv(DATA_ROOT+"/credit data.csv")
cc_prediction_data = pd.read_csv(DATA_ROOT+"/new_applications.csv")

train_test_data = pd.DataFrame(cc_train_data)
train_test_data.rename(columns={'default.payment.next.month': 'default'}, inplace=True)

prediction_data = pd.DataFrame(cc_prediction_data)

train_test_data['BILL_AMT1_LB'] = train_test_data.apply(lambda row: row['BILL_AMT1']/row['LIMIT_BAL'] if(row['BILL_AMT1'] < row['LIMIT_BAL']) else 1.0, axis=1)
train_test_data['BILL_AMT2_LB'] = train_test_data.apply(lambda row: row['BILL_AMT2']/row['LIMIT_BAL'] if(row['BILL_AMT2'] < row['LIMIT_BAL']) else 1.0, axis=1)
train_test_data['BILL_AMT3_LB'] = train_test_data.apply(lambda row: row['BILL_AMT3']/row['LIMIT_BAL'] if(row['BILL_AMT3'] < row['LIMIT_BAL']) else 1.0, axis=1)
train_test_data['BILL_AMT4_LB'] = train_test_data.apply(lambda row: row['BILL_AMT4']/row['LIMIT_BAL'] if(row['BILL_AMT4'] < row['LIMIT_BAL']) else 1.0, axis=1)
train_test_data['BILL_AMT5_LB'] = train_test_data.apply(lambda row: row['BILL_AMT5']/row['LIMIT_BAL'] if(row['BILL_AMT5'] < row['LIMIT_BAL']) else 1.0, axis=1)
train_test_data['BILL_AMT6_LB'] = train_test_data.apply(lambda row: row['BILL_AMT6']/row['LIMIT_BAL'] if(row['BILL_AMT6'] < row['LIMIT_BAL']) else 1.0, axis=1)

train_test_data['PAY_AMT1_BP'] = train_test_data.apply(lambda row: row['PAY_AMT1']/row['BILL_AMT1'] if(row['BILL_AMT1'] > 0) else 0.0, axis=1)
train_test_data['PAY_AMT2_BP'] = train_test_data.apply(lambda row: row['PAY_AMT2']/row['BILL_AMT2'] if(row['BILL_AMT2'] > 0) else 0.0, axis=1)
train_test_data['PAY_AMT3_BP'] = train_test_data.apply(lambda row: row['PAY_AMT3']/row['BILL_AMT3'] if(row['BILL_AMT3'] > 0) else 0.0, axis=1)
train_test_data['PAY_AMT4_BP'] = train_test_data.apply(lambda row: row['PAY_AMT4']/row['BILL_AMT4'] if(row['BILL_AMT4'] > 0) else 0.0, axis=1)
train_test_data['PAY_AMT5_BP'] = train_test_data.apply(lambda row: row['PAY_AMT5']/row['BILL_AMT5'] if(row['BILL_AMT5'] > 0) else 0.0, axis=1)
train_test_data['PAY_AMT6_BP'] = train_test_data.apply(lambda row: row['PAY_AMT6']/row['BILL_AMT6'] if(row['BILL_AMT6'] > 0) else 0.0, axis=1)

prediction_data['BILL_AMT1_LB'] = prediction_data.apply(lambda row: row['BILL_AMT1']/row['LIMIT_BAL'] if(row['BILL_AMT1'] < row['LIMIT_BAL']) else 1.0, axis=1)
prediction_data['BILL_AMT2_LB'] = prediction_data.apply(lambda row: row['BILL_AMT2']/row['LIMIT_BAL'] if(row['BILL_AMT2'] < row['LIMIT_BAL']) else 1.0, axis=1)
prediction_data['BILL_AMT3_LB'] = prediction_data.apply(lambda row: row['BILL_AMT3']/row['LIMIT_BAL'] if(row['BILL_AMT3'] < row['LIMIT_BAL']) else 1.0, axis=1)
prediction_data['BILL_AMT4_LB'] = prediction_data.apply(lambda row: row['BILL_AMT4']/row['LIMIT_BAL'] if(row['BILL_AMT4'] < row['LIMIT_BAL']) else 1.0, axis=1)
prediction_data['BILL_AMT5_LB'] = prediction_data.apply(lambda row: row['BILL_AMT5']/row['LIMIT_BAL'] if(row['BILL_AMT5'] < row['LIMIT_BAL']) else 1.0, axis=1)
prediction_data['BILL_AMT6_LB'] = prediction_data.apply(lambda row: row['BILL_AMT6']/row['LIMIT_BAL'] if(row['BILL_AMT6'] < row['LIMIT_BAL']) else 1.0, axis=1)

prediction_data['PAY_AMT1_BP'] = prediction_data.apply(lambda row: row['PAY_AMT1']/row['BILL_AMT1'] if(row['BILL_AMT1'] > 0) else 0.0, axis=1)
prediction_data['PAY_AMT2_BP'] = prediction_data.apply(lambda row: row['PAY_AMT2']/row['BILL_AMT2'] if(row['BILL_AMT2'] > 0) else 0.0, axis=1)
prediction_data['PAY_AMT3_BP'] = prediction_data.apply(lambda row: row['PAY_AMT3']/row['BILL_AMT3'] if(row['BILL_AMT3'] > 0) else 0.0, axis=1)
prediction_data['PAY_AMT4_BP'] = prediction_data.apply(lambda row: row['PAY_AMT4']/row['BILL_AMT4'] if(row['BILL_AMT4'] > 0) else 0.0, axis=1)
prediction_data['PAY_AMT5_BP'] = prediction_data.apply(lambda row: row['PAY_AMT5']/row['BILL_AMT5'] if(row['BILL_AMT5'] > 0) else 0.0, axis=1)
prediction_data['PAY_AMT6_BP'] = prediction_data.apply(lambda row: row['PAY_AMT6']/row['BILL_AMT6'] if(row['BILL_AMT6'] > 0) else 0.0, axis=1)

Default = train_test_data[train_test_data.default == 1]
NonDefault = train_test_data[train_test_data.default == 0]

# Set X_train equal to 80% of the observations that defaulted.
train = Default.sample(frac=0.8)

# Add 80% of the not-defaulted observations to X_train.
train = pd.concat([train, NonDefault.sample(frac=0.8)], axis=0)

# X_test contains all the observations not in X_train.
test = train_test_data.loc[~train_test_data.index.isin(train.index)]

# Shuffle the dataframes so that the training is done in a random order.
train = shuffle(train)
test = shuffle(test)

COLUMNS = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1_LB', 'BILL_AMT2_LB', 'BILL_AMT3_LB',
           'BILL_AMT4_LB', 'BILL_AMT5_LB', 'BILL_AMT6_LB', 'PAY_AMT1_BP', 'PAY_AMT2_BP', 'PAY_AMT3_BP', 'PAY_AMT4_BP',
           'PAY_AMT5_BP', 'PAY_AMT6_BP', 'default']

FEATURES = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1_LB', 'BILL_AMT2_LB', 'BILL_AMT3_LB',
            'BILL_AMT4_LB', 'BILL_AMT5_LB', 'BILL_AMT6_LB', 'PAY_AMT1_BP', 'PAY_AMT2_BP', 'PAY_AMT3_BP', 'PAY_AMT4_BP',
            'PAY_AMT5_BP', 'PAY_AMT6_BP']

LABEL = 'default'


def input_fn(data_set):
    feature_columns = {k: tf.constant(data_set[k].values) for k in FEATURES}
    labels = tf.constant(data_set[LABEL].values)
    return feature_columns, labels


def predict_fn(data_set):
    feature_columns = {k: tf.constant(data_set[k].values) for k in FEATURES}
    return feature_columns, None


feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]

estimator = tf.contrib.learn.DNNRegressor(
                    feature_columns=feature_cols,
                    optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                    hidden_units=[len(feature_cols), len(feature_cols)*2, len(feature_cols)],
                    dropout=0.1,
                    model_dir='c:\data\model')

estimator.fit(input_fn=lambda: input_fn(train), steps=100)

score = estimator.evaluate(input_fn=lambda: input_fn(test), steps=len(test)/10)

print('\nEvaluate Loss: {0:f}%\n'.format(score["loss"]))

print('\nTest Accuracy: {0:f}%\n'.format((1-score["loss"])*100))

predictions = estimator.predict_scores(input_fn=lambda: predict_fn(prediction_data))

for prediction in predictions:
    print('result: {}'.format(prediction)+' predicted default: {}'.format(1 if(prediction >= 0.5) else 0))



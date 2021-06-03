import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow.keras import Sequential
from keras import regularizers
from keras import layers
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

data = pd.read_csv('../data/data_magic.csv')
val_data = pd.read_csv('../data/val_magic.csv')
test = val_data.drop(['ID_code'], axis=1)
print(data.head())

# Set params for LGBM
params = {
    'feature_fraction': 0.05,
    'learning_rate': 0.01,
    'max_depth': -1,
    'metric': 'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,
    'num_threads': 6,
    'tree_learner': 'serial',
    'verbosity': 1,
    'bagging_freq': 5,
    'bagging_fraction': 0.4,
    'boost_from_average': 'false',
    'boost': 'gbdt',
    'objective': 'binary',
}

num_folds = 10

# Get the names of the features we want to use and the target
features = [c for c in data.columns if c not in ['ID_code', 'target']]
target = data['target']


# Create cross validation for lgbm
def pred_lgbm(params, num_rounds):
    print('Loading test dataset..')

    print('Starting cross validation..')
    folds = StratifiedKFold(n_splits=num_folds, shuffle=False)

    # Save predictions
    predictions = []
    oof = np.zeros(len(data))

    for fold, (trn_idx, val_idx) in enumerate(folds.split(data.values, target.values)):
        print(f"Fold {fold}")
        train_data = lgb.Dataset(data.iloc[trn_idx][features], label=target.iloc[trn_idx])
        test_data = lgb.Dataset(data.iloc[val_idx][features], label=target.iloc[val_idx])

        # Train the model
        lgbm_clf = lgb.train(params, train_data, num_rounds, valid_sets=[train_data, test_data], verbose_eval=1000,
                             early_stopping_rounds=3000)

        # Run predictions
        oof[val_idx] = lgbm_clf.predict(data.iloc[val_idx][features], num_iteration=lgbm_clf.best_iteration)
        predictions.append(lgbm_clf.predict(test, num_iteration=lgbm_clf.best_iteration))

    print('CV Score: ', roc_auc_score(target, oof))
    CV_score = [target, oof]
    return predictions, CV_score


# Save the final prediction
test_pred_lgbm, CV_score = pred_lgbm(params, 1000000)
print(test_pred_lgbm)

# Change to numpy array and divide every value by the amount of folds
test_pred_lgbm = np.array(np.sum(test_pred_lgbm, axis=0)) / num_folds
d = {'ID_code': val_data['ID_code'], 'LGBM_pred': test_pred_lgbm}

# Save as a dataset to evaluate later
result_data = pd.DataFrame(data=d)
pd.DataFrame(result_data).to_csv('../data/results/LGBM_result.csv', index=False)

# Create the NN model
# Set the input dimensions
input_dim = data.shape[1]
print('Input dimension: ', input_dim)

# Define new model
def nn_model(input_dim):
    print('Building the model..')
    model = Sequential()
    model.add(layers.Dense(128, input_dim=input_dim, activation='relu', kernel_regularizer=regularizers.l2(0.005)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, activation='sigmoid'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(16, input_dim=input_dim, activation='sigmoid', kernel_regularizer=regularizers.l2(0.005)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

model = nn_model(input_dim)
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# Get the results
def pred_nn(test_size, batch_size, epochs):
    # Split the data to train and test
    y = data['target']
    X = data.drop(['target', 'ID_code'], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=42)
    # Introduce early stopping to prevent overfitting
    early = EarlyStopping(monitor="val_loss",
                          mode="min",
                          patience=9)

    print('Fitting the model..')
    history = model.fit(x_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[early],
                        validation_data=(x_test, y_test))
    print(model.summary())

    # Run predictions
    prediction = model.predict(test)
    print('Model built successfully!')
    return prediction, history

test_pred_nn, nn_history = pred_nn(0.25, 512, 25)
flat_test_pred_nn = [item for sublist in test_pred_nn for item in sublist]

# Create a dictionary to be transformed to a dataframe
d = {'ID_code': val_data['ID_code'], 'NN_pred': flat_test_pred_nn}

# Save the history to a dictionary for both NN and LGBM
history_dict = {
    'acc': nn_history.history['accuracy'],
    'val_acc': nn_history.history['val_accuracy'],
    'loss': nn_history.history['loss'],
    'val_loss': nn_history.history['val_loss'],
    'LGBM_CV': CV_score
}

np.save('../data/history/history.npy', history_dict)

# Save this one as a dataset too
result_data = pd.DataFrame(data=d)
pd.DataFrame(result_data).to_csv('../data/results/NN_result.csv', index=False)

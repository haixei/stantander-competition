import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime

ID_code = pd.read_csv('../data/test.csv')['ID_code']
lgbm_result = pd.read_csv('../data/results/LGBM_result.csv')['LGBM_pred']
nn_result = pd.read_csv('../data/results/NN_result.csv')['NN_pred']
history = np.load('../data/history/history.npy', allow_pickle=True)

# Visualise the NN accuracy
acc = history.item()['acc']
val_acc = history.item()['val_acc']
loss = history.item()['loss']
val_loss = history.item()['val_loss']

epochs_range = list(range(25))

# Create a line plot
history_fig = make_subplots(rows=1, cols=2, subplot_titles=('Training & Validation Acc.', 'Training & Validation Loss'))
history_fig.add_trace(
    go.Scatter(x=epochs_range, y=acc, mode='lines', name='Acc.', marker=dict(color='#16d2f0')),
    row=1, col=1,
)

history_fig.add_trace(
    go.Scatter(x=epochs_range, y=val_acc, mode='lines', name='Val. Acc.', marker=dict(color='#a579ed')),
    row=1, col=1,
)

history_fig.add_trace(
    go.Scatter(x=epochs_range, y=loss, mode='lines', name='Loss', marker=dict(color='#16d2f0')),
    row=1, col=2,
)

history_fig.add_trace(
    go.Scatter(x=epochs_range, y=val_loss, mode='lines', name='Val. Loss', marker=dict(color='#a579ed')),
    row=1, col=2,
)

# history_fig.show()

# Blend and created a final dataframe submission
result_df = pd.concat([nn_result, lgbm_result], axis=1, join="inner")

def blend(nn_pred, lgbm_pred):
    return (0.75 * nn_pred) + (0.25 * lgbm_pred)


blended = result_df.apply(lambda row: row['LGBM_pred'] + row['NN_pred'], axis=1)
blended_dict = {'ID_code': ID_code, 'target': blended}
blended_df = pd.DataFrame(data=blended_dict)

print(blended_df.head())

# Save the submission
date = datetime.today().strftime('%Y-%m-%d')
submission_name = '../data/results/submission_' + date + '.csv'
pd.DataFrame(blended_df).to_csv(submission_name, index=False)

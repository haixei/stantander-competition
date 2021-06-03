import pandas as pd
import plotly.express as pltx
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.graph_objects as go
from numpy import sqrt, ceil

# Loading the data
data = pd.read_csv('../data/train.csv')
print(data.head())
print(data.describe())
print('Shape of the dataset: ', data.shape)
print('Type of the information: ', data.target.dtype)
print('Missing data: ', data.isnull().sum().sum())

data = data.drop(['ID_code'], axis=1)

# Looking into the target
X = data['target']
y = data.drop(['target'], axis=1)

# Plot the target
target_str = data.target.apply(str)
hist_y_fig = pltx.histogram(target_str, x="target", color_discrete_sequence=['#a579ed'], title='Target distribution')
# >> hist_y_fig.show()

hist_y_v_fig = pltx.violin(data, y=data.index, x="target", color="target", box=True, points="all",
                           color_discrete_sequence=['#a579ed', '#bab0f7'], title='Target distribution (violin)')
# >> hist_y_v_fig.show()

# Show distribution between the features and the target
def show_dist_target(cols, num_of_rows, data):
    cols = data[cols]
    data = data.head(num_of_rows)
    cols_and_rows = int(ceil(sqrt(cols.shape[1])))
    multiplot_fig = make_subplots(rows=cols_and_rows, cols=cols_and_rows,
                                  horizontal_spacing=0.02)
    row = 1
    col = 1

    for column in cols:
        # Add new figure to the canvas
        t1 = data.loc[data['target'] == 1]
        t0 = data.loc[data['target'] == 0]
        hist_data = [t1[column], t0[column]]

        group_labels = ['0', '1']
        colors = ['#914dff', '#aab3f2']

        fig = ff.create_distplot(hist_data, group_labels, show_hist=True, colors=colors, show_rug=False)
        multiplot_fig.add_traces(fig['data'], rows=row, cols=col)
        multiplot_fig.update_layout(barmode='stack', showlegend=False, font_size=10)

        # Add titles
        multiplot_fig.update_xaxes(title_text=column, row=row, col=col)

        col += 1
        if col > cols_and_rows:
            col = 1
            row += 1

    # Show the plots
    multiplot_fig.update_layout(title_text="Distribution of data")
    multiplot_fig.show()


cols = data.columns.values[28:53]
# >> show_dist_target(cols, 100000, data)

# Show the most important features
corr = data.corr().target
corr = corr.sort_values(ascending=False)
print('Most correlated to the target:\n', corr[1:11])
print('Least correlated to the target:\n', corr[191:202])

# Plot the top 3 features
top_list = corr[1:4]
cols = []
for i, v in top_list.items():
    cols.append(i)

# Create the canvas with the plots
data_batch = data.head(10000)
sc_corr_fig = make_subplots(rows=1, cols=3)
sc_corr_fig.add_trace(
    go.Histogram(x=data_batch[cols[0]], name=cols[0], marker=dict(color='#835AF1')),
    row=1, col=1
)

sc_corr_fig.add_trace(
    go.Histogram(x=data_batch[cols[1]], name=cols[1], marker=dict(color='#7FA6EE')),
    row=1, col=2
)

sc_corr_fig.add_trace(
    go.Histogram(x=data_batch[cols[2]], name=cols[2], marker=dict(color='#42f5f2')),
    row=1, col=3
)

sc_corr_fig.update_layout(title_text="Top 3 features (distribution)")
sc_corr_fig.update_traces(opacity=0.75)
# >> sc_corr_fig.show()

# Show the correlation between the features
feat_corr = data.corr().values.flatten()
feat_corr = feat_corr[feat_corr != 1]
feat_corr_fig = ff.create_distplot([feat_corr], group_labels=['Correlation'], colors=['#914dff'])
# >> feat_corr_fig.show()

def show_scatter_plots(cols, num_of_rows, data):
    data = data.head(num_of_rows)
    cols = data[cols]

    scatter_fig = pltx.scatter_matrix(cols, color='target', color_continuous_scale=['#a579ed', '#42f5f2'])
    scatter_fig.show()


# Get the top 5 features
top_5_list = corr[0:5]
cols = []
for i, v in top_5_list.items():
    cols.append(i)

# >> show_scatter_plots(cols, 1000, data)

# Maximum and minimum values by column
data_max = data.max().to_frame().T.to_numpy().flatten()
data_min = data.min().to_frame().T.to_numpy().flatten()

colors = ['#914dff', '#aab3f2']
dist_fig = ff.create_distplot([data_max, data_min], ['max', 'min'], show_hist=True, colors=colors)
dist_fig.update_layout(title='Distribution of max/min values of the columns')
# >> dist_fig.show()

# Skewness of the data set
skew = data.skew().to_numpy().flatten()
skew_dist_fig = ff.create_distplot([skew], ['skew'], show_hist=True, colors=['#914dff'])
skew_dist_fig.update_layout(title='Skewness of the columns')
# >> skew_dist_fig.show()

# Mean of the data set
t1 = data.loc[data['target'] == 1]
t0 = data.loc[data['target'] == 0]
t1_mean = t1.mean().to_numpy().flatten()
t0_mean = t0.mean().to_numpy().flatten()

mean_dist_fig = ff.create_distplot([t1_mean, t0_mean], ['target = 1', 'target = 0'], show_hist=True, colors=colors)
mean_dist_fig.update_layout(title='Mean of the columns')
# >> mean_dist_fig.show()

# Std of the data set
t1_std = t1.std().to_numpy().flatten()
t0_std = t0.std().to_numpy().flatten()

std_dist_fig = ff.create_distplot([t1_std, t0_std], ['target = 1', 'target = 0'], show_hist=True, colors=colors)
std_dist_fig.update_layout(title='Std of the columns')
# >> std_dist_fig.show()

# Count unique values in columns
unique = data.nunique()
unique_top_10 = unique.sort_values(ascending=True)[:10]
print('Top 10 features having the least unique values:\n', unique_top_10)


import pandas as pd

data = pd.read_csv('../data/train.csv')
val_data = pd.read_csv('../data/test.csv')

# Extract feature names that are not the target
target = 'target'
features = [i for i in data.columns if i != target]

# Create names of the new features
has_one = [f'var_{i}_has_one' for i in range(200)]
has_zero = [f'var_{i}_has_zero' for i in range(200)]
not_u = [f'var_{i}_not_unique' for i in range(200)]

# This part is responsible for the "magic" that will extract some important information
# about the dataset, and help us achieve better accuracy
for feature in features:
    print('Working on feature:', feature)
    data[feature + '_has_one'] = 0
    data[feature + '_has_zero'] = 0

    # Find features that appear in the dataset with the target value of 1
    f_1 = data.loc[data[target] == 1, feature].value_counts()

    # Divide into two categories, ones that appear at least one time and the ones that appear
    # at least two times
    f_1_1 = set(f_1.index[f_1 > 1])
    f_0_1 = set(f_1.index[f_1 > 0])

    # Same situation here but with the target value of 0 instead
    f_0 = data.loc[data[target] == 0, feature].value_counts()
    f_0_0 = set(f_0.index[f_0 > 1])
    f_1_0 = set(f_0.index[f_0 > 0])

    # Now we extract the information into our new features
    data.loc[data[target] == 1, feature + '_has_one'] = data.loc[data[target] == 1, feature].isin(f_1_1).astype(int)
    data.loc[data[target] == 0, feature + '_has_one'] = data.loc[data[target] == 0, feature].isin(f_0_1).astype(int)

    data.loc[data[target] == 1, feature + '_has_zero'] = data.loc[data[target] == 1, feature].isin(f_1_0).astype(int)
    data.loc[data[target] == 0, feature + '_has_zero'] = data.loc[data[target] == 0, feature].isin(f_0_0).astype(int)

# Do it similar way for the test data set
for feature in features:
    print('Working on val. feature:', feature)
    val_data[feature + '_has_one'] = 0
    val_data[feature + '_has_zero'] = 0
    f_1 = data.loc[data[target] == 1, feature].unique()
    f_0 = data.loc[data[target] == 0, feature].unique()
    val_data.loc[:, feature + '_has_one'] = val_data[feature].isin(f_1).astype(int)
    val_data.loc[:, feature + '_has_zero'] = val_data[feature].isin(f_0).astype(int)

# Check if everything got saved properly
print(data['var_0_has_one'].value_counts())
print(data.shape)

# Save new magic dataframes as a csv file
pd.DataFrame(data).to_csv('../data/data_magic.csv', index=False)
pd.DataFrame(val_data).to_csv('../data/val_magic.csv', index=False)
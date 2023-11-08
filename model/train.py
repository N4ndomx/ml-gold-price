from os import PathLike
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump
import pandas as pd
import pathlib
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# # Function for comparing different approaches
# def score_dataset(X_train, X_valid, y_train, y_valid):
#     print ('Training model.. ')
#     model = RandomForestRegressor(n_estimators=10, random_state=0)
#     model.fit(X_train, y_train)
#     preds = model.predict(X_valid)
#     return mean_absolute_error(y_valid, preds)

# Function for comparing different models
def score_model(model, X_t, X_v, y_t, y_v):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

df = pd.read_csv(pathlib.Path('data/dataset.csv'))


print(df.dtypes)
y = df.Price
X = df.drop('Price', axis = 1)
X = X.drop('Vin', axis = 1)
X = X.drop('Model', axis = 1)

x_train_full, x_valid_full, y_train, y_valid = train_test_split(X,y, test_size = 0.2)


# Get list of categorical variables
s = (x_train_full.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)

# Make copy to avoid changing original data 
label_X_train = x_train_full.copy()
label_X_valid = x_valid_full.copy()

# Apply ordinal encoder to each column with categorical data
ordinal_encoder = OrdinalEncoder()
label_X_train[object_cols] = ordinal_encoder.fit_transform(x_train_full[object_cols])
label_X_valid[object_cols] = ordinal_encoder.transform(x_valid_full[object_cols])

print(label_X_train.columns)
# Select numerical columns
numerical_cols = [cname for cname in x_train_full.columns if x_train_full[cname].dtype in ['int64', 'float64']]

# Define the models
model_2 = RandomForestRegressor(n_estimators=10, random_state=0)

print(label_X_train.head())
mae = score_model(model_2,label_X_train,label_X_valid, y_train, y_valid)
print("Model MAE: ", mae)


print ('Saving model...')

dump(model_2, pathlib.Path('model/model_ps.joblib'))

# # Keep selected columns only
# my_cols = object_cols + numerical_cols
# X_train = pd.concat([label_X_train, x_train_full[numerical_cols]], axis=1)
# X_valid = pd.concat([label_X_valid, x_valid_full[numerical_cols]], axis=1)

# print( df.isnull().values.any())

# print("MAE from Approach 2 (Ordinal Encoding):") 
# print(score_dataset(label_X_train,label_X_valid, y_train, y_valid))



# print(X_train.head())




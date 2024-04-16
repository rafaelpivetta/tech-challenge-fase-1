import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('archive/insurance.csv')

label_encoder = LabelEncoder()

dataset['genero_type'] = label_encoder.fit_transform(dataset['gênero'])
dataset['fumante_type'] = label_encoder.fit_transform(dataset['fumante'])
dataset['regiao_type'] = label_encoder.fit_transform(dataset['região'])

dataset_tratado = dataset.drop(columns = [ "fumante", "região", "gênero"]).copy()

X = dataset_tratado.drop(columns=['encargos'], axis=1)
y = dataset_tratado['encargos']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Normalização
scaler = MinMaxScaler() #Normalizacao dos dados (0-1)

scaler.fit(X_train)
x_train_min_max_scaled = scaler.transform(X_train)
x_test_min_max_scaled= scaler.transform(X_test)


#Padronização
scaler = StandardScaler()

scaler.fit(X_train)
x_train_standard_scaled = scaler.transform(X_train)
x_test_standard_scaled  = scaler.transform(X_test)

def evaluate_model(y_test, predictions):
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse) 
    r2 = r2_score(y_test, predictions)
    errors = np.abs(y_test - predictions)
    relative_errors = errors / np.abs(y_test)
    mape = np.mean(relative_errors) * 100
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    print('r²', r2)
    print(f"O MAPE é: {mape:.2f}%")

def run_model(model, X_train, y_train, X_test, tipo_scaling):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print()
    print(f"{model} - {tipo_scaling}")
    evaluate_model(y_test, predictions)


run_model(LinearRegression(), X_train, y_train, X_test, 'Sem os escalonadores')
run_model(LinearRegression(), x_train_min_max_scaled, y_train, x_test_min_max_scaled, 'Normalização')
run_model(LinearRegression(), x_train_standard_scaled, y_train, x_test_standard_scaled, 'Padronização')

run_model(DecisionTreeRegressor(), X_train, y_train, X_test, 'Sem os escalonadores')
run_model(DecisionTreeRegressor(), x_train_min_max_scaled, y_train, x_test_min_max_scaled, 'Normalização')
run_model(DecisionTreeRegressor(), x_train_standard_scaled, y_train, x_test_standard_scaled, 'Padronização')

run_model(RandomForestRegressor(), X_train, y_train, X_test, 'Sem os escalonadores')
run_model(RandomForestRegressor(), x_train_min_max_scaled, y_train, x_test_min_max_scaled, 'Normalização')
run_model(RandomForestRegressor(), x_train_standard_scaled, y_train, x_test_standard_scaled, 'Padronização')

run_model(GradientBoostingRegressor(), X_train, y_train, X_test, 'Sem os escalonadores')
run_model(GradientBoostingRegressor(), x_train_min_max_scaled, y_train, x_test_min_max_scaled, 'Normalização')
run_model(GradientBoostingRegressor(), x_train_standard_scaled, y_train, x_test_standard_scaled, 'Padronização')
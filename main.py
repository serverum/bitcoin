# подключаем gridsearch для поиска перебором по диапазонам наилучших параметров из библиотеки sklearn
import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler  # импортируем скелер данных, для приведения их в единый вид
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import max_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
# библиотека пикл для сохранения данных нашей обученной модели
import pickle

# распаковка архива с курсами криптовалют, курсы взяты с https://www.cryptodatadownload.com/data/binance/
# import zipfile, os
#
# b = os.getcwd()
# c = os.path.join(b, "bitfinex")
# print(c)
#
# with zipfile.ZipFile('BitFinexData.zip', 'r') as zfile:
#     zfile.extractall(c)

bitcoin = pd.read_csv("bitfinex/Bitfinex_BTCUSD_d.csv", skiprows=1)
bitcoin.set_index('unix', inplace=True)
bitcoin = bitcoin[::-1]
bitcoin.reset_index(inplace=True)
bitcoin.drop('unix', axis=1, inplace=True)
bitcoin['date'] = pd.to_datetime(bitcoin['date'])
print(bitcoin)

# рисуем графики

# plt.plot(bitcoin["date"], (bitcoin["open"] - bitcoin["close"]), label="Diff")
# plt.legend()
# plt.show()

# Feature Engineering

bitcoin["openclose_diff"] = bitcoin["open"] - bitcoin["close"]
bitcoin["highlow_diff"] = bitcoin["high"] - bitcoin["low"]
bitcoin["open2high"] = bitcoin["openclose_diff"] / bitcoin["highlow_diff"]

# print(bitcoin[['date', 'close']].head(50))
# shift(1) сдвигает на одно поле вперед, так как max/mean и др. считают учитывая текущий день, т.е. 6 вместо 7-ми
bitcoin['close_max_7d'] = bitcoin['close'].shift(1).rolling(window=7).max()
bitcoin['open_mean_14d'] = bitcoin['open'].shift(1).rolling(window=14).mean()
bitcoin['weekday'] = bitcoin['date'].dt.weekday
bitcoin['year'] = bitcoin['date'].dt.year
bitcoin['month'] = bitcoin['date'].dt.month

print("Проверка первых 20ти записей")
print(bitcoin.head(20))

for day in range(1, 15):
    bitcoin[f'close_d{day}'] = bitcoin['close'].shift(day)  # используем форматированные строки + shift(day) сдвигает

df = pd.get_dummies(bitcoin,
                    columns=['year', 'month', 'weekday'])  # приведем в категориальные все необходимые поля (0 и 1)
df.fillna(method='bfill')  # заполним все пустые данные / df.dropna() - второй способ, удалить пустые значения

# удаляем лишние данные, мешающие процессу обработки

df.drop('date', axis=1, inplace=True)
df.drop('symbol', axis=1, inplace=True)

# подключаем скэлер

# scaler = StandardScaler()
# scaler.fit(df)
#
# df = pd.DataFrame(data=scaler.transform(df), columns=df.columns)

# Заполняем значения для предсказывания, т.е. X и y, т.е выборку
df.dropna(inplace=True)
X = df.drop(['close', 'high', 'low', 'open'], axis=1)
y = df['close']

print(df.head(15))
# без random_state=42 будет выдавать случайный результат
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# X_train = X_train.reshape(-1, 1)
# y_train = y_train.reshape(-1, 1)
# X_test = X_test.reshape(-1, 1)

# создаем функцию для передачи моделей ML в качестве аргумента

def check_model(model, coef=True):
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    print("max_error", max_error(y_pred, y_test))
    print("mean_absolute_error", mean_absolute_error(y_pred, y_test))

    print(f'train_score = {train_score}, test_score = {test_score}')
    if coef:
        print(pd.DataFrame(data=[model.coef_], columns=X.columns).T)
    else:
        print(pd.DataFrame(data=[model.feature_importances_], columns=X.columns).T)


# вывод весов по колонкам с коэффициентами
# pd.DataFrame(data=[model.coef_], columns=X.columns).T

model = LinearRegression()
check_model(model, True)

# начинаем работу с gridsearchcv

model = RandomForestRegressor(random_state=42)  # random_state=42 для повторимости результатов на всех машинах
# составляем набор из параметров(param_grid) для передачи в GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 400],
    'min_samples_split': [2, 5, 10]
}

# GridSearchCV принимает 3 аргумента, 1-ый сама модель, 2-й решетка со значениями, 3-ий, оценка погрешности
# 3-й аргумент https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
# 4-й аргумент cv кол-во прогонов кросс-валидации через разделенные данные датасета для исключения случайности

gs = GridSearchCV(model, param_grid, scoring='max_error', cv=3)

# чтобы запустить GridSearch, его надо запустить так gs.fit(X_train, y_train), передав тренировочную выборку на обучение

gs.fit(X_train, y_train)

# лучшие параметры, которые он нашел, выводятся командой gs.best_params_
print(gs.best_params_)

# лучший результат у лучших параметров выводится через gs.best_score_
print(gs.best_score_)

# лучший эстиматор, т.е. модель выводится так
best_model = gs.best_estimator_  # сохраняем в переменную лучшую модель с параметрами, найденную гридсерч
print(gs.best_estimator_)

# как сохранить эту модель, для дальнейшего использования, обычно через import pickle
# как использовать https://wiki.python.org/moin/UsingPickle
# 1-ый аргумент наша модель, 2-й открытие файла для сохранения
pickle.dump(best_model, open('RF.model', 'wb'))

# чтобы загрузить обратно нужно указать pickle.load(open('наш файл', 'rb'))

loaded_model = pickle.load(open('RF.model', 'rb'))

print(loaded_model.predict(X_test))

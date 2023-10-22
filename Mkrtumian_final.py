# импортируем библиотеки
import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

# формируем боковую панель
st.sidebar.header('Введите значения для расчета ширины и глубины сварного шва')

# создаем функцию, которая берёт данные, введённые пользователем
def user_input_features():
    IW = st.sidebar.number_input('Величина сварочного тока(IW)')
    IF = st.sidebar.number_input('Ток фокусировки электронного пучка (IF)')
    VW = st.sidebar.number_input('Скорость сварки (VW)')
    FP = st.sidebar.number_input('Расстояние от поверхности образцов до электронно-оптической системы (FP)')
    data = {'IW': IW,
            'IF': IF,
            'VW': VW,
            'FP': FP}
    features = pd.DataFrame(data, index=[0])
    return features

# Записываем в переменную df датафрейм, сформированный функцией user_input_features()
df_2 = user_input_features()

# Загружаем набор данных
df = pd.read_csv(r'C:\Users\l.mkrtumyan\ebw_data.csv')

X = df.drop(["Width", "Depth"], axis=1)  # убираем целевые переменные
Y = df[["Width", "Depth"]].copy()  # копируем, чтобы не трогать исходный датафрейм

# Деление датасета на обучающую и тестовую выборки
X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    train_size=0.8,
                                                    test_size=0.2,
                                                    random_state=0,
                                                    shuffle=True)

# Создание и обучение модели
model_RFR = RandomForestRegressor(bootstrap=False, criterion='friedman_mse', max_depth=7,
                      max_features='log2', n_estimators=51)
model_RFR.fit(X_train, Y_train)

y4_pred = model_RFR.predict(X_test)

# Получение сведений с помощью обученной модели
prediction = model_RFR.predict(df_2)

# Формирование основной панели
# Вывод результата
st.write("""
# Результат прогнозирования параметров сварного шва
""")

st.subheader('Введенные данные:')
st.write(df_2)

st.subheader('Значение ширины шва')
st.write(prediction[:,[0]])

st.subheader('Значение глубины шва')
st.write(prediction[:,[1]])

# Вывод третьего подзаголовка для раздела, в котором будет находиться результат
st.subheader('Точность прогноза 95%')
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#streamlit containers, rows, columns
header = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()

#to make a requirements.txt file for deployment, install pipreqs
#then do pipreqs PATHTOPROJECT. so piprequs ./

st.markdown(
    """
<style>
.main {
background-color: #cccccc;
}
</style>
    """
)

#caching this func
@st.cache_data
def get_data():
    taxi_data = pd.read_csv('data/table.csv')
    return taxi_data


#write something in the container
with header:
    st.title('title')
    st.header('header')
    st.text('text')

with dataset:
    st.header('data part')
    taxi_data = get_data()
    st.subheader('location id dist')
    pulocation_dist = pd.DataFrame(taxi_data['PULocationID'].value_counts().head(50))
    st.bar_chart(pulocation_dist)

with features:
    #for rich text mardown
    st.markdown('* **first feature**: yup')

with modelTraining:
    sel_col, disp_col = st.columns(2)

    max_depth = sel_col.slider('slider title', min_value=10, max_value=100, value=20, step=10) #value is default value slider is set to
    n_estimators = sel_col.selectbox('select title', options=[100, 200, 'No limit'], index=0) #index is default val

    # if n_est == 'No limit':


    input_feature = sel_col.text_input('input title', 'PULocationID') #second value is default
    regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
    x = taxi_data[[input_feature]]
    y = taxi_data[['trip_distance']]
    regr.fit(x, y)
    prediction = regr.predict(y)

    disp_col.subheader('subhead')
    disp_col.write(mean_squared_error(y, prediction))
import streamlit as st
import time
import pandas as pd
import numpy as np
import json
import pickle

st.set_page_config(
    page_title="Shopee", page_icon="random",
    initial_sidebar_state="expanded", layout="wide"
)
pd.set_option('display.max_columns', None)


def predict(data):
    if data['cluster'].values == 0:
        with open("model/rf_0.pkl", "rb") as f:
            model = pickle.load(f)
    elif data['cluster'].values == 1:
        with open("model/rf_1.pkl", "rb") as f:
            model = pickle.load(f)
    prediction = model.predict(data.drop(columns=['cluster']))
    return prediction


def convert_ctime_to_two_time(time_start):
    nowaday = time.time()
    interval = nowaday-time_start
    time_tmp = interval - 2529000
    if time_tmp <= 0:
        return 0, int(interval)
    interval -= 2529000
    k = interval // 15768000
    odd = interval % 15768000
    time_summer = 0
    if k % 2 == 0:
        time_summer = (k//2)*15768000 + odd
    else:
        time_summer = (k//2 + 1) * 15768000
    return int(time_summer), int(interval - time_summer + 2529000)


def create_category(category, names):
    category_np = np.zeros(len(names), dtype=int)
    for i, name in enumerate(names):
        for k, cat in enumerate(category):
            l_cat = cat.split('/')
            for j in l_cat:
                if j in name:
                    category_np[i] = k+1
                    break
    return category_np


def data_preprocessing(data):
    with open('config.json') as f:
        config = json.load(f)

    data['gender'] = data['product_name'].apply(lambda x: 2 if ('nam' in x.lower()) and (
        'nữ' in x.lower()) else 1 if 'nữ' in x.lower() else 0 if 'nam' in x.lower() else 2)

    genders = [0, 1, 2]

    for gender in genders:
        data[gender] = data['gender'].apply(lambda x: 1 if gender == x else 0)
    data.rename(columns={0: 'Male', 1: 'Female', 2: 'Unisex'}, inplace=True)

    data['brand'] = data['brand'].replace(to_replace=[
                                          'No brand', 'NoBrand', '0', 'None'], value='No Brand').fillna(value='No Brand')
    data['brand'] = data['brand'].apply(lambda x: 0 if x == 'No Brand' else 1)

    category = ['None', 'kimono', 'kaki', 'jean', 'bomber', 'cardigan', 'sơ mi', 'gió/dù/jacket', 'blazer', 'croptop/gile',
                'thể thao', 'áo khoác phao/áo phao', 'nỉ/hoodie', 'denim', 'dạ/lông', 'len', 'da', 'chống nắng', 'thun', 'lửng']

    data['category'] = create_category(
        category[1:], data.loc[:, 'product_name'])
    data['category'] = data['category'].apply(lambda x: category[x])
    for cat in category:
        data[cat] = data['category'].apply(lambda x: 1 if x == cat else 0)

    data['two_time'] = data['product_ctime'].apply(
        lambda x: convert_ctime_to_two_time(x))
    data['time_summer'] = data['two_time'].apply(lambda x: x[0])
    data['time_winter'] = data['two_time'].apply(lambda x: x[1])
    data.drop(columns=['two_time'], inplace=True)

    data['shop_location_encoded'] = data['shop_location'].apply(
        lambda x: config['shop_encoded'][x])

    col_to_log = ['price', 'follower_count', 'products', 'rating_good',
                  'rating_bad', 'rating_normal', 'shop_location_encoded']
    for col in col_to_log:
        if data[col].values != 0:
            data[col] = np.log(data[col])

    with open("model/k_model.pkl", "rb") as f:
        k_model = pickle.load(f)
    cluster = k_model.predict(data[['follower_count', 'response_rate']])
    data['cluster'] = cluster

    data.drop(columns=['shop_location', 'product_ctime',
              'category'], inplace=True)
    cols = ['stock', 'brand', 'price', 'follower_count',
            'products', 'rating_normal', 'rating_bad', 'rating_good', 'rating_star',
            'response_time', 'response_rate', 'Unisex', 'Female', 'Male', 'color',
            'size'] + category + ['time_summer', 'time_winter', 'shop_location_encoded', 'cluster']

    data = data[cols]
    return data


def make_df(data):
    data = pd.DataFrame([data])
    return data


with st.container():
    st.markdown('# Shopee Product Sales Prediction')
    with st.form(key='info'):
        with st.expander(label='Shop Information', expanded=False):
            follower_count = st.number_input(label="Follower", min_value=0)

            products = st.number_input(label="Products", min_value=1)

            rating_good = st.slider(
                label='Rating Good', min_value=1, max_value=5, value=5, step=1)

            rating_bad = st.slider(
                label='Rating Bad', min_value=1, max_value=5, value=1, step=1)

            rating_normal = st.slider(
                label='Rating Normal', min_value=1, max_value=5, value=3, step=1)

            location_options = ['Hà Nội', 'Nam Định', 'TP. Hồ Chí Minh', 'Hải Dương', 'Thái Bình', 'Quốc Tế', 'Bắc Ninh', 'Thanh Hóa', 'Bình Dương', 'Nghệ An', 'Long An', 'Hải Phòng', 'Lâm Đồng', 'Đà Nẵng', 'Đắk Nông', 'Phú Yên', 'Quảng Ninh', 'Đắk Lắk', 'Hưng Yên',
                                'Quảng Ngãi', 'Hà Nam', 'Thái Nguyên', 'Cần Thơ', 'Kiên Giang', 'Lào Cai', 'Sơn La', 'Thừa Thiên Huế', 'Đồng Nai', 'Điện Biên', 'Gia Lai', 'Tây Ninh', 'Bình Thuận', 'Ninh Bình', 'Bà Rịa - Vũng Tàu', 'Lạng Sơn', 'An Giang', 'Bắc Giang', 'Tiền Giang']

            shop_location = st.selectbox(
                label='Shop Location', options=location_options)

            response_rate = st.slider(
                label="Response Rate", min_value=1, max_value=100, value=10, step=1)

            response_time = st.number_input(label="Response Time", min_value=1)

        with st.expander(label='Product Information', expanded=False):
            product_name = st.text_input(label='Product Name')

            brand = st.text_input(
                label='Brand', value='No Brand', placeholder='No Brand')

            price = st.number_input(label="Price", min_value=10000, step=10000)

            stock = st.number_input(label="Stock", min_value=0, step=1)

            color = st.number_input(label="Color", min_value=1, step=1)

            size = st.number_input(label="Size", min_value=1, step=1)

            time_interval = st.select_slider(label='Time interval', options=[
                                             '1 month', '2 month', '3 month'])
            ctime = int(time.time()) - int(time_interval.split()[0])

            rating_star = st.number_input(
                label="Rating Star", min_value=1, max_value=5)
        submit_button = st.form_submit_button('Submit')

        if submit_button and product_name != '':
            data = {
                'follower_count': follower_count,
                'products': products,
                'rating_normal': rating_normal,
                'rating_good': rating_good,
                'rating_bad': rating_bad,
                'shop_location': shop_location,
                'response_rate': response_rate,
                'product_name': product_name,
                'rating_star': rating_star,
                'response_time': response_time,
                'stock': stock,
                'brand': brand,
                'product_ctime': ctime,
                'price': price,
                'color': color,
                'size': size,
            }
            data = make_df(data)
            data = data_preprocessing(data)
            result = predict(data)
            st.write(
                f'Sales of product {product_name} in the next {" ".join(time_interval.split("_"))} will be approximately {int(result)}')
        elif submit_button and product_name == '':
            st.write('Please provide a product name')

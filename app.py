# app.py
import streamlit as st
import pandas as pd
import pickle

# Uygulama başlığı
st.title('Amazon Ürün Değerlendirme Sistemi')

# train.csv dosyasını yükle
@st.cache
def load_data():
    data = pd.read_csv('train.csv')
    return data

# amazon.pkl modelini yükle
@st.cache(allow_output_mutation=True)
def load_model():
    with open('amazon.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Verileri yükleyelim
data = load_data()

# Arayüz - Kullanıcı ve Ürün ID seçimi
st.sidebar.header("Kullanıcı ve Ürün Seçimi")

# Kullanıcı ID ve Ürün ID seçici
user_id = st.sidebar.selectbox('Kullanıcı ID Seçiniz:', data['userId'].unique())
product_id = st.sidebar.selectbox('Ürün ID Seçiniz:', data['productId'].unique())

# Kullanıcının daha önce verdiği puanlar
user_ratings = data[data['userId'] == user_id]

st.subheader(f"Kullanıcı {user_id} Tarafından Verilen Puanlar")
st.dataframe(user_ratings)

# Tahmin yapmak için model yükleyelim
model = load_model()

# Ürün puanı tahmini (örnek model kullanımı)
if st.button('Puan Tahmini Yap'):
    # Modelin input formatına göre değiştirin
    rating_prediction = model.predict([[user_id, product_id]])
    st.write(f"Tahmini Puan: {rating_prediction[0]:.2f}")

# train.csv'deki ilk 5 kaydı göster
st.subheader("İlk 5 Kayıt")
st.write(data.head())

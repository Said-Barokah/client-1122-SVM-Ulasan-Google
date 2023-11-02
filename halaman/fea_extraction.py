import streamlit as st
import pandas as pd
import time


def app():
    column_data = pd.read_csv('data/meta/column_data.csv')
    column = column_data['column'][0]
    from sklearn.feature_extraction.text import TfidfVectorizer

    from sklearn.preprocessing import normalize
    import numpy as np
    data = pd.read_csv('data/main_data.csv')
    
    st.subheader('Text')
    st.write(data)
    if st.button('Ubah data ke TF-IDF Vektor'):
        with st.spinner('tunggu sebentar ...'):
            time.sleep(2)
            st.subheader('TF-IDF')
            vectorizer = TfidfVectorizer()
            TF_IDF_vector = vectorizer.fit_transform(data[column])
            df_TF_IDF_vector = pd.DataFrame(TF_IDF_vector.toarray(),columns=vectorizer.get_feature_names_out())
            st.write(df_TF_IDF_vector)
            pd.DataFrame(df_TF_IDF_vector).to_csv('data/df_TF_IDF_vector.csv',index=False)
            st.success("Data Berhasil diubah menjadi TF-IDF")
        




    


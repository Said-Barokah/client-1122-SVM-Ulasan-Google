import streamlit as st
import pandas as pd
import time
import os
def app():
    st.title('APLIKASI SENTIMEN ANALASIS METODE SVM')
    if (os.path.exists("data/data_master.csv")):
         st.text('Data master')
         df = pd.read_csv('data/data_master.csv')
         st.write(df)
    if (os.path.exists("data/meta/column_data.csv")):
         column = pd.read_csv('data/meta/column_data.csv')
         feature = column['column'][0]
         label  = column['label'][0]

         
    data = st.file_uploader("upload data berformat csv (untuk mengubah data master)", type=['csv'])
    if data is not None:
            dataframe = pd.read_csv(data)
            st.text(f'kolom yang digunakan dalam klasifikasi adalah "{feature}" dan kolom yang akan dijadikan label adalah "{label}"')
            dataframe.columns = dataframe.columns.str.replace("^\s+|\s+$","",regex=True)
            
            # st.write(dataframe)
            col1, col2 = st.columns(2)
            with col1 :
                column = st.selectbox("Pilih Kolom yang akan di proses :",
                list(dataframe.columns))
            with col2 :
                label = st.selectbox("Pilih Kolom yang akan dijadikan label atau class :",
                list(dataframe.columns))

            column_data = pd.DataFrame(data={'column': [column], 'label': [label]})
            column_data.to_csv('data/meta/column_data.csv',index=False)
            if st.button('simpan data') :
                dataframe[label] = dataframe[label].str.replace("^\s+|\s+$","",regex=True)
                dataframe.to_csv('data/data_master.csv',index=False)
                column_data = pd.read_csv('data/meta/column_data.csv')
                if os.path.exists("data/data_branch.csv"):
                    os.remove("data/data_branch.csv")
                dataframe = dataframe[[column_data['column'][0],column_data['label'][0]]]
                dataframe.to_csv('data/main_data.csv',index=False)
                with st.spinner('tunggu sebentar ...'):
                    time.sleep(1)
                st.success('data berhasil disimpan')
                st.info('column ' + column_data['column'][0] + ' akan diproses')
                st.info('column ' + column_data['label'][0] + ' akan dijadikan label')

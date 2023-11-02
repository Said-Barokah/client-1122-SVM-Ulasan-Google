import streamlit as st
import pandas as pd

def app() :
    from sklearn.model_selection import train_test_split
    data_master = pd.read_csv('data/data_master.csv')
    data = pd.read_csv('data/main_data.csv')
    df_TF_vector = pd.read_csv('data/df_TF_IDF_vector.csv')
    test_size = pd.read_csv('data/meta/test_size.csv')
    column_data = pd.read_csv('data/meta/column_data.csv')


    # # ## pembagian data test dengan data secara otomatis
    st.subheader('Klasifikasi SVM')
    
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.svm import SVC

    from mlxtend.plotting import plot_confusion_matrix
    X_train, X_test, y_train, y_test = train_test_split(df_TF_vector, data[column_data['label'][0]], test_size = test_size['test size'][0],random_state=1221)
    train_class = st.number_input("Masukkan jumlah percobaan klasifikasi SVM",min_value=1,value=3,key=99)
    list_c = []
    list_gamma = []
    key = 1
    for i in range(1,train_class+1):
        st.caption(f'Pada Percobaan {i}')
        c_trade_off = st.number_input(f"Masukkan nilai C",key=key,min_value=1.0,value=1.0)
        gamma_var = st.number_input(f"Masukkan nilai λ",key=key+1,min_value=0.0,value=0.1)
        key=key+2
        list_c.append(c_trade_off)
        list_gamma.append(gamma_var)
    # if(st.button("Mulai Klasifikasi")):
    expander_1 = st.expander("Hasil dari Klasifikasi SVM")
    for c_trade_off,gamma_var in zip(list_c,list_gamma):
            model_svm = SVC(kernel='linear',C=c_trade_off, gamma=gamma_var)
            svm_train = model_svm.fit(X_train, y_train)
            y_pred = svm_train.predict(X_test)
            accuracy = accuracy_score(y_test,y_pred)
            c_matrik = confusion_matrix(y_test,y_pred)
            column = expander_1.multiselect('Pilih kolom yang akan di digunakan, kecuali kolom label',
                        list(data_master.columns), key=120+list_c.index(c_trade_off)
                        )
            data_pred = pd.DataFrame({'Predsi Kelas':y_pred,'Kelas Sesunggunya':y_test})
            data_pred = (data_pred.join(data_master[column]))
            expander_1.write(data_pred)
            expander_1.write(f' ketika C= {c_trade_off} dan λ = {gamma_var} akurasi yang didapatkan adalah {round(accuracy,3)}')
            fig, ax = plot_confusion_matrix(conf_mat=c_matrik,class_names=svm_train.classes_)
            ax.set_title('Confusion Matrik')
            ax.set_ylabel('Actual Label')
            ax.set_xlabel('Predicted Label')
            expander_1.pyplot(fig)
    


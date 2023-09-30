from keras.models import load_model
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
import numpy as np
import streamlit as st
from PIL import Image

model = load_model('models\model.h5')

st.header("Определитель кошкодевочек и песиков в их естественной среде обитания")

img_data = st.file_uploader(label='Загрузите мяу-мяу или гав-гав', type=['jpg'])

if img_data is not None:
    uploaded_img = Image.open(img_data)
    st.image(uploaded_img, caption='Давай посмотрим, что за хуйню ты мне прислал...')

    bytes_data = img_data.getvalue()
    with open("temp.jpg", 'wb') as f:
        f.write(bytes_data)

    img = image.load_img('temp.jpg', target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    
    prediction = model.predict(preprocessed_img)

    if (prediction[0][0] > prediction[0][1]):
        st.header("С ВЕРОЯТНОСТЬЮ " + str(prediction[0][0]) + " БЛЯ БУДУ ЭТО КОШКОДЕВКА ЭТО КОШКОДЕВОЧКА РУКИ НА СТОЛ")
    else:
        st.header("он голодный пес он готов порвать... "  + str(prediction[0][1]) + " раз он будет твою мать ебать...")
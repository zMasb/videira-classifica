import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px

@st.cache_resource
def carrega_modelo():
    # https://drive.google.com/file/d/1-3VSWJ_vO6HNDK9uN0dWKygdeCL3fwrA/view?usp=drive_link

    url = 'https://drive.google.com/uc?id=1-3VSWJ_vO6HNDK9uN0dWKygdeCL3fwrA'

    gdown.download(url, 'modelo_quantizado16bits.tflite')
    interpreter = tf.lite.Interpreter(modelo_path = 'modelo_quantizado16bits.tflite')
    interpreter.allocate_tensors()

    return interpreter

def carrega_imagem():
    uploaded_file = st.file_uploader('Arraste e solte uma imagem ou clique aqui para selecionar uma', type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))

        st.image(image)
        st.succes('Imagem foi carregada com sucesso.')

        image = np.array(image, dtype=np.float32)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        return image

def previsao(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    classes = ['BlackMeasles', 'BlackRot', 'HealthyGrapes', 'LeafBlight']

    df = pd.Dataframe()
    df['classes'] = classes
    df['probabilidades (%)'] = 100*output_data[0]

    fig = px.bar(df, y = 'classes', x= 'probabilidades (%)', orientation = 'h', text = 'probabilidades (%)',
                 title = 'Probabailidade de Classe de Doenças em Uvas')

    st.plotly_chart(fig)

def main():

    st.set_page_config(
        page_title = "Classfica Folhas de videira",
        page_icon = "🍇",
    )

    st.write("# Classifica Folhas de Videira")

    # Carregar modelo

    interpreter = carrega_modelo()

    # Carregar imagem

    image = carrega_imagem()

    # Classificar

    if image is not None:

        previsao(interpreter, image)

if __name__ == "__main__":
    main()
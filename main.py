# Importando bibliotecas
import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image
import pandas as pd

from src.codes.util import classify, set_background, createModel

# Colocando uma imagem de fundo
set_background('https://img.rawpixel.com/s3fs-private/rawpixel_images/website_content/rm373batch15-bg-11-kqaiv1bm.jpg?w=800&dpr=1&fit=default&crop=default&q=65&vib=3&con=3&usm=15&bg=F4F4F3&ixlib=js-2.2.1&s=1f21f7cdacb8138b3890ac267cf16c84', 360)

# Definindo um título
st.title("Classificador de Raio X")

# Texto
st.subheader("Descrição do Projeto")
st.write("Este aplicativo Streamlit é uma demonstração de um sistema de classificação de pneumonia em imagens de raio X do tórax. A pneumonia é uma inflamação dos pulmões frequentemente causada por infecções bacterianas ou virais. O diagnóstico preciso é crucial para um tratamento eficaz. Neste contexto, a análise de imagens de raio X tem se mostrado uma ferramenta promissora.")

st.write("O sistema é alimentado por um modelo de aprendizado de máquina previamente treinado utilizando a arquitetura de redes neurais convolucionais (CNN). A rede neural é treinada em um extenso conjunto de dados de raio X de tórax, contendo amostras saudáveis e casos de pneumonia. A arquitetura da CNN é capaz de aprender características distintas nas imagens que são indicativas de padrões associados à pneumonia.")

st.write("Ao fornecer uma imagem de raio X através da interface do usuário, o sistema processa a imagem utilizando a CNN treinada para realizar a classificação. O resultado é uma previsão de probabilidade que a imagem contenha sinais de pneumonia. Esta abordagem tem o potencial de auxiliar profissionais de saúde na triagem inicial de casos suspeitos, permitindo a priorização de casos mais urgentes.")

st.write("Esta demonstração destaca a integração de técnicas de aprendizado de máquina com interface de usuário amigável, permitindo a aplicação prática de algoritmos de análise de imagem em um cenário clínico. É importante notar que a precisão do modelo depende da qualidade dos dados de treinamento e da capacidade de generalização para novos casos.")

st.divider()

# Definindo modelo

model_name = st.selectbox("Selecione o Modelo:", ["Nenhum", "CNN_pneumonia"])
model_flag = 0
st.caption("Escolha diferentes modelos para testa-los!")

# Carregar arquivos de classes
with open('./model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

if model_name != "Nenhum":
    model = load_model(f"./model/{model_name}.h5")
    try:
        st.write("Modelo Importado com sucesso!")
        model_flag = 1
    except:
        st.write("Problemas com a importação do modelo.")

    st.write(model.summary())

# Subindo um arquivo
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# Exibindo, ajustando e classificando 
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    class_name, conf_score = classify(image, model, class_names)

    # write classification
    st.write(f"## Classe Prevista: {class_name}")
    st.write(f"### Score: {int(conf_score * 1000) / 10}%")

st.divider()

st.title("Criando e Exibindo detalhes de um modelo: ")

col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Nome do Modelo:")
    shape = st.number_input("Dimensão da Imagem", min_value=200, max_value=1000, value=250)

with col2:
    num_classes = st.number_input("Classes: ", min_value=2, max_value=100, value=2)

include_top = st.checkbox("Include Top?")

layers = st.multiselect("Selecione as camadas para adicionar", ["Dropout", "Flatten", "Dense"])

create_model = st.button("Criar Modelo vgg16")

if create_model:
    model_vgg = createModel(shape, num_classes, include_top, layers)
    model_vgg.summary(print_fn=lambda x: st.text(x))
    
    comentario = st.text_area("Comentários sobre o Modelo")

    # Podemos inserir um pipeline de pré-processamento, treinamento, avaliação e salvar o modelo

st.divider()

# Manipulando dados
st.title("Manipulando Dados")

# Lendo dataframe
dataframe = pd.read_csv("./data/final/mtsamples.csv")
dataframe = dataframe.drop(["Unnamed: 0"], 1)

# Exibir Dataframe
st.write("Exibindo um Dataframe")
st.dataframe(dataframe)

# Editando dados
st.write("Modo de Edição de Dados")
dataframe_edited = st.data_editor(dataframe)

save_df = st.button("Salvar Dataframe")

if save_df:
    dataframe_edited.to_csv("./data/final/DataframeNew.csv")

st.divider()

st.title("Outras Features e ideias de uso do Streamlit: ")

st.write(f"Visualização e Análise de Dados: https://archesz-analyseherbert-app-c5rc0l.streamlit.app")
st.write(f"Documentações: https://cheat-sheet.streamlit.app")
st.write(f"Integração com API (GPT): https://ai-talks.streamlit.app")
st.write(f"Processamento de Imagens: https://bgremoval.streamlit.app")
st.write(f"Realizar simulações: https://gw-quickview.streamlit.app")
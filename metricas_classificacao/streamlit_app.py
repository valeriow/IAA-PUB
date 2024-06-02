import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.title("Métricas Classificação Binária")

# Entrada da matriz de confusão com sliders
st.markdown("**Matriz de Confusão:**")
col1, col2 = st.columns(2)
with col1:
    VP = st.slider("Verdadeiros Positivos (VP)", 0, 100, 50)
    FN = st.slider("Falsos Negativos (FN)", 0, 100, 20)
with col2:
    FP = st.slider("Falsos Positivos (FP)", 0, 100, 10)
    VN = st.slider("Verdadeiros Negativos (VN)", 0, 100, 80)

# Cálculo das métricas
try:
    # Calcula o total para normalizar a matriz
    total = VP + FN + FP + VN

    # Normaliza os valores da matriz de confusão para que somem 1
    VP_norm = VP / total
    FN_norm = FN / total
    FP_norm = FP / total
    VN_norm = VN / total


    acuracia = (VP + VN) / (VP + VN + FP + FN)
    precisao = VP / (VP + FP)
    recall = VP / (VP + FN)
    f1_score = 2 * (precisao * recall) / (precisao + recall)

    acuracia_str = f"{acuracia:.2}"
    precisao_str = f"{precisao:.2}"
    recall_str = f"{recall:.2}"
    f1_score_str = f"{f1_score:.2}"
    st.latex(fr"""
             \begin{{aligned}}
             \text{{Acurácia}} &=\frac{{VP + VN}}{{VP + VN + FP + FN}} &=\frac{{ {VP} + {VN} }}{{ {VP} + {VN} + {FP} + {FN} }}&={acuracia_str}
             \end{{aligned}}
             """)
    st.latex(fr"""
             \begin{{aligned}}
             \text{{Precisão}} &=\frac{{VP}}{{VP + FP}} &=\frac{{ {VP} }}{{ {VP} + {FP} }} &={precisao_str} 
             \end{{aligned}}
             """)
    st.latex(fr"""
             \begin{{aligned}}
             \text{{Recall}} &=\frac{{VP}}{{VP + FN}} &= \frac{{ {VP} }}{{ {VP} + {FN} }} &={recall_str} 
             \end{{aligned}}
             """)
    st.latex(fr"""
             \begin{{aligned}}
             \text{{F1-Score}} &=2 \cdot \frac{{\text{{Precisão}} \cdot \text{{Recall}}}}{{\text{{Precisão}} + \text{{Recall}}}} &=2 \cdot \frac{{\text{{ {precisao_str} }} \cdot \text{{ {recall_str} }}}}{{\text{{ {precisao_str} }} + \text{{ {recall_str} }}}} &={f1_score_str} 
             \end{{aligned}}
             """)

    st.subheader("Métricas:")
    data = {'Métrica': ['Acurácia', 'Precisão', 'Recall', 'F1-Score'],
            'Valor': [acuracia, precisao, recall, f1_score]}
    df = pd.DataFrame(data)
    st.dataframe(df.style.format({'Valor': '{:.2%}'}))

    st.subheader("Matriz de Confusão:")
    confusion_matrix = np.array([[VP, FN], [FP, VN]])
    st.write(confusion_matrix)

    st.subheader("Matriz de Confusão Normalizada:")
    confusion_matrix = np.array([[VP_norm, FN_norm], [FP_norm, VN_norm]])
    st.write(confusion_matrix)

except ZeroDivisionError:
    st.warning("Divisão por zero. Ajuste os valores da matriz de confusão.")

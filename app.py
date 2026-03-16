# -*- coding: utf-8 -*-
import streamlit as st
import pdfplumber
import re
import nltk
from collections import Counter
from itertools import islice
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import tempfile
import os

# Configuração da página
st.set_page_config(
    page_title="Analisador de Texto - Tracoma",
    page_icon="📊",
    layout="wide"
)

# Título principal
st.title("Análise Estatística de Texto")
st.markdown("---")

# Download dos recursos do NLTK
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

download_nltk_data()

# Funções de processamento
@st.cache_data
def processar_texto(texto):
    """Processa o texto seguindo os passos do código original"""
    
    # 1. Lowercase
    texto = texto.lower()
    
    # 2. Remove números
    texto = re.sub(r'\d', '', texto)
    
    # 3. Remove pontuações e caracteres especiais
    texto = re.sub(r'[^\w\s]', '', texto)
    
    # 4. Normaliza espaços
    texto = re.sub(r'\s+', ' ', texto)
    
    # 5. Tokenização
    tokens = re.findall(r'\w+', texto)
    
    # 6. Remove stopwords e palavras curtas
    stopwords = nltk.corpus.stopwords.words('portuguese')
    tokens_limpos = [item for item in tokens 
                     if (item not in stopwords) and (len(item) > 2)]
    
    return tokens_limpos, texto

# Função para treemap
@st.cache_data
def treemap_palavras(palavras_frequentes):
    """Visualização hierárquica das palavras mais frequentes"""
    
    if not palavras_frequentes:
        return None
    
    # Preparar dados
    palavras = [p[0] for p in palavras_frequentes[:30]]
    frequencias = [p[1] for p in palavras_frequentes[:30]]
    
    # Criar categorias baseadas na primeira letra
    categorias = [p[0][0].upper() if p[0] else '?' for p in palavras_frequentes[:30]]
    
    df = pd.DataFrame({
        'Palavra': palavras,
        'Frequência': frequencias,
        'Categoria': categorias
    })
    
    fig = px.treemap(
        df,
        path=['Categoria', 'Palavra'],
        values='Frequência',
        color='Frequência',
        color_continuous_scale='Viridis',
        title='Treemap de Frequência de Palavras'
    )
    
    fig.update_layout(
        height=500,
        title_font_size=16
    )
    
    return fig

@st.cache_data
def gerar_graficos(tokens_limpos):
    """Gera todos os gráficos baseado nos tokens processados"""
    
    # Dicionário para armazenar os gráficos
    graficos = {}
    
    # 1. Gráfico de palavras mais frequentes
    palavras_frequentes = Counter(tokens_limpos).most_common(20)
    words_tokens = [p[0] for p in palavras_frequentes]
    freq_tokens = [p[1] for p in palavras_frequentes]
    
    fig_palavras = go.Figure(go.Bar(
        x=words_tokens,
        y=freq_tokens, 
        text=freq_tokens, 
        textposition='outside',
        marker_color='rgb(55, 83, 109)'
    ))
    fig_palavras.update_layout(
        title_text='20 palavras mais utilizadas no texto',
        xaxis_title='Palavras',
        yaxis_title='Frequência',
        height=500,
        xaxis_tickangle=-45,
        showlegend=False
    )
    graficos['palavras'] = fig_palavras
    
    # 2. Gráfico de bigramas
    if len(tokens_limpos) >= 2:
        bigrams = [' '.join(b) for b in zip(tokens_limpos, islice(tokens_limpos, 1, None))]
        bigramas_frequentes = Counter(bigrams).most_common(20)
        
        if bigramas_frequentes:
            words_bigrams = [b[0] for b in bigramas_frequentes]
            freq_bigrams = [b[1] for b in bigramas_frequentes]
            
            fig_bigramas = go.Figure(go.Bar(
                x=words_bigrams,
                y=freq_bigrams, 
                text=freq_bigrams, 
                textposition='outside',
                marker_color='rgb(26, 118, 255)'
            ))
            fig_bigramas.update_layout(
                title_text='20 bigramas mais frequentes no texto',
                xaxis_title='Bigramas',
                yaxis_title='Frequência',
                height=500,
                xaxis_tickangle=-45,
                showlegend=False
            )
            graficos['bigramas'] = fig_bigramas
    
    # 3. Gráfico de trigramas
    if len(tokens_limpos) >= 3:
        trigramas = []
        for i in range(len(tokens_limpos) - 2):
            trigrama = (tokens_limpos[i], tokens_limpos[i+1], tokens_limpos[i+2])
            trigramas.append(' '.join(trigrama))
        
        trigramas_frequentes = Counter(trigramas).most_common(20)
        
        if trigramas_frequentes:
            words_trigramas = [t[0] for t in trigramas_frequentes]
            freq_trigramas = [t[1] for t in trigramas_frequentes]
            
            fig_trigramas = go.Figure(go.Bar(
                x=words_trigramas,
                y=freq_trigramas, 
                text=freq_trigramas, 
                textposition='outside',
                marker_color='rgb(50, 171, 96)'
            ))
            fig_trigramas.update_layout(
                title_text='20 trigramas mais frequentes no texto',
                xaxis_title='Trigramas',
                yaxis_title='Frequência',
                height=600,
                xaxis_tickangle=-45,
                showlegend=False
            )
            graficos['trigramas'] = fig_trigramas
    
    # 4. Treemap
    if palavras_frequentes:
        try:
            fig_treemap = treemap_palavras(palavras_frequentes)
            if fig_treemap:
                graficos['treemap'] = fig_treemap
        except Exception as e:
            st.warning(f"Não foi possível gerar o treemap: {e}")
    
    return graficos, palavras_frequentes

# Sidebar com informações
with st.sidebar:
    st.header("ℹ️ Sobre")
    st.markdown("""
    
    Esta aplicação processa textos e gera:
    - **Palavras mais frequentes**
    - **Bigramas mais comuns**
    - **Trigramas mais comuns**
    - **Treemap de frequência**
    
    **Como usar:**
    1. Digite um texto ou faça upload de PDF
    2. Clique em "Analisar Texto"
    3. Visualize os gráficos gerados
    """)
    
    st.markdown("---")

# Área principal - Entrada de dados
st.subheader("Entrada de Texto")

# Abas para diferentes formas de entrada
tab1, tab2 = st.tabs(["Digitar Texto", "Upload de PDF"])

with tab1:
    texto_input = st.text_area(
        "Digite ou cole seu texto aqui:",
        height=200,
        placeholder="Cole seu texto para análise..."
    )
    
    if st.button("Analisar Texto Digitado", type="primary", key="btn_digitado"):
        if texto_input and len(texto_input.strip()) > 10:
            st.session_state['texto_analise'] = texto_input
            st.session_state['fonte'] = 'digitado'
            st.rerun()
        else:
            st.warning("Por favor, digite um texto com pelo menos 10 caracteres.")

with tab2:
    arquivo = st.file_uploader(
        "Carregue um arquivo PDF",
        type=['pdf'],
        help="Selecione um arquivo PDF para análise"
    )
    
    if arquivo and st.button("Analisar PDF", type="primary", key="btn_pdf"):
        try:
            with st.spinner('Processando PDF...'):
                # Salvar arquivo temporariamente
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(arquivo.getvalue())
                    tmp_path = tmp_file.name
                
                # Extrair texto do PDF
                texto_pdf = ''
                with pdfplumber.open(tmp_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            texto_pdf += page_text
                
                # Limpar arquivo temporário
                os.unlink(tmp_path)
                
                if texto_pdf:
                    st.session_state['texto_analise'] = texto_pdf
                    st.session_state['fonte'] = 'pdf'
                    st.rerun()
                else:
                    st.error("Não foi possível extrair texto do PDF. O arquivo pode estar vazio ou ser apenas imagens.")
                    
        except Exception as e:
            st.error(f"Erro ao processar o PDF: {e}")

# Área de resultados
if 'texto_analise' in st.session_state:
    st.markdown("---")
    st.header("Resultados da Análise")
    
    texto = st.session_state['texto_analise']
    
    # Mostrar preview do texto
    with st.expander("Visualizar texto processado"):
        st.write(texto[:500] + "..." if len(texto) > 500 else texto)
    
    # Processar texto e gerar gráficos
    with st.spinner('Processando texto e gerando gráficos...'):
        tokens_limpos, texto_limpo = processar_texto(texto)
        graficos, palavras_frequentes = gerar_graficos(tokens_limpos)
    
    # Estatísticas rápidas
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de palavras", len(tokens_limpos))
    with col2:
        st.metric("Palavras únicas", len(set(tokens_limpos)))
    with col3:
        if len(tokens_limpos) >= 2:
            bigramas_unicos = len(set([' '.join(b) for b in zip(tokens_limpos, islice(tokens_limpos, 1, None))]))
            st.metric("Bigramas únicos", bigramas_unicos)
        else:
            st.metric("Bigramas únicos", 0)
    with col4:
        if len(tokens_limpos) >= 3:
            trigramas_unicos = len(set([' '.join(t) for t in zip(tokens_limpos, islice(tokens_limpos, 1, None), islice(tokens_limpos, 2, None))]))
            st.metric("Trigramas únicos", trigramas_unicos)
        else:
            st.metric("Trigramas únicos", 0)
    
    # Exibir tabela de palavras mais frequentes
    with st.expander("Ver tabela de palavras mais frequentes"):
        df_palavras = pd.DataFrame(palavras_frequentes, columns=['Palavra', 'Frequência'])
        st.dataframe(df_palavras, use_container_width=True)
    
    st.markdown("---")
    
    # Exibir gráficos em abas
    if graficos:
        # Determinar quantas abas temos
        abas_disponiveis = []
        nomes_abas = []
        
        if 'palavras' in graficos:
            abas_disponiveis.append('palavras')
            nomes_abas.append("Palavras")
        if 'bigramas' in graficos:
            abas_disponiveis.append('bigramas')
            nomes_abas.append("Bigramas")
        if 'trigramas' in graficos:
            abas_disponiveis.append('trigramas')
            nomes_abas.append("Trigramas")
        if 'treemap' in graficos:
            abas_disponiveis.append('treemap')
            nomes_abas.append("Treemap")
        
        # Criar abas dinamicamente
        tabs = st.tabs(nomes_abas)
        
        for i, aba_nome in enumerate(abas_disponiveis):
            with tabs[i]:
                if aba_nome == 'palavras':
                    st.plotly_chart(graficos['palavras'], use_container_width=True)
                elif aba_nome == 'bigramas':
                    st.plotly_chart(graficos['bigramas'], use_container_width=True)
                elif aba_nome == 'trigramas':
                    st.plotly_chart(graficos['trigramas'], use_container_width=True)
                elif aba_nome == 'treemap':
                    st.plotly_chart(graficos['treemap'], use_container_width=True)
                    with st.expander("ℹ️ Sobre o treemap"):
                        st.markdown("""
                        **Treemap de Frequência de Palavras**
                        
                        - **Cores mais escuras**: Maior frequência
                        - **Tamanho dos retângulos**: Proporcional à frequência
                        - **Agrupamento por primeira letra**: Organização hierárquica
                        """)
    else:
        st.warning("Não foi possível gerar gráficos com o texto fornecido.")
    
    # Botão para nova análise
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Nova Análise", use_container_width=True):
            del st.session_state['texto_analise']
            del st.session_state['fonte']
            st.rerun()

# Rodapé
st.markdown("---")
st.markdown(
    "Análise de texto | Luca Soares | IESB - Campus Sul"
)
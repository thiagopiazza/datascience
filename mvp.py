#Date           :   02/07/2025
#Author         :   Thiago Bif Piazza
#Matrícula      :   XXXX
#Description    :   Script consruído para realizar a avaliação do MVP a disciplina de Análise de Dados e Boas Práticas do curso de Ciência de Dados e Analytics da PUC-Rio turma 2025/1

#CONTEXTUALIZAÇÃO
"""
Este script tem como objetivo realizar uma análise exploratória e pré-processamento dos dados de monitoramento do Alvo Caranguejo-Uçá do Componente Manguezal do 
Subprograma Marinho e Costeiro do Programa Nacional de Monitoramento da Biodiversidade (Monitora) coordenado pela Coordenação de Monitoramento da Biodiversidade (COMOB) 
do Instituto Chico Mendes de Conservação da Biodiversidade (ICMBio) - Ministério do Meio Ambiente (MMA) - Governo Federal (Brasil).
"""

#INTRODUÇÃO
"""
O Manguezal é um ecossistema costeiro, de transição entre o ambiente terrestre e marinho, característico de regiões tropicais e subtropicais, 
sujeito ao regime das marés. É um ambiente de alta produtividade biológica, abrigando uma rica biodiversidade e desempenhando funções ecológicas importantes como 
proteção costeira e sequestro de carbono, contribuindo assim para mitigar os efeitos das mudanças climáticas. O caranguejo uçá, Ucides cordatus (Linnaeus,
1763) é um crustáceo extrema importância dos manguezais, com destaque socioeconômico. É considerado um dos crustáceos endêmicos e de maior importância socioeconômica 
dos manguezais (Silva, 2014; Lima et al., 2018 apud Gonçalves et. al, 2022). Ele atua como fonte de renda, principalmente para comunidades litorâneas carentes, 
muitas vezes correspondendo à única forma de renda dessas comunidades em determinados períodos (Lima et al., 2018 apud Gonçalves et. al, 2022).
"""

#OBJETIVO
"""
Analisar a densidade de tocas e o diâmetro da galeria (tocas) do caranguejo uçá nas diferentes regiões do Brasil
"""

#PROBLEMA
"""
Descrição   :   Identificar regiões em território nacional com maior diâmetro de toca de caranguejo uçá
Tipo        :   Regressão. Dados as características de diâmetro de galeria (toca), o objetivo é identificar quais regiões brasileiras possuem maior diâmetro de toca de caranguejo
Hipóteses   :   H0 - Não há diferença no diâmetro de toca do caranguejo uçá nas diferentes regiões brasileiras
                H1 - Há regiões brasileiras com maior diâmetro de toca de caranguejo uçá
"""

#DADOS
"""
Os dados foram adquiridos através da funcionalidade de download do Sistema de Gestão de Dados do Programa Monitora (SISMonitora) e para fins de confidencialidade foram
mascarados da seguinte forma:

1. Regiões (N,NE,SE,S) = Nomeadas aleatoriamente de A a D (Ex: A, C, D, B)
5. Zonas de manguezal = Nomeadas aleatoriamente de A a C (Ex: A, C, B)
5. Número de tocas abertas (int) = Acrescidas do valor real + DP desvio padrão da variável no dataset (distribuído aleatoriamente)
6. Diâmetro da galeria (mm) = Acrescidas do valor real + DP desvio padrão da variável no dataset (distribuído aleatoriamente)
7. Nível médio de inundação (cm) = Acrescidas do valor real + DP desvio padrão da variável no dataset (distribuído aleatoriamente)
8. Densidade populacional (tocas/m2) = Acrescidas do valor real + DP desvio padrão da variável no dataset (distribuído aleatoriamente)

OBS: Cada unidade amostral (UA) possuí área de 5 m2.

Fonte       :   Programa Nacional de Monitoramento da Biodiversidade - Monitora, 2025.
Arquivo     :   caranguejouca.xlsx
Seleção     :   Os dados de uc, ea, ua, data, lat, lon, lat_lon e obs foram retirados pois não contribuem para solução do problema. 
                Os dados de densidade_tocas foram calculados a partir dos campos tocas_abertas_sum / area_m2.
                Os dados de classe_tamanho_tocas foram calculados a partir do arrendondamento do campo diametro_galeria_mm com precisão de 0 casas decimais.
                Os dados não apresentaram campos faltantes, por este motivo não foi necessário a exclusão ou substituição de instâncias em branco (NA).
                
Atributos   :   regiao                      (categorico)
                ano                         (categorico)
                zona                        (categorico)
                tocas_sum                   (inteirgo)
                area_m2                     (decimal)
                diametro_galeria_mm         (decimal)
                nivel_inundacao_medio_cm    (decimal)
                densidade_tocas             (decimal)
                classe_tamanho_tocas        (inteiro)

"""



#REFERÊNCIAS BIBLIOGRÁFICAS
"""
GONCALVES et al. Monitoramento do Caranguejo-uçá (Ucides cordatus) no Lagamar Paranaense. Biodiversidade Brasileira, 12(1): 143-158, 2022.;
"""

#MVP
#Importação
import os as os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#carregamento
os.getcwd() #Verificando o diretório atual
dataset = pd.read_excel("data/caranguejouca.xlsx") #importando dados da planilha excel (.xlsx)
dataset.head() # verificando importação atravpes da visualização da parte inicial dos dados

dataset['densidade_tocas'] = dataset['tocas_sum']/dataset['area_m2'] # calculando densidade de tocas por m2
dataset['classe_tamanho_tocas'] = round(dataset['diametro_galeria_mm'],0)


#ANÁLISE DE DADOS
#Estatísticas descritivas

dataset.info() # Verificando instâncias totais e seus tipos
# Temos 8 atributos com 63114 instâncias (observações) contendo dados do tipo int64, float64 e categoricos (object)
dataset.describe() # Verificando valores de tendência central no dataset
dataset.describe().loc["mean"] # Verificando média dos atributos numéricos

#Há uma média de 70 tocas por quadrado (área de amostragem)
#A área de amostragem é sempre 5m2
#O diâmetro médio das tocas é de 70.53mm
#O nível médio de inundação das marés é de 49.35cm

#plotando dados # gráfico de barras simples
#por região
plt.figure(figsize=(7, 5))
sns.countplot(x='regiao', data=dataset)
plt.title('Distribuição Tocas de Caranguejo por Região')
plt.xlabel('Região')
plt.ylabel('Contagem de Tocas')
plt.show()
#As regiões NE e S possuem número de instâncias similares, enquanto a região N possuí aproximadamente 4 vezes mais
dataset.groupby('regiao').describe() 
#As regiões NE(10.555) e S possuem número de instâncias similares (10.585), enquanto a região N possuí aproximadamente 4 vezes mais (41.974)

#por zona
plt.figure(figsize=(7, 5))
sns.countplot(x='zona', data=dataset)
plt.title('Distribuição Tocas de Caranguejo por Zona')
plt.xlabel('Zona')
plt.ylabel('Contagem de Tocas')
plt.show()
#As zona possuem número de instâncias similares
dataset.groupby('zona').describe() 
#A zona a(28.811) e b (34.303)


#plotando dados # histograma
plt.figure(figsize=(8, 6))
# Histograma do comprimento da sépala (um dos atributos)
sns.histplot(dataset['diametro_galeria_mm'], kde=True)
plt.title('Distribuição do Diâmetero das Tocas')
plt.xlabel('Diâmetro da toca (mm)')
plt.ylabel('Frequência')
plt.show()
#Diâmetro médio próximo a classe de 70mm e alguns possível outliers acima de 170mm


# Boxplot do diâmetro da toca por região
plt.figure(figsize=(10, 6))
sns.boxplot(x='regiao', y='diametro_galeria_mm', data=dataset)
plt.title('Diâmetro da Toca por Região')
plt.xlabel('Região')
plt.ylabel('Diâmetro da Toca (mm)')
plt.show()
#As regiões parecem ter mediana e amplitude entre quartis similares

# Boxplot da densidade de tocas por região
plt.figure(figsize=(10, 6))
sns.boxplot(x='regiao', y='densidade_tocas', data=dataset)
plt.title('Densidade de Tocas por Região')
plt.xlabel('Região')
plt.ylabel('Densidade (tocas/m2)')
plt.show()
#As regiões parecem ter mediana e amplitude entre quartis similares

# Boxplot do diâmetro da toca por zona
plt.figure(figsize=(10, 6))
sns.boxplot(x='zona', y='diametro_galeria_mm', data=dataset)
plt.title('Diâmetro da Toca por Zona')
plt.xlabel('Zona')
plt.ylabel('Diâmetro da Toca (mm)')
plt.show()
#Possivelmente as regiões não diferen quanto ao diametro da galeria

# Boxplot da densidade de tocas por zona
plt.figure(figsize=(10, 6))
sns.boxplot(x='zona', y='densidade_tocas', data=dataset)
plt.title('Densidade de Tocas por Zona')
plt.xlabel('Zona')
plt.ylabel('Densidade de Tocas (tocas/m2)')
plt.show()
#Possivelmente a região a possuí maior densidade de tocas


#Matriz de correlação
dataset.iloc[:,6:9].corr() #verificando a correlação entre os atributos numéricos (diametro_galeraia_mm,nivel_inundacao_medio_cm e densidade_tocas)
#densidade de tocas tem correlação negativa fraca com diâmetro_galeria -0.31

# mapa de calor das variáveis numéricas
plt.figure(figsize=(8, 6))
sns.heatmap(dataset.iloc[:,6:9].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlação das Características Numéricas do Dataset de Caranguejo Uçá')
plt.show()

#Tratamento de Valores Nulos
dataset.isnull()
#Não há dados faltantes


#PRÉ-PROCESSAMENTO DE DADOS
# Separar features (X) e target (y)
x = dataset.drop('regiao', axis=1)
y = dataset['regiao']

# Dividir os dados em conjuntos de treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

x_train.shape
x_test.shape
y_train.shape
y_test.shape

# Inicializar o MinMaxScaler
scaler_norm = MinMaxScaler()

# Aprende min e max APENAS de X_train
scaler_norm.fit(x_train)
x_train_normalized = scaler_norm.transform(x_train)
# Usa a média e o desvio padrão aprendidos de X_train
x_test_normalized = scaler_norm.transform(x_test)

# Exibir as primeiras linhas dos dados normalizados (como DataFrame para melhor visualização)
dataset_normalized = pd.DataFrame(x_train_normalized, columns=x_train.columns)
dataset_normalized.head()
#

# Visualização da distribuição após a normalização (exemplo para uma característica)
plt.figure(figsize=(8, 6))
sns.histplot(dataset_normalized['duametro_galeria_mm'], kde=True)
plt.title('Diâmetro das Tocas (Normalizado)')
plt.xlabel('Diâmetro da Toca (mm) Normalizado')
plt.ylabel('Frequência')
plt.show()

#Outras Transformações e Etapas de Pré-Processamento
#Incluiria o atributo de intensidade El Niño de 0 a 3 (0 - ausente, 1 - franco, 2 - moderado e 3 - forte) para tornar as predições mais robustas em relação a efeitos nas variáveis em anos com presença de El Niño e suas respectivas intensidades

#CONCLUSÃO
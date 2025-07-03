# FMF_Churn-PAD

## Boas-vindas!
Nosso projeto trata-se do desenvolvimento de um algoritimo de Machine Learning aplicado à área da educação, mais especificamente para predição de "Churn"/evasão de alunos no contexto de uma universidade EAD. 
Firmamos acordo com uma empresa de escala nacional deste segmento e conseguimos dados reais disponibilizados para prepararmos nossos datasets. Nossos objetivos, para mais do que desenvolvimento acadêmico, visam também resolver problemas do mundo real.

### Grupo
João Arthur, Cindy Stephánie, Júlia Albuja, André Koraleski, Lucas Nogueira e Sandro Xavier.


### Limpeza dos dados!

O passo a passo da limpeza dos dados realizados no arquivo .csv, tenham em mente de analisar cada ação realizada e opinarem mudanças relevantes no grupo. O link do [colab](https://colab.research.google.com/drive/1eoWDy48g5JCYWvaJrJmGMkxPT6pWQXM7?usp=sharing) de está liberado para acesso de indivíduos que estão utilizando contas vinculadas a UFG, lembrem-se de colocar o arquivo .csv que está no [drive](https://drive.google.com/drive/folders/1G6vIumLHe8rB-FUZhAe4ovnqerbAlr7X?usp=sharing) no path correto no código, lembrando novamente que apenas contas vinculadas a UFG terão permissão de acessar o link.

## Exploração dos Dados (Análise Exploratória)

Nesta etapa inicial, buscamos compreender, limpar e preparar o conjunto de dados para a modelagem. O trabalho foi realizado no Google Colab, utilizando principalmente a biblioteca Pandas.

**Conjunto de dados**  
- Arquivo: `Alunos_Atendimentos_Merged.csv`  
- Registros: 17 164  
- Colunas: 51 (dados de matrícula, demográficos, status acadêmico e atendimentos)

**As principais etapas**  
1. **Análise de valores ausentes**  
   - Comando: `df.isnull().sum()`  
   - Observação: elevada quantidade de nulos em colunas de atendimento (ex.: `Apoio Pedagógico`, `Financeiro`) e em `Data de nascimento`.

2. **Limpeza e tratamento**  
   - **Preenchimento de nulos**  
     - Colunas numéricas → substituídas por `-1`.  
     - Colunas categóricas → preenchidas com `"Não Informado"`.  
   - **Remoção de duplicatas**  
     - Eliminação de linhas repetidas para garantir registros únicos.  
   - **Conversão de tipos**  
     - Colunas de data (`DATAMATRICULA`, `ENCERRAMENTO_CONTRATO`, `Data de nascimento`) convertidas para `datetime`.

3. **Engenharia de features**  
   - Criação da variável alvo `churn`, mapeando a coluna `SITUACAO`:  
     - **1 (Churn)**: `CANCELADO`, `EVADIDO`, `CONCLUIDO_REPROVADO`.  
     - **0 (Não churn)**: `CURSANDO`, `FORMADO`, `CONCLUIDO`, `Não Informado`.  
   - Distribuição final da variável alvo:  
     - Churn = 1: 9 445 alunos (≈ 55 %)  
     - Churn = 0: 7 719 alunos (≈ 45 %)  

4. **Análises complementares**  
   - **Estatísticas descritivas**: `.describe()` em variáveis numéricas; identificação de outliers.  
   - **Distribuição de churn**: gráfico de barras ou pizza para visualizar o leve desbalanceamento.  
   - **Bivariadas**: cruzamentos entre `churn` e variáveis-chave (percentual de conclusão, disciplinas aprovadas, acessos ao portal).  
   - **Matriz de correlação**: identificação de multicolinearidade e seleção de features.

> *todos os scripts e gráficos estão disponíveis no notebook do Colab. Consulte-o para detalhes e exemplos visuais.*

---

## Modelos Utilizados

Para prever evasão (churn), selecionamos algoritmos capazes de equilibrar interpretabilidade e desempenho.

| Modelo                      | Descrição                                                                                   | Hiperparâmetros principais                  |
|-----------------------------|---------------------------------------------------------------------------------------------|---------------------------------------------|
| **Regressão Logística**     | Baseline linear, rápido e interpretável.                                                    | `C` (regularização), `solver`               |
| **Árvore de Decisão**       | Regras claras de decisão; fácil interpretação.                                               | `max_depth`, `min_samples_split`            |
| **Random Forest**           | Ensemble de árvores; reduz variância e overfitting.                                         | `n_estimators`, `max_depth`, `max_features` |
| **XGBoost**                 | Boosting de gradiente eficiente; alta performance em dados tabulares.                        | `learning_rate`, `n_estimators`, `max_depth`|
| **SVM**                     | Margem máxima; eficaz em alta dimensionalidade.                                             | `C`, `kernel`                               |


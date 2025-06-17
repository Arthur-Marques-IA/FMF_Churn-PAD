# -*- coding: utf-8 -*-
"""
# Pré-processamento de Dados de Atendimentos de Alunos

Este script detalha o processo completo de pré-processamento aplicado ao conjunto 
de dados de atendimentos de alunos. O objetivo é realizar a limpeza, 
o tratamento de valores ausentes e a transformação dos dados, preparando-os 
para análises futuras e para a modelagem de previsão de churn.

**Responsáveis pelo processo:**
* André
* Arthur
* Ju
"""

import os
import pandas as pd

# ==============================================================================
# --- Passo 1: Importação e Análise Exploratória Inicial ---
# ==============================================================================
# Nesta etapa, damos o primeiro passo fundamental: carregar os dados brutos e
# realizar uma análise exploratória inicial. Utilizamos a biblioteca `pandas`
# para importar o arquivo `atendimentos_de_alunos.csv`.
# ==============================================================================

# === Define o caminho do Arquivo ===
file_path = 'data/interim/atendimentos_de_alunos.csv'    # Ajuste conforme necessário.

# Define o diretório para os dados processados
output_dir_processed = '../data/processed'
if not os.path.exists(output_dir_processed):
    os.makedirs(output_dir_processed)

output_path = os.path.join(output_dir_processed, 'atendimentos_de_alunos_processado.csv')


if not os.path.exists(file_path):
    raise FileNotFoundError(f"Arquivo não encontrado no caminho: {file_path}. Crie um arquivo CSV de exemplo ou aponte para o seu.")

# === Carregar o CSV ===
try:
    df = pd.read_csv(file_path, sep=',', encoding='utf-8')
except pd.errors.ParserError as e:
    raise ValueError(f"Erro de parsing no arquivo CSV. Verifique a estrutura do arquivo: {e}")
except Exception as e:
    raise RuntimeError(f"Ocorreu um erro inesperado durante a leitura do arquivo: {e}")

print("--- Passo 1: Arquivo CSV carregado com sucesso. ---")
print(f"Shape inicial do DataFrame: {df.shape}")

# ==============================================================================
# --- Passo 2: Diagnóstico e Tratamento de Valores Ausentes ---
# ==============================================================================
# Antes de tratar os dados, é essencial identificar a extensão dos valores ausentes.
# Esta etapa diagnostica e depois aplica uma estratégia de imputação.
#
# ### Estratégia de Imputação de Valores Ausentes
#
# A função `fill_missing_values` aplica diferentes estratégias de preenchimento
# com base na natureza de cada coluna.
#
# - **Colunas Categóricas**: Substituídos por 'Não Informado'.
# - **Colunas Numéricas**: Preenchidos com 0.
# - **Colunas de Data**:
#   1. **Criação de Flag**: Uma nova coluna booleana é criada para registrar a ausência.
#   2. **Imputação pela Mediana**: Preenchimento com a mediana das datas existentes.
# ==============================================================================

# Função para preencher valores ausentes
def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preenche valores ausentes (NaN) em colunas específicas de um DataFrame.
    """
    # --- ID de Matrícula ---
    df['MATRICULAID'] = df['MATRICULAID'].fillna(-1)

    # --- Colunas Categóricas/Texto ---
    categorical_cols = [
        'Grupo % Cursado', 'Grupo_Acesso', 'NOME CURSO PADRÃO',
        'Situação Contrato', 'Documentos Pessoais Pendentes', 'SITUACAO',
        'Status_Cliente', 'fezPrimeiroAcesso', 'Forma de Pagamento Oficial', 'ESTADO'
    ]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Não Informado')

    # --- Colunas numéricas que podem ser lidas como string ---
    df['PercentualConclusao'] = df['PercentualConclusao'].fillna(0)
    df['% Docs Pessoais'] = df['% Docs Pessoais'].fillna(0)

    # --- Colunas numéricas ---
    numeric_cols = [
        'DisciplinasAprovadas', 'DisciplinasTotais', '# parcelas Vencidas',
        'Total _Atendimentos', 'Acesso ao Portal', 'Anexar Documentos',
        'Apoio Pedagogico', 'Bot de Atendimento', 'Contato Via Ligação',
        'Correção - Plataforma', 'Correção cadastral', 'Diploma', 'Disparos',
        'Duvidas Gerais', 'Erro', 'Estágio', 'Financeiro', 'Informações Comercias',
        'Onboarding', 'Outros Atendimentos', 'Ouvidoria', 'Problema Técnico',
        'Processos Secretaria', 'Reclame aqui', 'Rematrícula', 'Retenção',
        'Solicitação de documentos', 'Suporte de Acesso', 'Suporte Pedagogico'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # --- Colunas de Data com Mediana + Flag ---
    date_cols = ['DATAMATRICULA', 'ENCERRAMENTO_CONTRATO', 'Data de nascimento']
    for col in date_cols:
        if col in df.columns:
            df[f'{col}_is_missing'] = df[col].isna().astype(int)
            df[col] = pd.to_datetime(df[col], errors='coerce', format="%d/%m/%Y")   # Convertendo para datetime
            median_date = df[col].median()
            df[col] = df[col].fillna(median_date)
            df[col] = df[col].dt.strftime("%d/%m/%Y")                               # Retorna o formato para dd/mm/yyyy

    return df

# === Execução da Limpeza e Verificação ===
print("\n--- Passo 2: Iniciando limpeza de dados... ---")

# 1. Aplica a função para preencher todos os valores nulos
df_processed = fill_missing_values(df.copy()) # Usa cópia para evitar SettingWithCopyWarning

# 2. Remove linhas duplicadas
df_processed.drop_duplicates(inplace=True)

# 3. Realiza uma verificação final
remaining_nulls = df_processed.isnull().sum().sum()
if remaining_nulls == 0:
    print("Verificação concluída: Não há mais valores ausentes no DataFrame.")
else:
    print(f"Atenção: Ainda existem {remaining_nulls} valores ausentes.")

print(f"Shape do DataFrame após limpeza: {df_processed.shape}")

# ==============================================================================
# --- Passo 3: Transformações de Dados e Análise Categórica ---
# ==============================================================================
# Com os dados sem valores nulos, realizamos transformações para preparar
# os dados para modelagem.
#
# - **Conversão para Binário:** Colunas com dois estados ('Sim'/'Não', 'Vigente'/'Encerrado')
#   são mapeadas para 1 e 0.
# - **Análise Categórica:** Identificamos a natureza das variáveis de texto.
# ==============================================================================

print("\n--- Passo 3: Iniciando transformações de dados... ---")

df = df_processed # Continua o processamento no mesmo DataFrame

# === Transformações de Colunas Binárias ===
mapa_binario_sim_nao = {'Sim': 1, 'Não': 0}
mapa_binario_sim_nao_upper = {'SIM': 1, 'NÃO': 0}
mapa_contrato = {'Vigente': 1, 'Encerrado': 0}

if 'fezPrimeiroAcesso' in df.columns:
    df['fezPrimeiroAcesso'] = df['fezPrimeiroAcesso'].map(mapa_binario_sim_nao).fillna(0).astype(int)
    print("Coluna 'fezPrimeiroAcesso' transformada.")

if 'has_contact' in df.columns:
    df['has_contact'] = df['has_contact'].map(mapa_binario_sim_nao_upper).fillna(0).astype(int)
    print("Coluna 'has_contact' transformada.")

if 'Situação Contrato' in df.columns:
    df['Situação Contrato'] = df['Situação Contrato'].map(mapa_contrato).fillna(0).astype(int)
    print("Coluna 'Situação Contrato' transformada.")

print("--- Transformações concluídas! ---")

# ==============================================================================
# --- Passo 4: Engenharia de Atributos - Criação da Variável Alvo (Churn) ---
# ==============================================================================
# O passo final é criar a variável alvo (`churn`) que queremos prever.
#
# **Regra de Negócio para Churn:**
# - **Churn (1)**: Alunos com status `CANCELADO`, `EVADIDO` ou `CONCLUIDO_REPROVADO`.
# - **Não Churn (0)**: Alunos com status `CURSANDO`, `FORMADO`, `CONCLUIDO` ou `Não Informado`.
#
# A coluna original `SITUACAO` é removida após a criação da variável alvo.
# ==============================================================================

print("\n--- Passo 4: Criando a variável alvo 'churn'... ---")

# --- Mapeamento de Churn ---
mapeamento_churn = {
    'CONCLUIDO_REPROVADO': 1,
    'CANCELADO': 1,
    'EVADIDO': 1,
    'Não Informado': 0, # Tratado como não-churn
    'CONCLUIDO': 0,
    'CURSANDO': 0,
    'FORMADO': 0
}

# --- Criação e Validação da Coluna 'churn' ---
if 'SITUACAO' in df.columns:
    df['churn'] = df['SITUACAO'].map(mapeamento_churn)
    df.dropna(subset=['churn'], inplace=True) # Remove linhas onde o mapeamento falhou, por precaução
    df['churn'] = df['churn'].astype(int)

    # --- Remoção da Coluna Original ---
    try:
        df.drop(columns=['SITUACAO'], inplace=True)
        print("Coluna 'SITUACAO' mapeada para 'churn' e removida com sucesso.")
    except KeyError:
        print("A coluna 'SITUACAO' não foi encontrada para remoção (provavelmente já foi removida).")

    # --- Análise da Nova Coluna ---
    print("\n## Análise da Nova Coluna 'churn' ##")
    print("\nContagem de Churn (1) vs. Não Churn (0):")
    print(df['churn'].value_counts())
    print("\nProporção de Churn no Dataset:")
    print(df['churn'].value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))
else:
    print("Coluna 'SITUACAO' não encontrada, a criação do 'churn' foi pulada.")


# ==============================================================================
# --- Conclusão e Salvamento do Arquivo Final ---
# ==============================================================================
# O DataFrame limpo e processado é salvo em um novo arquivo CSV,
# separando os dados brutos dos dados prontos para análise.
# ==============================================================================

try:
    # Usa ponto e vírgula como separador para compatibilidade com Excel em algumas regiões
    df.to_csv(output_path, index=False, sep=';', encoding='utf-8')
    print("-" * 50)
    print(f"\nSUCESSO: Script de pré-processamento concluído.")
    print(f"O arquivo final foi salvo em: {output_path}")
    print(f"Shape final do DataFrame: {df.shape}")
    print("-" * 50)
except Exception as e:
    print(f"\nERRO: Não foi possível salvar o arquivo processado. Motivo: {e}")
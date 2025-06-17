# Path: utils/id_obfuscator.py

# === START ===

"""
Ofuscador de IDs para dados sensíveis.

Este módulo realiza a ofuscação determinística de identificadores numéricos (no exemplo, matrícula de alunos)
com base em uma transformação modular. A ofuscação garante a irreversibilidade prática dos dados sem o
conhecimento dos parâmetros definidos no arquivo `.env`, preservando a privacidade em conjuntos de dados
pessoais ou sensíveis.

Variáveis de ambiente esperadas:
- OBFUSCATE_MULTIPLIER: multiplicador utilizado na ofuscação
- OBFUSCATE_OFFSET: deslocamento aplicado após a multiplicação
- OBFUSCATE_MODULUS: valor para a operação módulo

Além disso, o script pode ser executado diretamente para aplicar a ofuscação à coluna `MATRICULAID`
de um arquivo CSV específico (`atendimentos_de_alunos.csv`), sobrescrevendo o arquivo original com os
dados ofuscados.

Funções principais:
- obfuscate_id(id_int): retorna o ID ofuscado.
- reverse_obfuscate_id(obfuscated): reverte um ID ofuscado para seu valor original, assumindo os parâmetros corretos.

Requer:
- Variáveis no arquivo `.env`
- Pandas
- python-dotenv
"""

# === Importações ===

import os
from dotenv import load_dotenv
import pandas as pd

# === Carregando o `.env` ===

load_dotenv()

# --- Validando as variáveis no .env como do tipo Int ---
def get_env_int(key: str) -> int:
    value = os.getenv(key)
    if value is None:
        raise ValueError(f"Missing environment variable: {key}")
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"Invalid integer for environment variable {key}: {value}")

# === Variáveis Ocultas ===

MULTIPLIER = get_env_int("OBFUSCATE_MULTIPLIER")
OFFSET = get_env_int("OBFUSCATE_OFFSET")
MODULUS = get_env_int("OBFUSCATE_MODULUS")

# === Funções ===

def obfuscate_id(id_int: int) -> int:
    """
    Retorna um inteiro ofuscado baseado em transformações do ID original.
    Determinista e irreversível sem saber os parÂmetros originais.
    """
    return (id_int * MULTIPLIER + OFFSET) % MODULUS

def reverse_obfuscate_id(obfuscated: int) -> int:
    """
    Reverte o inteiro ofuscado para um ID original.
    Funciona com sucesso apenas com as chaves corretas.
    """
    inverse_multiplier = pow(MULTIPLIER, -1, MODULUS)
    return ((obfuscated - OFFSET) * inverse_multiplier) % MODULUS

# === Exemplo de Execução ===

"""
⚠️ Atenção: Execute este código apenas se tiver certeza do que está fazendo.

Recomenda-se executar antes o notebook 'preprocessing/data_preprocessing.ipynb'
para garantir que os dados estejam devidamente preparados e validados.
"""

# --- Carregando o CSV ---

file_path = 'data/interim/atendimentos_de_alunos.csv'
df = pd.read_csv(file_path, sep=";")

# --- Aplica Ofuscação na coluna MATRICULAID ---

df = df.dropna(subset=["MATRICULAID"])
df["MATRICULAID"] = df["MATRICULAID"].astype(int).apply(obfuscate_id)
df.to_csv(file_path, index=False)

# === END ===
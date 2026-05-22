import requests
import re

url = "https://www.demneuropsy.org/abn/"

response = requests.get(url)
html = response.text

# Regex: Nome (Cidade/UF ou Cidade, Brazil)
padrao = re.findall(r'([A-ZÁÉÍÓÚÂÊÔÃÕÇa-zÀ-ÿ\s\.\-]+)\s*\(([^)]+)\)', html)

linhas = []

for nome, local in padrao:
    nome = nome.strip().upper()
    print("---",nome, local )

    # Extrair cidade (antes da vírgula ou /)
    cidade = re.split(r"[,/]", local)[0].strip().upper()

    linhas.append(f"{nome} - {cidade}")
    print(f"{nome} - {cidade}")
    

# Remover duplicados
linhas = sorted(set(linhas))

# Salvar TXT
with open("medicos_abn.txt", "w", encoding="utf-8") as f:
    for linha in linhas:
        f.write(linha + "\n")

print("Arquivo gerado: medicos_abn.txt")
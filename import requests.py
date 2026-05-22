import requests
import pdfplumber
import re

# URL do PDF
url = "https://www.gov.br/saude/pt-br/composicao/sgtes/mais-medicos/comunicados/lista-de-profissionais-aptos-avaliacao-desempenho-anual.pdf"

pdf_path = "lista_medicos.pdf"
txt_path = "nomes.txt"

# 🔽 Baixar o PDF
print("Baixando PDF...")
response = requests.get(url)
with open(pdf_path, "wb") as f:
    f.write(response.content)

print("PDF baixado com sucesso.")

# 📖 Extrair texto do PDF
print("Extraindo texto...")
linhas = []

with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        texto = page.extract_text()
        if texto:
            linhas.extend(texto.split("\n"))

print("Texto extraído.")

# 🔍 Processar para pegar só nomes
nomes = []

for linha in linhas:
    linha = linha.strip()

    if not linha:
        continue

    # Ignorar CPF mascarado
    if re.search(r"XXX\.\d+", linha):
        continue

    # Ignorar UF (2 letras)
    if re.fullmatch(r"[A-Z]{2}", linha):
        continue

    # Filtrar linhas com padrão de nome (tudo maiúsculo)
    if re.fullmatch(r"[A-ZÁÉÍÓÚÂÊÔÃÕÇ\s]+", linha):
        # Evitar cidades (heurística: nomes geralmente >= 2 palavras)
        if len(linha.split()) >= 2:
            nomes.append(linha)

# Remover duplicados e ordenar
nomes = sorted(set(nomes))

# 💾 Salvar arquivo
with open(txt_path, "w", encoding="utf-8") as f:
    for nome in nomes:
        f.write(nome + "\n")

print(f"Arquivo gerado: {txt_path}")
print(f"Total de nomes: {len(nomes)}")

import pdfplumber
import pandas as pd
import re

# Caminho do PDF
pdf_path = "lista-de-profissionais-mediciana_para_enriquecimento.pdf"

registros = []

with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        
        if not text:
            continue
        
        # Quebra por CPF (cada registro começa com XXX.xxx.xxx-XX)
        blocos = re.split(r'XXX\.\d{3}\.\d{3}-XX', text)
        
        for bloco in blocos:
            bloco = bloco.strip()
            
            if not bloco:
                continue
            
            tokens = bloco.split()
            
            if len(tokens) < 4:
                continue
            
            # UF = último token (2 letras)
            uf = tokens[-1]
            
            # Cidade = tokens antes da UF até detectar padrão
            cidade_tokens = []
            i = len(tokens) - 2
            
            while i >= 0 and tokens[i].isupper():
                cidade_tokens.insert(0, tokens[i])
                i -= 1
                
                # evita "comer" o nome todo
                if len(cidade_tokens) >= 4:
                    break
            
            cidade = " ".join(cidade_tokens)
            
            # Nome = resto
            nome_tokens = tokens[:i+1]
            nome = " ".join(nome_tokens)
            
            # Filtro de segurança
            if len(nome.split()) >= 2:
                registros.append({
                    "nome": nome,
                    "cidade": cidade,
                    "UF": uf
                })

# Criar DataFrame
df = pd.DataFrame(registros)

# Remover duplicados
df = df.drop_duplicates()

# Salvar CSV
df.to_csv("medicos_extraidos.csv", index=False, encoding="utf-8")

print("✅ CSV gerado com sucesso!")


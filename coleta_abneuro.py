
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

BASE = "https://abneuro.org.br/encontre-seu-medico/"

headers = {
    "User-Agent": "Mozilla/5.0"
}

def get_listing(page=1):
    url = f"{BASE}?paged={page}"
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")
    return soup

def extract_doctors(soup):
    docs = []
    entries = soup.select("h2, h3, p")

    for e in entries:
        text = e.get_text(strip=True)
        print("Texto -- ", text)

        if "-" in text and len(text) < 120:
            parts = text.split("-")
            if len(parts) >= 3:
                cidade = parts[0].strip()
                estado = parts[-1].strip()
                
                print("Cidade -- ", cidade)

                # nome aparece logo depois
                next_el = e.find_next("strong")
                if next_el:
                    nome = next_el.get_text(strip=True)
                    link = next_el.find_parent("a")
                    link = link["href"] if link else None

                    docs.append({
                        "nome": nome,
                        "cidade": cidade,
                        "uf": estado,
                        "link": link
                    })
                    
                    print("nome -- ", nome)
    return docs


def get_email(url):
    if not url:
        return None
    try:
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text()

        for word in text.split():
            if "@" in word:
                return word.strip()
    except:
        return None
    return None


all_docs = []

page = 1
while True:
    print(f"Página {page}")
    soup = get_listing(page)
    docs = extract_doctors(soup)

    if not docs:
        break

    for d in docs:
        d["email"] = get_email(d["link"])
        all_docs.append(d)
        time.sleep(1)

    page += 1


df = pd.DataFrame(all_docs)

df.head()

df.to_csv("medicos_abn.csv", index=False)

print("Finalizado!")


from googlesearch import search

# Na maioria dos IPs o resultado não será exibido por conta 
# de um capcha exigido pelo google.
#


def web_search(query:str) -> str: # here -> str is the data type

  max_results = 10
  language = "en"
  results = search(query, num_results=max_results, lang=language, advanced=True)
  context = ""
  for result in results:
    context += result.description
  return context


print('Chamda da funcao de busca')
print('Aqui vai o resutlado da busca..: ', web_search("Robson Tavares Nonat"))


from googlesearch import search
for result in search("Python programming", num_results=5):
    print(result) # Prints the result URL
    
    


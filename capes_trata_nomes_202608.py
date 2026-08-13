
#####################################################
# Trata nomes Lista CAPES
# data 12 de agosto de 2026
#
#

from networkx import display
import pandas as pd

df = pd.read_csv('/workspaces/OBJDETEC/resultadoCNPQ_2026_04.csv', sep=",", encoding='utf-8', engine='python')
df.info()

df_menor = df[['Proposta', 'Candidato','Nota_Final','Linha_Original']]
df_menor.info()
print(df_menor)
print('Dados carregados com sucesso!')

tabela_final = pd.DataFrame({'id':['1'],'Candidato':['Robson Tavares']})



for index, row in df_menor.iterrows():
    print(row['Proposta'], row['Candidato'],row['Nota_Final'])
    novo_nome = row['Candidato'] + ' ' + row['Nota_Final']
    vetor = novo_nome.split(' ')
    final = ''
    for nome in vetor:
        nome = nome.replace(',', '')
        if nome.isdigit():
            break
        else:
            #print(f'Nome inválido: {nome}') 
            final = final + ' ' + nome     
        
            
    print(' -- ' + final)
    #print(index)
    # 2. Insert using a dictionary (safer, targets specific columns)
    tabela_final.loc[len(tabela_final)] = {'id':index,'Candidato': final}
    

tabela_final.to_csv('/workspaces/OBJDETEC/resultado_CAPES_2026_04_tratado.csv', sep=',', index=False, encoding='utf-8')

    

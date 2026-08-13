
#####################################################
# Trata nomes Lista CAPES
# data 12 de agosto de 2026
#
#

from networkx import display
import pandas as pd

df = pd.read_csv('/workspaces/OBJDETEC/Resultado_CAPES_CNPQ_2026_04.csv', sep=",", encoding='utf-8', engine='python')
df.info()

df_menor = df[['Linha', 'Processo','Nome_Proponente','Instituicao']]

df_menor.info()
print(df_menor)
print('Dados carregados com sucesso!')

tabela_final = pd.DataFrame({'id':['1'],'Candidato':['Robson Tavares'],'Instituicao':['SISGEENCO']})



for index, row in df_menor.iterrows():
    #print(row['Processo'], row['Nome_Proponente'])
    novo_nome = row['Nome_Proponente'].replace('Universidade','|Universidade')
    vetor = novo_nome.split('|')
    final = ''
    for nome in vetor:
        #nome = nome.replace(',', '')
        if nome.isdigit():
            break
        else:
            #print(f'Nome inválido: {nome}') 
            final = final + ' ' + nome     
        
    final = final.replace('Universidade',',Universidade')
    final = final.replace('Centro',',Centro')
    final = final.replace('Instituto',',Instituto')
    final = final.replace('Funda',',Funda')
    final = final.replace('Assoc',',Assoc')
    final = final.replace('SENAI',',SENAI')
    final = final.replace('Socie',',Socie')
    final = final.replace('Pontif',',Pontif')
    final = final.replace('Emp',',Emp')
    final = final.replace('SOCI',',SOCI')
    final = final.replace('Escola',',Escola')
    final = final.replace('Secretaria',',Secretaria')
    final = final.replace('UNIVERSIDADE',',UNIVERSIDADE')
    final = final.replace('COMIS',',COMIS')
    final = final.replace('Faculd',',Faculd')
    final = final.replace('Hospit',',Hospot')
    final = final.replace('Grupo',',Grupo')
    final = final.replace('Labor',',Labor')
    final = final.replace('Soc',',Soc')
    final = final.replace('INSTITU',',INSTITU')
    final = final.replace('Serv',',Serv')
    final = final.replace('Acad',',Acad')
    final = final.replace('ASSO',',ASSO')
    final = final.replace('FUNDA',',FUNDA')
    final = final.replace('FUCA',',FUCA')
    final = final.replace('Federa',',Federa')
        
    
    
    
    
    print(' -- ' + final)
    #print(index)
    # 2. Insert using a dictionary (safer, targets specific columns)
    tabela_final.loc[len(tabela_final)] = {'id':index,'Candidato': final}
    

tabela_final.to_csv('/workspaces/OBJDETEC/resultado_CNPQ_2026_04_tratado.csv', sep=',', index=False, encoding='utf-8')

    

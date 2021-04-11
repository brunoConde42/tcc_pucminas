print ('Importando bibliotecas...')
import tweepy as tw
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from sklearn.metrics import classification_report
import unidecode
import itertools
import collections

###CARGA TWEETS

#variaveis
consumer_key = "***"
consumer_secret = "***"
access_token = "***"
access_token_secret = "***"

print('Iniciando API...')
#definindo as variaveis de autenticacao e iniciando API
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth,wait_on_rate_limit=True)

print('Consultando API...')
#definindo palavras chave
search_words = "(@ANS_Reguladora) OR #ANS OR #planosdesaude" + " -filter:retweets"

#coletando tweets
tweets = tw.Cursor(api.search,
              q=search_words,
              lang="pt").items(1000)

dados = [[tweet.id,
          tweet.text,
          tweet.created_at,
          tweet.retweet_count,
          tweet.favorite_count,
          tweet.user.screen_name, 
          tweet.user.location,
          tweet.user.description,
          tweet.user.verified,
          tweet.user.followers_count,
          tweet.user.friends_count,
          tweet.user.created_at] for tweet in tweets]

df = pd.DataFrame(data=dados, 
                        columns=['id','text', "created_at", 'retweet_count',
                                 'favorite_count','user_screen_name','user_location','user_description','user_verified',
                                 'followers_count','friends_count','user_created_at'])

print(df.shape)

### TRATAMENTO DADOS

print('Iniciando tratamento dos dados')

def remove_url(txt):
    return " ".join(re.sub(r'http\S+', '', txt, flags=re.MULTILINE).split())
def remove_users(txt):
    return " ".join(re.sub(r'@\S+',"",txt).split())
def remove_hashtags(txt):
    return " ".join(re.sub(r'#\S+',"",txt).split())

print('Removendo urls, users, hashtags...')
df["text_tratado"] = df.text.apply(remove_url)
df["text_tratado"] = df.text_tratado.apply(remove_users)
df["text_tratado"] = df.text_tratado.apply(remove_hashtags)

print('Mais tratamentos...')
df["text_tratado"] = df.text_tratado.apply(str.lower)
df["text_tratado"] = df['text_tratado'].str.replace('[^\w\s]','')
df["text_tratado"] = df['text_tratado'].apply(unidecode.unidecode)

print('Dividindo as palavras...')
df["text_tratado"] = df['text_tratado'].str.split()
todas_palavras = list(itertools.chain(*df['text_tratado']))
frequencia_palavras = collections.Counter(todas_palavras)

print('Baixando as stopwords...')
nltk.download('stopwords')
stop_words_acento = set(stopwords.words('portuguese'))
stop_words = set([unidecode.unidecode(word) for word in stop_words_acento])

print('Removendo as stop_words...')
for index,row in df.iterrows():
    for palavra in row['text_tratado']:
        if palavra in stop_words:
            df.at[index,'text_tratado'].remove(palavra)

todas_palavras = list(itertools.chain(*df['text_tratado']))            
frequencia_palavras = collections.Counter(todas_palavras)

print(df.shape)

###ANALISA SENTIMENTOS

def carrega_dicionario():
    sentilexLem = open("SentiLex-lem-PT02.txt", 'r')
    sentilexFlex = open("SentiLex-flex-PT02.txt", 'r')

    #gera a lista com as palavras lem
    for i in sentilexLem.readlines(): 
        pos_ponto = i.find('.')
        palavra = (i[:pos_ponto])
        palavra = palavra.lower()
        palavra = palavra.replace('[^\w\s]','')
        palavra = unidecode.unidecode(palavra)    
        pol_pos = i.find('POL')
        polaridade = (i[pol_pos+7:pol_pos+9]).replace(';','')
        if int(polaridade) == 0:
            palavras_neutras.append(palavra)
        elif int(polaridade) == 1:
            palavras_positivas.append(palavra)
        elif int(polaridade) == -1:
            palavras_negativas.append(palavra)

    #gera a lista com as palavras flex
    for i in sentilexFlex.readlines(): 
        pos_ponto = i.find('.')
        palavra = (i[:pos_ponto])
        #aplicando os mesmos tratamentos nas palavras
        palavra = palavra.lower()
        palavra = palavra.replace('[^\w\s]','')
        palavra = unidecode.unidecode(palavra)
        pol_pos = i.find('POL')
        polaridade = (i[pol_pos+7:pol_pos+9]).replace(';','')
        if int(polaridade) == 0:
            palavras_neutras.append(palavra)
        elif int(polaridade) == 1:
            palavras_positivas.append(palavra)
        elif int(polaridade) == -1:
            palavras_negativas.append(palavra)

def altera_dicionario():
    #negativas
    global palavras_negativas
    global palavras_positivas
    global palavras_neutras
    novas_palavras_negativas = [
    'denuncia','genocidio','esculhambacao','inacreditavel','baixaria','quadrilha','cumplice','boicote','barrando',
    'porra','esperando','delegacia','facista','fim','cansei','prejudicada','fuder','goela abaixo',
    'fura fila','burocracia','processar','crime','bizarro','jamais','charlatanismo','descredenciamento',
    'atrasar','bosta','farroupilha','lotado','reclamacao','cu','expulsou']
    palavras_negativas += novas_palavras_negativas
    palavras_negativas.remove('obrigado')
    #positivas
    palavras_positivas.remove('seguro')
    palavras_positivas.append('parabens')

def analisa_sentimento():
    df['valor_sentimento'] = df.text_tratado.apply(sentiment,args=(palavras_positivas, palavras_negativas))
    df['palavras_negativas'] = df.text_tratado.apply(sentiment_negative,args=(palavras_positivas, palavras_negativas))
    df['palavras_positivas'] = df.text_tratado.apply(sentiment_positive,args=(palavras_positivas, palavras_negativas))

def sentiment(words, pos_list, neg_list):
    sent = 0

    for word in words:
        if word in neg_list:
            sent = -1
            break
        elif word in pos_list:
            sent = 1
            break
    
    return sent

def sentiment_positive(words, pos_list, neg_list):
    sent = []
    
    for word in words:
        if word in pos_list:
            sent.append(word)
    
    return sent

def sentiment_negative(words, pos_list, neg_list):
    sent = []
    
    for word in words:
        if word in neg_list:
            sent.append(word)
    
    return sent

palavras_negativas = []
palavras_positivas = []
palavras_neutras = []

print('Carregando dicionario...')
carrega_dicionario()
print('Enriquecendo dicionário...')
altera_dicionario()
print('Analisando sentimento...')
analisa_sentimento()

print(df.shape)

### PREPARACAO DASHBOARD

df['POSITIVO'] = 0
df['NEGATIVO'] = 0
df['NEUTRO'] = 0
df['SPAM'] = 0
df['POLARIDADE'] = ''

print('Aplicando rótulos...')
for index,row in df.iterrows():
    if row['valor_sentimento'] == 0:
        df.at[index,'NEUTRO'] = 1
        df.at[index,'POLARIDADE'] = 'NEUTRO'
    elif row['valor_sentimento'] > 0:
        df.at[index,'POSITIVO'] = 1
        df.at[index,'POLARIDADE'] = 'POSTIVO'
    elif row['valor_sentimento'] < 0:
        df.at[index,'NEGATIVO'] = 1
        df.at[index,'POLARIDADE'] = 'NEGATIVO'
        
print('Definindo a coluna index como str...')
df['id']=df['id'].astype(str)

def remove_stopwords(text, *stopwords):
    return ' '.join([word for word in text.split() if word not in stopwords])

#aplicando minusculas
df["nuvem_palavras"] = df.text.apply(str.lower)

#adiconando a coluna text_tratada removendo as urls
df["nuvem_palavras"] = df.nuvem_palavras.apply(remove_url)
df["nuvem_palavras"] = df.nuvem_palavras.apply(remove_users)

df["nuvem_palavras"] = df['nuvem_palavras'].str.replace('[^\w\s]','')
df["nuvem_palavras"] = df['nuvem_palavras'].apply(unidecode.unidecode)

print('baixando as stopwords...')
nltk.download('stopwords')
stop_words_acento = set(stopwords.words('portuguese'))

print('retirando acentuacao...')
stop_words = set([unidecode.unidecode(word) for word in stop_words_acento])

print('adicionando palavras que nao vieram nas stopwords...')
tirar_essas_tambem = set([
'e','a','pra','o','nao','dia','ta','dm','vc','ser','vcs','que','ai',
'q','vai','tar','ola','faz','fazer','ter','p','pq','to','ne'])

print('retirando palavras...')
df["nuvem_palavras"] = df.nuvem_palavras.apply(remove_stopwords,args=stop_words)
df["nuvem_palavras"] = df.nuvem_palavras.apply(remove_stopwords,args=tirar_essas_tambem)

print('colocando todas as palavras em uma única lista...')
palavras_tweets = [tweet.lower().split() for tweet in df['nuvem_palavras']]
todas_palavras = list(itertools.chain(*palavras_tweets))

print('Criando novas colunas...')
df['USER_SEGUIDORES'] = ''
df['USER_SEGUIDORES_SEGUIDORES'] = ''
df['USER_SEGUIDORES_ALTA'] = 0
df['USER_SEGUIDORES_MEDIA'] = 0
df['USER_SEGUIDORES_BAIXA'] = 0
df['USER_SEGUIDORES_ALTO'] = 0
df['USER_SEGUIDORES_MEDIO'] = 0
df['USER_SEGUIDORES_BAIXO'] = 0
df['USER_SEGUIDORES_SINISTRO'] = 0
df['USER_SEGUIDORES_SINISTRA'] = 0

print('Criando rótulo de popularidade...')
for index,row in df.iterrows():
    if row['followers_count'] in range(0,100):
        df.at[index,'USER_SEGUIDORES'] = 'BAIXA'
        df.at[index,'USER_SEGUIDORES_BAIXO'] = 1
    elif row['followers_count'] in range(101,1000):
        df.at[index,'USER_SEGUIDORES'] = 'MEDIA'
        df.at[index,'USER_SEGUIDORES_MEDIO'] = 1
    elif row['followers_count'] in range(1001,100000):
        df.at[index,'USER_SEGUIDORES'] = 'ALTA'
        df.at[index,'USER_SEGUIDORES_ALTO'] = 1
    elif row['followers_count'] > 100000:
        df.at[index,'USER_SEGUIDORES'] = 'SINISTRA'         
        df.at[index,'USER_SEGUIDORES_SINISTRO'] = 1
print('Gerando excel...')
df.to_excel("tweets_ans_teste.xlsx")
print('tchau...')


# coding: utf-8

# In[389]:


import re
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
import nltk as nk
import pandas as pd
import nltk
from textblob import TextBlob
file = open("PostDebateTweets.csv")


# In[390]:

df = pd.read_csv(file)


# In[391]:

df


# In[392]:




# In[393]:




# In[394]:


stop = stopwords.words('english') + ["RT", "via", "...", "I","Clinton","Debate","DebateNight","U","ed","The","Trump","Hillary","Donald","https","trump","donald","say","said"]
positive = ["love","like","agree","correct","great","good",""]
negative = ["scandal","mistreat","hate","racist","strike","Controversy","Investigation","horrible","stupid","disgust"]     
hilList=[]
trumpList=[]
def contains(comp,lst):
    k=0
    for sentence in lst:
        each = sentence.split()
        k=0
        for word in each:
            if(word in comp):
                hilList.append(sentence)
            else:
                trumpList.append(sentence)            


def calAverage(candidate):
    words = 0
    total = 0
    for each in candidate:
        for word in each:
            words = words + 1
    
        total = total + 1
    return (words/total)

hilpp = []
trumpp = []
def process(lst):
    i=0
    stop_words = stopwords.words("english")
    contains(["Hillary","hillary","Clinton","clinton","@hillaryclinton"],lst)
   



# In[395]:

def clean(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet).split())
lis = []
polarity = []
subjectivity = []
for each in df['x']:
    lis.append(clean(each))
    analysis = TextBlob(clean(each))
    polarity.append(analysis.sentiment.polarity)
    subjectivity.append(analysis.sentiment.subjectivity)
clean = pd.DataFrame(lis)
pol = pd.DataFrame(polarity)
sub = pd.DataFrame(subjectivity)
clean.columns = ['tweets']
clean['polarity'] = pol[0]
clean['subjectivity'] = sub[0]
clean


# In[ ]:




# In[396]:

process(clean.tweets)
hildf = pd.DataFrame(hilList)
trumpdf = pd.DataFrame(trumpList)
trumpdf.drop(trumpdf.index[0:4])
polTrump=[]
subTrump=[]
polHil=[]
subHil=[]
for each in trumpdf[0]:
    analysis = TextBlob(each)
    polTrump.append(analysis.sentiment.polarity)
    subTrump.append(analysis.sentiment.subjectivity)
polTdf = pd.DataFrame(polTrump)
subTdf = pd.DataFrame(subTrump)
trumpdf['polarity'] = polTdf[0]
trumpdf['subjectivity']=subTdf[0]

for each in hildf[0]:
    analysis = TextBlob(each)
    polHil.append(analysis.sentiment.polarity)
    subHil.append(analysis.sentiment.subjectivity)
polHdf = pd.DataFrame(polHil)
subHdf = pd.DataFrame(subHil)
hildf['polarity'] = polHdf[0]
hildf['subjectivity']=subHdf[0]
trumpdf


# In[397]:




# In[398]:

from matplotlib import pyplot as plt
import numpy as np
tot = 0
bot =0
for each in clean['polarity']:
    if(each<0):
        tot = tot +1
    elif(each>0):
        bot = bot+1       
proportions = [tot,bot]
plt.pie(
    proportions,
    labels = ['Positive', 'Negative'],
    shadow = False,
    colors = ['blue','red'],
    startangle = 90,
    autopct = '%1.1f%%'
    )
plt.axis('equal')
plt.title('Polarity of entire data set')
plt.show()


# In[399]:

binsVal = np.arange(-1,1,.1)
td = clean['polarity'].sort_values(ascending = False)
np.array(td).astype(np.float)
plt.hist(td, bins = binsVal)
plt.xlabel('polarity')
plt.ylabel('Frequency')
plt.title('Polarity of clean tweets')
plt.show()


# In[400]:

plotPol=[]
subPol=[]
size = []
for each in clean['polarity']:
    plotPol.append(each)
for each in clean['subjectivity']:
    subPol.append(each)
for each in clean['tweets']:
    words =  each.split()
    num =0
    for word in words:
        num = num+1
    size.append(num)
i = 0
for each in size:
    i=i+1
    if(each>170):
        print(each)
        print(i)
size.remove(216)
del plotPol[1426]
len(size)
len(plotPol)
del subPol[1426]


# In[401]:

plt.scatter(subPol,plotPol,color = 'black')
plt.xlabel('subjectivity')
plt.ylabel('polarity')
plt.title('Subjectivity vs Polarity of clean tweets')
plt.show()


# In[402]:

plt.scatter(plotPol,size,color ='green')
plt.xlabel('polarity')
plt.ylabel('size')
plt.title('Polarity vs length of each tweet of entire data set')
plt.show()


# In[403]:

calAverage(trumpdf[0])
calAverage(hildf[0])
data = []
trumpdf['polarity'].sum()
hildf['polarity'].sum()
sizeTrumpdf=[]
sizeHildf=[]
for each in trumpdf[0]:
    words =  each.split()
    num =0
    for word in words:
        num = num+1
    sizeTrumpdf.append(num)

for each in hildf[0]:
    words =  each.split()
    num =0
    for word in words:
        num = num+1
    sizeHildf.append(num)


np.var(sizeTrumpdf)
np.var(sizeHildf)
np.var(size)
np.mean(sizeTrumpdf)
np.mean(sizeHildf)
np.mean(size)
np.mean(clean['polarity'])

np.mean(trumpdf['polarity'])
np.mean(hildf['polarity'])
np.var(clean['polarity'])
np.var(trumpdf['polarity'])
np.var(hildf['polarity'])
np.std(size)
np.std(clean['polarity'])
np.std(clean['subjectivity'])
np.var(clean['subjectivity'])
np.mean(clean['subjectivity'])
np.std(trumpdf['polarity'])
np.std(trumpdf['subjectivity'])
np.std(sizeTrumpdf)
np.mean(trumpdf['subjectivity'])
np.var(hildf['subjectivity'])
np.var(hildf['polarity'])
np.mean(hildf['subjectivity'])
np.std(sizeHildf)
np.std(hildf['polarity'])
np.std(hildf['subjectivity'])
del trumpdf.polarity[399]
sizeTrumpdf.remove(216)


# In[404]:

plt.subplot(221)
plt.scatter(trumpdf['polarity'],sizeTrumpdf,color = 'red')
plt.title('Trump tweets')
plt.xlabel('polarity')
plt.ylabel('size')
plt.subplot(222)
plt.scatter(hildf['polarity'],sizeHildf,color = 'blue')
plt.xlabel('polarity')
plt.ylabel('size')
plt.title('Hillary tweet')
plt.subplots_adjust(hspace=.35)
plt.show()


# In[405]:

count_word = Counter()
for each in hildf[0]:
    word = each.split()
    if(word not in stop):
        count_word.update(word)


# In[406]:

word = (count_word.most_common(150))


# In[407]:




# In[408]:

hilwords = []
counts = []
x = 0
for each,item in word:
       if(x<7 and each not in stop):
            hilwords.append(each)
            counts.append(item)
            x=x+1
            
y_pos = np.arange(len(hilwords))
plt.bar(y_pos, counts, align='center', alpha=0.5,color = 'blue')
plt.xticks(y_pos, hilwords)
plt.ylabel('Usage')
plt.title('Most Common words: Hillary Tweets')
plt.show()


# In[409]:

count_wordT = Counter()
for each in trumpdf[0]:
    word = each.split()
    if(word not in stop):
        count_wordT.update(word)
Twords = []
Tcounts = []
x = 0
Tword = (count_wordT.most_common(150))
for each,item in Tword:
       if(x<7 and each not in stop):
            Twords.append(each)
            Tcounts.append(item)
            x=x+1
            
y_posT = np.arange(len(Twords))
plt.bar(y_posT, Tcounts, align='center', alpha=0.5,color = 'red')
plt.xticks(y_posT, Twords)
plt.ylabel('Usage')
plt.title('Most Common words: Trump Tweets')
plt.show()


# In[410]:

plt.subplot(321)
binsValT = np.arange(-1,1,.1)
tt = trumpdf['polarity'].sort_values(ascending = False)
np.array(tt).astype(np.float)
plt.hist(tt, bins = binsValT,color='red')
plt.xlabel('polarity')
plt.ylabel('Frequency')
plt.title('Polarity of Trump tweets')
plt.subplot(322)
binsValH = np.arange(-1,1,.1)
th = hildf['polarity'].sort_values(ascending = False)
np.array(th).astype(np.float)
plt.hist(th, bins = binsValH)
plt.xlabel('polarity')
plt.ylabel('Frequency')
plt.title('Polarity of Hillary tweets')
plt.show()


# In[ ]:




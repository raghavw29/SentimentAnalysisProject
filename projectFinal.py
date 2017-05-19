
import re
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
import nltk as nk
import pandas as pd
import nltk
from textblob import TextBlob
from matplotlib import pyplot as plt
import numpy as np

file = open("PostDebateTweets.csv")
df = pd.read_csv(file)
stop = stopwords.words('english') + ["RT", "via", "...", "I","Clinton","Debate","DebateNight","U","ed","The","Trump","Hillary","Donald","https","trump","donald","say","said"]
positive = ["love","like","agree","correct","great","good",""]
negative = ["scandal","mistreat","hate","racist","strike","Controversy","Investigation","horrible","stupid","disgust"]     
hilList=[]
trumpList=[]

# Sorts tweets into hillary and trump tweets
def contains(comp,lst):
    for sentence in lst:
        each = sentence.split()
        for word in each:
            if(word in comp):
                hilList.append(sentence)
            else:
                trumpList.append(sentence)            

#calculates average words in each candidates tweets
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
#processing the tweets removing stopwords
def process(lst):
    i=0
    stop_words = stopwords.words("english")
    contains(["Hillary","hillary","Clinton","clinton","@hillaryclinton"],lst)

#removes hashtags,@ symbols RT's
def clean(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet).split())

#Determines the Polarity and Subjectiviy of the tweets
def find_Polarity_and_Subjectivity(list):
    lis = []
    polarity = []
    subjectivity = []
    for each in list:
        lis.append(clean(each))
        analysis = TextBlob(clean(each))
        polarity.append(analysis.sentiment.polarity)
        subjectivity.append(analysis.sentiment.subjectivity)
    Frame_to_Make = pd.DataFrame(lis)
    pol = pd.DataFrame(polarity)
    sub = pd.DataFrame(subjectivity)
    Frame_to.Make.columns = ['tweets']
    Frame_to_Make['polarity'] = pol[0]
    Frame_to_Make['subjectivity'] = sub[0]
    return Frame_to_Make



# Find the size of each tweets and prepares the generated list to plotting
def find_Size(lis):
    plotPol=[]
    subPol=[]
    size = []
    for each in lis['polarity']:
        plotPol.append(each)
    for each in lis['subjectivity']:
        subPol.append(each)
    for each in lis['tweets']:
         words =  each.split()
    num =0
    for word in words:
        num = num+1
    size.append(num)

    size.remove(216)
    del plotPol[1426]
    len(size)
    len(plotPol)
    del subPol[1426]
    return size,plotPol,subPol


# gathers the most common words in each candidates tweets
def count_words(lis):
    count_word = Counter()
    for each in lis[0]:
        word = each.split()
        if(word not in stop):
            count_word.update(word)
    word = (count_word.most_common(150))
    return word










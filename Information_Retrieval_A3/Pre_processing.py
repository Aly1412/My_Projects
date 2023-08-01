import os
#from pydoc import doc
import re
from nltk.stem import WordNetLemmatizer
import json
from pathlib import Path
import math
import heapq
from nltk.corpus import stopwords

class Preprocessor:
    def __init__(self):

        self.tf_index = {}          # term frequency index
        self.tf_k_index = {}
        self.idf_index = {}         # inverse document frequency index
        self.tfidf_index = {}       # tfidf index
        self.tfidf_k_index = {}
        self.vocabulary = {}
        self.topfeatures_tf = {}
        self.topfeatures_tfidf = {}
        self.noOfDocs = 0           # total document count
        self.stop_words = []         # stopword list
        self.CollectionDir = "E:\\IR\\IR_A3_19K1412\\Dataset"
        self.DataDir = "E:\\IR\\IR_A3_19K1412\\Data"
    
    def Preprocessing_Pipeline(self):        
        if not os.path.isdir(self.DataDir):  # if directory already not present
            self.LoadStopwordsList()
            # creating indexes
            self.BuildTfIndex()
            self.length_normalization()
            self.BuildIdfIndex()
            self.BuildTfIdfIndex()
            os.mkdir(self.DataDir)          
            self.topKFeatures()
            self.BuildTfIndex_topK()
            self.BuildTfIdfIndex_topK()
            # creating data directory for storing and loading indexes
            self.WriteToDisk(self.tf_index, "tf_index")
            self.WriteToDisk(self.idf_index, "idf_index")
            self.WriteToDisk(self.tfidf_index, "tfidf_index")
            self.WriteToDisk(self.topfeatures_tf,"tf_topKFeatures")
            self.WriteToDisk(self.topfeatures_tfidf,"tfidf_topKFeatures")
            self.WriteToDisk(self.idf_index, "idf_index")
            self.WriteToDisk(self.tfidf_index, "tfidf_index")
            self.WriteToDisk(self.tf_k_index, "tf_index_topk")
            self.WriteToDisk(self.tfidf_k_index, "tfidf_index_topk")
        else:
            # loading indexes from data directory
            self.LoadStopwordsList()
            self.tf_index = self.ReadFromDisk("tf_index")
            self.idf_index = self.ReadFromDisk("idf_index")
            self.tfidf_index = self.ReadFromDisk("tfidf_index")
            self.tf_k_index = self.ReadFromDisk("tf_index_topk")
            self.tfidf_k_index = self.ReadFromDisk("tfidf_index_topk")
            self.noOfDocs = len(self.tf_index.keys())

    
    def tokenization(self, text):
        text = text.lower()                   # case folding
        text = re.sub(r"-", " ", text)        # handling hyphen
        text = re.sub(r"[^\w\s]", " ", text)  # noise removal - replacing all types of [^a-zA-Z0-9] and [^\t\r\n\f] with space for splitting on space
        text = text.split()                   # splitting on space
        return text
    
    def LoadStopwordsList(self):
        self.stop_words = set(stopwords.words("english"))

    def isStopword(self, term):
        # returns true if term present in stopwords list
        return term in self.stop_words

    
    def Lemmatization(self, token):
        # Lemmatization - WordNetLemmatizer used
        l = WordNetLemmatizer()
        return l.lemmatize(token)

    def FilterTokens(self, text):
        # for query parsing and preprocessing
        filteredList = []
        tokens = self.tokenization(text)
        for tok in tokens:
            if not self.isStopword(tok):
                tok = self.Lemmatization(tok)
                filteredList.append(tok)

        return filteredList

    def BuildTfIndex(self):
        # tf_index = {doc1 : { t1 : 3, t2: 4, ... ,tn: 5}, doc1 : { t1 : 2, t2: 1, ... ,tn: 4}, ... , docN : { t1 : 1, t2: 4, ... ,tn: 2} )  
        files = os.listdir(self.CollectionDir)
        files = [int(x.replace(".txt", "")) for x in files]
        files.sort()
        files = [str(x) + ".txt" for x in files]

        docNo = 0
        for filename in files:
            with open(os.path.join(self.CollectionDir, filename), "r") as f:
                text = f.read()
                text_words = self.tokenization(text)
                for word in text_words:

                    if word not in self.vocabulary.keys():
                        self.vocabulary[word]=1

                    if not self.isStopword(word):
                        word = self.Lemmatization(word)
                        if docNo not in self.tf_index.keys():
                            self.tf_index[docNo] = {}              
                        if word not in self.tf_index[docNo].keys():
                            self.tf_index[docNo][word] = 1         
                        else:
                            self.tf_index[docNo][word] += 1        
            docNo += 1

        self.noOfDocs = docNo

    def BuildTfIndex_topK(self):
        files = os.listdir(self.CollectionDir)
        files = [int(x.replace(".txt", "")) for x in files]
        files.sort()
        files = [str(x) + ".txt" for x in files]

        docNo = 0
        for filename in files:
            with open(os.path.join(self.CollectionDir, filename), "r") as f:
                text = f.read()
                text_words = self.tokenization(text)
                self.tf_k_index[docNo]={}
                for word in text_words:
                    if word in self.topfeatures_tf:
                        if not self.isStopword(word):
                            word = self.Lemmatization(word)
                            if word in self.tf_index[docNo].keys():
                                self.tf_k_index[docNo][word]=self.tf_index[docNo][word]   
            docNo += 1
        self.noOfDocs = docNo


    def length_normalization(self):
        self.magnitude = [0] * self.noOfDocs        # each index stores magnitude for a particular document
        for i in range(self.noOfDocs):
            for key in self.tf_index[i].keys():
                self.magnitude[i] += self.tf_index[i][key] ** 2
            self.magnitude[i] = math.sqrt(self.magnitude[i])        # sqrt(tf1^2 + tf2^2 + tf3^2 + ... + tfn^2)


    def BuildIdfIndex(self):
        # idf_index = { t1: idf-Val, t2: idf-Val , t3: idf-Val , ... , t4: idf-val }
        df = {}
        for i in range(self.noOfDocs):
            temp = []
            for key in self.tf_index[i].keys():
                if key not in temp:
                    if key not in df.keys():
                        df[key] = 1
                    else:
                        df[key] += 1
                    temp.append(key)
        # idf will calculated for each unique term
        for k in df.keys():
            self.idf_index[k] = math.log10(self.noOfDocs / df[k])  # idf = log(N/df)
    

    def BuildTfIdfIndex_topK(self):
        # tfidf_k_index = {doc1 : { t1 : 0.21, t2: 2.4, ... ,tn: 0.11}, doc1 : { t1 : 2.4, t2: 0.01, ... ,tn: 0.234}, ... , docN : { t1 : 0.21, t2: 0.344, ... ,tn: 0.2})
        for i in range(self.noOfDocs):
            #print(i)
            self.tfidf_k_index[str(i)] = {}
            for key in self.topfeatures_tfidf:
                if key in self.tfidf_index[str(i)].keys():
                    self.tfidf_k_index[str(i)][key] = self.tfidf_index[str(i)][key]            # tfidf = tf * log(N/df)
                

    def BuildTfIdfIndex(self):
        for i in range(self.noOfDocs):
            self.tfidf_index[str(i)] = {}
            for key in self.tf_index[i].keys():
                tf = (self.tf_index[i][key] / self.magnitude[i])    # length normalizing term frequency vector
                idf = self.idf_index[key]
                self.tfidf_index[str(i)][key] = tf * idf            # tfidf = tf * log(N/df)

    def topKFeatures(self, k=100):
        # Baseline 1 (TF)
        tfidf_sum = {}
        for key in self.vocabulary.keys():
            tfidf_sum[key] = 0
            for i in self.tfidf_index.keys():
                if key in self.tfidf_index[i].keys():
                    tfidf_sum[key] += self.tfidf_index[i][key]
        self.topfeatures_tfidf = heapq.nlargest(k, tfidf_sum, key=tfidf_sum.get)
        #print(self.topfeatures_tfidf)    
        
        # Baseline 2 (TFIDF)
        tf_sum = {}
        for key in self.vocabulary.keys():
            tf_sum[key] = 0
            for i in self.tf_index.keys():
                if key in self.tf_index[i].keys():
                    tf_sum[key] += self.tf_index[i][key]    
        self.topfeatures_tf = heapq.nlargest(k, tf_sum, key=tf_sum.get)
        #print(self.topfeatures_tf)    
   
    def WriteToDisk(self, index, indexType):
        filename = "\\" + indexType + ".txt"
        with open(self.DataDir + filename, "w") as filehandle:
            filehandle.write(json.dumps(index))

    
    def ReadFromDisk(self, indexType):
        filename = "\\" + indexType + ".txt"
        with open(self.DataDir + filename, "r") as filehandle:
            index = json.loads(filehandle.read())
        return index
from pathlib import Path
from Pre_processing import Preprocessor
import os
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
import pandas as pd

class Clustering:

    def __init__(self):
        self.vectors_tf=[]
        self.vectors_tfidf=[]
        self.B1_results = {'Doc':[], 'Predicted_cluster':[], 'Real_cluster':[]} #Baseline 1 approach results ground truth vs predicted
        self.B2_results = {'Doc':[], 'Predicted_cluster':[], 'Real_cluster':[]} #Baseline 2 approach results ground truth vs predicted
        self.p=Preprocessor()
        self.p.Preprocessing_Pipeline()
        #self.DataDir = str(Path(__file__).parent.resolve()).replace("src", "data")
    

    def tokenize_vector_docs(self):
        #vectorization for baseline 1 approach
        for i in self.p.tf_k_index.keys():
            v=[]
            for j in self.p.tf_k_index[i].keys():
                if j.isdigit() and len(j) <= 1:
                    continue
                else:
                    v.append(j)
            self.vectors_tf.append(v)
        #print(self.vectors_tf)

        #vectorization for baseline 2 approach
        for i in self.p.tfidf_k_index.keys():
            v=[]
            for j in self.p.tfidf_k_index[i].keys():
                if j.isdigit() and len(j) <= 1:
                    continue
                else:
                    v.append(j)
            self.vectors_tfidf.append(v)
        #print(self.vectors_tfidf)

    def isStopword(self, term):
        # returns true if term present in stopwords list
        return term in self.p.stop_words

    
    def Lemmatization(self, token):
        # Lemmatization - WordNetLemmatizer used
        l = WordNetLemmatizer()
        return l.lemmatize(token)
    
    # word2vec Models for both baseline 1 and 2  
    def word2vec_embedding(self):
        self.word2vec_model_tf = Word2Vec(sentences=self.vectors_tf, vector_size=100, workers=1, seed=0) # Baseline1 
        self.word2vec_model_tfidf = Word2Vec(sentences=self.vectors_tfidf, vector_size=100, workers=1, seed=1) # Baseline 2

    def tokenization(self, text):
        text = text.lower()                   # case folding
        text = re.sub(r"-", " ", text)        # handling hyphen
        text = re.sub(r"[^\w\s]", " ", text)  # noise removal - replacing all types of [^a-zA-Z0-9] and [^\t\r\n\f] with space for splitting on space
        text = text.split()                   # splitting on space
        return text

    # list of list for features , doc by features 2D array using word2vec model
    def vectorize(self):

        files = os.listdir(self.p.CollectionDir)
        files = [int(x.replace(".txt", "")) for x in files]
        files.sort()
        files = [str(x) + ".txt" for x in files]

        # Baseline 1
        features = []
        for filename in files:
            with open(os.path.join(self.p.CollectionDir, filename), "r") as f:
                text = f.read()
                text_words = self.tokenization(text)
                zero_vector = np.zeros(self.word2vec_model_tf.vector_size)
                vectors = []
                for token in text_words:
                    if not self.isStopword(token):# and token.isdigit() and len(token)>1:
                        token = self.Lemmatization(token)
                        if token in self.word2vec_model_tf.wv:
                            try:
                                vectors.append(self.word2vec_model_tf.wv[token])
                            except KeyError:
                                continue
                if vectors:
                    vectors = np.asarray(vectors)
                    avg_vec = vectors.mean(axis=0)
                    features.append(avg_vec)
                else:
                    features.append(zero_vector)
        self.vectorized_docs_tf=features

        # Basleine 2 
        features = []
        for filename in files:
            with open(os.path.join(self.p.CollectionDir, filename), "r") as f:
                text = f.read()
                text_words = self.tokenization(text)
                zero_vector = np.zeros(self.word2vec_model_tfidf.vector_size)
                vectors = []
                for token in text_words:
                    if not self.isStopword(token):# and token.isdigit() and len(token)>1:
                        token = self.Lemmatization(token)
                        if token in self.word2vec_model_tfidf.wv:
                            try:
                                vectors.append(self.word2vec_model_tfidf.wv[token])
                            except KeyError:
                                continue
                if vectors:
                    vectors = np.asarray(vectors)
                    avg_vec = vectors.mean(axis=0)
                    features.append(avg_vec)
                else:
                    features.append(zero_vector)
        self.vectorized_docs_tfidf=features
    
    
    def Clustering_Baseline_1(self):
        # Clustering for Baseline 1
        self.kmeans_B1 = KMeans(n_clusters=5, random_state=0, n_init="auto")
        self.kmeans_B1.fit(self.vectorized_docs_tf)
        self.predictions_topk_tf = self.kmeans_B1.predict(self.vectorized_docs_tf) 
        #print(self.predictions_topk_tf)
        files = os.listdir(self.p.CollectionDir)
        files = [int(x.replace(".txt", "")) for x in files]
        files.sort()
        files = [str(x) + ".txt" for x in files]
        i=0
        for filename in files:
            self.B1_results['Doc'].append(filename)
            self.B1_results['Predicted_cluster'].append(self.predictions_topk_tf[i]+1)
            i+=1
        
    def Clustering_Baseline_2(self):
        # Clustering for Baseline 2
        self.kmeans_B2 = KMeans(n_clusters=5, random_state=0, n_init="auto")
        self.kmeans_B2.fit(self.vectorized_docs_tfidf)
        self.predictions_topk_tfidf = self.kmeans_B2.predict(self.vectorized_docs_tfidf) 
        #print(self.predictions_topk_tfidf)
        files = os.listdir(self.p.CollectionDir)
        files = [int(x.replace(".txt", "")) for x in files]
        files.sort()
        files = [str(x) + ".txt" for x in files]
        i=0
        for filename in files:
            self.B2_results['Doc'].append(filename)
            self.B2_results['Predicted_cluster'].append(self.predictions_topk_tfidf[i]+1)
            i+=1
        
    def Read_GT(self):
        # Getting actual clusters or ground truth
        for i in range(1,6):
            files_c1 = os.listdir("E:\IR\IR_A3_19K1412\Dataset GT\C"+str(i))
            files_c1 = [int(x.replace(".txt", "")) for x in files_c1]
            files_c1.sort()
            files_c1 = [str(x) + ".txt" for x in files_c1]
            for filename in files_c1:
                self.B1_results['Real_cluster'].append(i)
                self.B2_results['Real_cluster'].append(i)

    def gen_dataframes(self):
        # Generating dataframes for baseline 1 and baseline 2 results
        self.B1_results=pd.DataFrame.from_dict(self.B1_results, orient='columns', dtype=None, columns=None)
        print(self.B1_results.head(5))
        self.B2_results=pd.DataFrame.from_dict(self.B2_results, orient='columns', dtype=None, columns=None)
        print(self.B2_results.head(5))

    def Purity_B2(self):
        # Caculating purity for baseline 2 results
        cls1=self.B2_results.where(self.B2_results["Predicted_cluster"]==1)
        cls2=self.B2_results.where(self.B2_results["Predicted_cluster"]==2)
        cls3=self.B2_results.where(self.B2_results["Predicted_cluster"]==3)
        cls4=self.B2_results.where(self.B2_results["Predicted_cluster"]==4)
        cls5=self.B2_results.where(self.B2_results["Predicted_cluster"]==5)
        
        c1=0
        c2=0
        c3=0
        c4=0
        c5=0
        for i in range(len(cls1)):
            if cls1["Real_cluster"][i]==1:
                c1+=1
            if cls1["Real_cluster"][i]==2:
                c2+=1
            if cls1["Real_cluster"][i]==3:
                c3+=1
            if cls1["Real_cluster"][i]==4:
                c4+=1
            if cls1["Real_cluster"][i]==5:
                c5+=1
        p1=max(c1,c2,c3,c4,c5)

        c1=0
        c2=0
        c3=0
        c4=0
        c5=0
        for i in range(len(cls2)):
            if cls2["Real_cluster"][i]==1:
                c1+=1
            if cls2["Real_cluster"][i]==2:
                c2+=1
            if cls2["Real_cluster"][i]==3:
                c3+=1
            if cls2["Real_cluster"][i]==4:
                c4+=1
            if cls2["Real_cluster"][i]==5:
                c5+=1
        p2=max(c1,c2,c3,c4,c5)

        c1=0
        c2=0
        c3=0
        c4=0
        c5=0
        for i in range(len(cls3)):
            if cls3["Real_cluster"][i]==1:
                c1+=1
            if cls3["Real_cluster"][i]==2:
                c2+=1
            if cls3["Real_cluster"][i]==3:
                c3+=1
            if cls3["Real_cluster"][i]==4:
                c4+=1
            if cls3["Real_cluster"][i]==5:
                c5+=1
        p3=max(c1,c2,c3,c4,c5)

        c1=0
        c2=0
        c3=0
        c4=0
        c5=0
        for i in range(len(cls4)):
            if cls4["Real_cluster"][i]==1:
                c1+=1
            if cls4["Real_cluster"][i]==2:
                c2+=1
            if cls4["Real_cluster"][i]==3:
                c3+=1
            if cls4["Real_cluster"][i]==4:
                c4+=1
            if cls4["Real_cluster"][i]==5:
                c5+=1
        p4=max(c1,c2,c3,c4,c5)

        c1=0
        c2=0
        c3=0
        c4=0
        c5=0
        for i in range(len(cls5)):
            if cls5["Real_cluster"][i]==1:
                c1+=1
            if cls5["Real_cluster"][i]==2:
                c2+=1
            if cls5["Real_cluster"][i]==3:
                c3+=1
            if cls5["Real_cluster"][i]==4:
                c4+=1
            if cls5["Real_cluster"][i]==5:
                c5+=1
        p5=max(c1,c2,c3,c4,c5)

        pure=(p1+p2+p3+p4+p5)/50
        print(pure)  
    
    def Purity_B1(self):
        # Caculating purity for baseline 1 results
        cls1=self.B1_results.where(self.B1_results["Predicted_cluster"]==1)
        cls2=self.B1_results.where(self.B1_results["Predicted_cluster"]==2)
        cls3=self.B1_results.where(self.B1_results["Predicted_cluster"]==3)
        cls4=self.B1_results.where(self.B1_results["Predicted_cluster"]==4)
        cls5=self.B1_results.where(self.B1_results["Predicted_cluster"]==5)
        
        c1=0
        c2=0
        c3=0
        c4=0
        c5=0
        for i in range(len(cls1)):
            if cls1["Real_cluster"][i]==1:
                c1+=1
            if cls1["Real_cluster"][i]==2:
                c2+=1
            if cls1["Real_cluster"][i]==3:
                c3+=1
            if cls1["Real_cluster"][i]==4:
                c4+=1
            if cls1["Real_cluster"][i]==5:
                c5+=1
        p1=max(c1,c2,c3,c4,c5)

        c1=0
        c2=0
        c3=0
        c4=0
        c5=0
        for i in range(len(cls2)):
            if cls2["Real_cluster"][i]==1:
                c1+=1
            if cls2["Real_cluster"][i]==2:
                c2+=1
            if cls2["Real_cluster"][i]==3:
                c3+=1
            if cls2["Real_cluster"][i]==4:
                c4+=1
            if cls2["Real_cluster"][i]==5:
                c5+=1
        p2=max(c1,c2,c3,c4,c5)

        c1=0
        c2=0
        c3=0
        c4=0
        c5=0
        for i in range(len(cls3)):
            if cls3["Real_cluster"][i]==1:
                c1+=1
            if cls3["Real_cluster"][i]==2:
                c2+=1
            if cls3["Real_cluster"][i]==3:
                c3+=1
            if cls3["Real_cluster"][i]==4:
                c4+=1
            if cls3["Real_cluster"][i]==5:
                c5+=1
        p3=max(c1,c2,c3,c4,c5)

        c1=0
        c2=0
        c3=0
        c4=0
        c5=0
        for i in range(len(cls4)):
            if cls4["Real_cluster"][i]==1:
                c1+=1
            if cls4["Real_cluster"][i]==2:
                c2+=1
            if cls4["Real_cluster"][i]==3:
                c3+=1
            if cls4["Real_cluster"][i]==4:
                c4+=1
            if cls4["Real_cluster"][i]==5:
                c5+=1
        p4=max(c1,c2,c3,c4,c5)

        c1=0
        c2=0
        c3=0
        c4=0
        c5=0
        for i in range(len(cls5)):
            if cls5["Real_cluster"][i]==1:
                c1+=1
            if cls5["Real_cluster"][i]==2:
                c2+=1
            if cls5["Real_cluster"][i]==3:
                c3+=1
            if cls5["Real_cluster"][i]==4:
                c4+=1
            if cls5["Real_cluster"][i]==5:
                c5+=1
        p5=max(c1,c2,c3,c4,c5)

        pure=(p1+p2+p3+p4+p5)/50
        print(pure)


c=Clustering()
c.tokenize_vector_docs()
c.word2vec_embedding()
c.vectorize()
c.Clustering_Baseline_1()
c.Clustering_Baseline_2()
c.Read_GT()
#print(c.B1_results)
#print(c.B2_results)
c.gen_dataframes()
c.Purity_B1()
c.Purity_B2()
import os
import re
from nltk.stem import PorterStemmer
import json
from pathlib import Path

class PreProcessor:
    
    #Inverted index
    II_Dictionary={}
    II_PostingList={}
    #Positional index
    PI_Dictionary={}
    PI_PostingList={}
    stopwords=[]
    docs = []
    Docs_Count=0
    collectionDIR="E:\IR_A1\IR_A1\DS1"
    DataDIR="E:\IR_A1\IR_A1\Data"
    
    def __init__(self):
        #Fetching stop words
        with open(os.path.join("E:\IR_A1\IR_A1", "Stopword-List.txt"), 'r') as f:
            self.stopwords += f.read().splitlines()
        f.close()
        while '' in self.stopwords: 
            self.stopwords.remove('')
        self.stopwords = [x.replace(' ','') for x in self.stopwords]
        return
    
    #lower case folding
    def Casefolding(self,text):
        return text.lower()
        
    def Tokenization(self,text):
        text = self.Casefolding(text)    
        text = re.sub(r'[^\w\s]',' ',text)
        text = text.split()
        return text

    def Stemming(self,token):
        ps = PorterStemmer()
        return ps.stem(token)
    
    def IsStopWord(self,token):
        return token in self.stopwords
    
    def BuildInvertedIndex(self):
        Docs = os.listdir(self.collectionDIR)
        D_Count=0
        Docs = [int(x.replace('.txt','')) for x in Docs]
        Docs.sort()
        Docs = [ str(x)+'.txt' for x in Docs ]

        for Doc in Docs:
            with open(os.path.join(self.collectionDIR+"\\", Doc), 'r') as f:
                text = f.read()
                tokens = self.Tokenization(text)
                self.docs.append(Doc) 

                for tk in tokens:
                    if not self.IsStopWord(tk):
                        tk = self.Stemming(tk)
                        if tk not in self.II_Dictionary:
                            self.II_PostingList[tk] = [D_Count]         
                            self.II_Dictionary[tk] = 1                     
                        
                        elif D_Count not in self.II_PostingList.get(tk):
                            self.II_PostingList[tk].append(D_Count)       
                            self.II_Dictionary[tk] += 1
                D_Count+=1
        return
    
    def BuildPositionalIndex(self):
        Docs = os.listdir(self.collectionDIR)
        D_Count=0
        Docs = [int(x.replace('.txt','')) for x in Docs]
        Docs.sort()
        Docs = [ str(x)+'.txt' for x in Docs ]
        
        for Doc in Docs:
            with open(os.path.join(self.collectionDIR+"\\", Doc), 'r') as f:
                text = f.read()
                tokens = self.Tokenization(text)
                #docs.append(Doc)
                i=0
                for tk in tokens:
                    if not self.IsStopWord(tk):
                        tk = self.Stemming(tk)
                        if tk not in self.PI_Dictionary:
                            self.PI_PostingList[tk] = {D_Count:[i]}         
                            self.PI_Dictionary[tk] = 1                     
                        
                        elif D_Count not in [*self.PI_PostingList[tk].keys()]:
                            self.PI_PostingList[tk][D_Count]=[i]      
                            self.PI_Dictionary[tk] += 1
                        else:
                            self.PI_PostingList[tk][D_Count].append(i)
                        i+=1 
                D_Count+=1
        return
    
    def PrintInvertedIndex(self):
        for i in self.II_Dictionary:
            print(i)
            print(self.II_PostingList[i])
        return
    
    def PrintPositionalIndex(self):
        for i in self.PI_Dictionary:
            print(i)
            for j in self.PI_PostingList[i]:
                print(self.PI_PostingList[i])
                print(j)
            print(self.PI_PostingList[i])
        return
    
    def WriteToDisk(self,index,indexType):
        filename = '\\' + indexType + ".txt"
        with open(self.DataDIR + filename, 'w') as filehandle:
            filehandle.write(json.dumps(index))
        return

    def ReadFromDisk(self,indexType):
        filename = '\\' + indexType + ".txt"
        with open(self.DataDIR + filename, 'r') as filehandle:
            index = json.loads(filehandle.read())
        return index
    
    def PreprocessingPipeline(self):

        #when inverted index and positional index is not build yet
        if not os.path.isdir(self.DataDIR):               

            #Creating inverted and positional index
            self.BuildInvertedIndex()
            self.BuildPositionalIndex()

            os.mkdir(self.DataDIR)                       

            #Saving Indexes to directory
            self.WriteToDisk(self.docs,'Documents')
            self.WriteToDisk(self.II_Dictionary,'II_Dictionary')
            self.WriteToDisk(self.II_PostingList,'Inverted_Index')
            self.WriteToDisk(self.PI_Dictionary,'PI_Dictionary')
            self.WriteToDisk(self.PI_PostingList,'Positional_Index')
        
        #when inverted index and positional index has been already build
        else:
            self.docs = self.ReadFromDisk('Documents')
            self.Docs_Count = len(self.docs)
            #Fetching Indexes from directory                     
            self.II_Dictionary = self.ReadFromDisk('II_Dictionary')
            self.II_PostingList = self.ReadFromDisk('Inverted_Index')
            self.PI_Dictionary =  self.ReadFromDisk('PI_Dictionary')
            self.PI_PostingList = self.ReadFromDisk('Positional_Index')
        
        return
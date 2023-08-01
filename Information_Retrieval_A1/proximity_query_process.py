import os
import re
from preprocessing import PreProcessor

class Proximity_Query:

    def __init__(self, PI_Dictionary, PI_Postings):
        self.Dictionary = PI_Dictionary
        self.Postings = PI_Postings
    

    def get_posting(self, term):
        if term in [*self.Postings.keys()]:
            return self.Postings[term]
        else:
            print(term + " Not in Vocablary")
            return None

    def get_positions(self, term, docID):
        return self.Postings[term][docID]

    # Merging proximity query Result
    def Posting_Merging(self, term1, term2, k):

        Result = []

        #Fetching posting list from Positional index
        PL1 = self.get_posting(term1)                       
        PL2 = self.get_posting(term2)

        if PL1 == None or PL2 == None:
            return [] 
        
        PL1 = list(PL1.keys())                            
        PL2 = list(PL2.keys())

        i = 0
        j = 0

        while i < len(PL1) and j < len(PL2):
            if int(PL1[i]) == int(PL2[j]):
                l = []              

                #Fetching positional list form a term in a document
                Pos1 = self.get_positions(term1, PL1[i])     
                Pos2 = self.get_positions(term2, PL2[j])

                indexPos1 = 0
                indexPos2 = 0

                while indexPos1 != len(Pos1):
                    while indexPos2 != len(Pos2):
                        if abs(Pos1[indexPos1] - Pos2[indexPos2]) <= k + 1:
                            l.append(Pos2[indexPos2])  
                        elif Pos2[indexPos2] > Pos1[indexPos1]:  
                            break
                        indexPos2 += 1 
        
                    while l != [] and abs(l[0] - Pos1[indexPos1]) > k + 1:  
                        l.remove(l[0])  
                    
                    for ps in l:
                        Result.append([PL1[i], Pos1[indexPos1], ps])                    
                    indexPos1 += 1
                
                i += 1
                j += 1
                
            elif int(PL1[i]) < int(PL2[j]):
                i += 1         
            else:   
                j += 1

        return Result

    def Process_Proximity_Query(self, query):
        
        tokens = query.split()      
        
        p = PreProcessor()

        #Stemming using port stemmer
        term1 = p.Stemming(tokens[0])                             
        term2 = p.Stemming(tokens[1])

        k = int(re.sub(r"[^\w\s]", "", tokens[2]))             
        result_set = self.Posting_Merging(term1, term2, k)

        ret_docs = {}
        for result in result_set:
            docNo = int(result[0]) + 1                         
            if docNo not in ret_docs:
                ret_docs[docNo] = [(result[1], result[2])]
            else:
                ret_docs[docNo].append((result[1], result[2]))

        return ret_docs



#p = PreProcessor()
#p.PreprocessingPipeline()
#px = Proximity_Query(p.PI_Dictionary, p.PI_PostingList)
#r=px.Process_Proximity_Query('australia william /2')
#print(r)

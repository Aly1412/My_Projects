from preprocessing import PreProcessor
from boolean_query_process import Boolean_Query
from proximity_query_process import Proximity_Query

class Query_Processor():
    
    def __init__(self):
        pass

    def ProcessQuery(self,query):
        if query != "":
                p = PreProcessor()
                #Processing for Boolean and Positional Index 
                p.PreprocessingPipeline()
                tokens = query.split()
                Processing_Cost = -1
                # For proximity queries
                if len(tokens)>2 and '/' == tokens[2][0]:
                    prox = Proximity_Query(p.PI_Dictionary,p.PI_PostingList)
                    Result = prox.Process_Proximity_Query(query)
                #For Boolean queries
                else:
                     b = Boolean_Query(p.II_Dictionary, p.II_PostingList, p.Docs_Count)
                     Result,Processing_Cost = b.Process_Boolean_Query(query)
                return (Result,Processing_Cost,query)

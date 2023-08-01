from Cosine_similarity import Similarity

class QueryProcessor():
    
    def __init__(self):
        pass

    def ProcessQuery(self,query):
        if query != "":                
            try:
                s = Similarity()
                result_set = s.process_query(query)
            except:
                return ["error",query,0.5]
            
            print(result_set)
            return (result_set,query,0.5)
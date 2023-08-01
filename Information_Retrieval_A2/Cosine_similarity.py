from Pre_processing import Preprocessor
import re
import math

class Similarity:

    def process_query(self,query):
        p = Preprocessor()
        p.Preprocessing_Pipeline()
        tokens = p.FilterTokens(query)                  # proprocessing query
        query_tf_index = self.BuildTfVector(tokens)     # term frequency vector for query
        query_tfidf_index = self.BuildTfIdfVector(query_tf_index,p.idf_index)   # tfidf vector for query
        return self.CosineSimilarity(p.noOfDocs,p.tfidf_index,query_tfidf_index)  

    def BuildTfVector(self,tokens):
        # query_tf_index = { t1 : 2, t2: 1, ... ,tn: 4}
        query_tf_index = {}
        #   calculating frquency count for terms in query
        for tok in tokens:
            if tok not in query_tf_index:
                query_tf_index[tok] = 1
            else:
                query_tf_index[tok] += 1

        #   calculating magnitude of query vector
        magnitude = 0
        for k in query_tf_index.keys():
            magnitude += query_tf_index[k] ** 2    
        magnitude = math.sqrt(magnitude)   

        for k in query_tf_index:
            query_tf_index[k] = query_tf_index[k] / magnitude

        return query_tf_index

    def BuildTfIdfVector(self,query_tf_index,idf_index):
        #   query_tfidf_index = { t1 : 2.4, t2: 0.01, ... ,tn: 0.234}
        query_tfidf_index = {}
        for k in query_tf_index.keys():
            print(k)
            idf = idf_index[k]
            tf = query_tf_index[k]
            query_tfidf_index[k] = tf * idf
    
        return query_tfidf_index
            
    def CosineSimilarity(self,noOfDocs,docs_tfidf_index,query_tfidf_index):
        # finds cosine similarity b/w query and every document in the collection
        sim_score = {}
        # cosine_sim(d,q) = (d . q) /  || d || . || q || =  d . q
        for i in range(noOfDocs):
            for key in query_tfidf_index.keys():
                if key in docs_tfidf_index[str(i)].keys():
                    if i not in sim_score.keys():
                        sim_score[i] = 0
                    sim_score[i] += (query_tfidf_index[key] * docs_tfidf_index[str(i)][key])
                    
        sim_score=sorted(sim_score.items(), key=lambda x:x[1], reverse=True)

        print(sim_score)

        result = []
        for k in range(len(sim_score)):
            if sim_score[k][1] >= 0.005:
                result.append(sim_score[k][0])
        result = [x+1 for x in result]

        return result

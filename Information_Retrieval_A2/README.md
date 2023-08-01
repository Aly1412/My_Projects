# IR_VSM_A2

##   Preprocessing Chain
    
  Parse document
    
  Noise and Puntuations Removal
    
  Stopword removal

  Lemmatization

  Build TF index     
  
    {doc1 : { t1 : 3, t2: 4, ... ,tn: 5}, doc1 : { t1 : 2, t2: 1, ... ,tn: 4}, ... , docN : { t1 : 1, t2: 4, ... ,tn: 2} )  
       
  Length Normalization of Term Fequency
    
  Build IDF index     
    
    { t1: idf-Val, t2: idf-Val , t3: idf-Val , ... , t4: idf-val }
    
  Build TF-IDF Index  
    
    {doc1 : { t1 : 0.21, t2: 2.4, ... ,tn: 0.11}, doc1 : { t1 : 2.4, t2: 0.01, ... ,tn: 0.234}, ... , docN : { t1 : 0.21, t2: 0.344, ... ,tn: 0.2})

  Store All three indexes

##  Query Time - Run Time Processing

  Parse query
    
  Calculate TF vector for query 
  
    { t1 : 2, t2: 1, ... ,tn: 4}
    
  Calculate TF-IDF for query 
  
    { t1 : 2.4, t2: 0.01, ... ,tn: 0.234}

  Calculate cosine similarity of query and each document(488)
    
    cos(d,q) = d . q   [Because frequency is length normalized] 
    
  Optimization - only multiply tf-idf of term common in query and document

  Filter documents based on alpha=0.0001

## Dependencies

    pip install flask
  
## Run
  
    app.py

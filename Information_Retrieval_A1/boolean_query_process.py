from preprocessing import PreProcessor

class Boolean_Query:
    
    def __init__(self, II_Dictionary, II_Postings, Docs_Count):
        self.Postings = II_Postings
        self.Dictionary = II_Dictionary
        self.t_Docs = Docs_Count

    def get_posting(self, term):
        if term in [*self.Postings.keys()]:
            return self.Postings[term]
        else:
            print(term + " Not in Vocablary")
            return []

    def get_posting_size(self, term):
        if term in [*self.Dictionary.keys()]:
            return self.Dictionary[term]
        else:
            print(term + " not in vocablary")
            return -1

    #Intersection method
    def Merging_AND_Operation(self, PL1, PL2):
        result = []
        i = 0
        j = 0
        while i < len(PL1) and j < len(PL2):
            if PL1[i] == PL2[j]:         
                result.append(PL1[i])
                i += 1
                j += 1

            elif PL1[i] < PL2[j]:
                i += 1
            else:
                j += 1
        return result

    #Union method
    def Merging_OR_Operation(self, PL1, PL2):
        result = []
        i = 0
        j = 0

        while i < len(PL1) and j < len(PL2):
            if PL1[i] == PL2[j]:
                result.append(PL1[i])
                i += 1
                j += 1
            elif PL1[i] < PL2[j]:
                result.append(PL1[i])
                i += 1
            else:
                result.append(PL2[j])
                j += 1

        while i < len(PL1):
            result.append(PL1[i])
            i += 1

        while j < len(PL2):
            result.append(PL2[j])
            j += 1
        return result

    #Complement method
    def NOT_Operation(self, PL):
        result = []
        for i in range(self.t_Docs):         
            if i not in PL:
                result.append(i)
        return result

    def Process_Boolean_Query(self, query):

        processingCost = 0

        p = PreProcessor()
                                       
        tokens = query.split() 
          
        for i in range(len(tokens)): 
            tokens[i] = p.Stemming(tokens[i])

        #Single term query
        if (len(tokens)) == 1:
            processingCost = 0
            Intermediate_Result = self.get_posting(tokens[0])

        #Complement term simple query
        elif (len(tokens)) == 2 and tokens[0].upper() == "NOT": 
            processingCost = self.t_Docs - self.get_posting_size(tokens[1])
            Intermediate_Result = self.NOT_Operation(self.get_posting(tokens[1]))

        #For complex queries
        else:
            i = 0
            Intermediate_Result = None

            while i < len(tokens):

                if tokens[i].upper() == "AND":
                    #Initial query execution for the first time
                    if Intermediate_Result is None:
                        #AND operation with NOT-W1
                        if i - 2 >= 0 and tokens[i - 2] == "NOT":
                            p1 = self.NOT_Operation(self.get_posting(tokens[i - 1]))
                        #AND operation without NOT-W1
                        else:
                            p1 = self.get_posting(tokens[i - 1])

                    #Query execution with intermediate result
                    #AND operation with NOT-W1
                    if tokens[i + 1] == "NOT" and i + 2 < len(tokens):
                        p2 = self.NOT_Operation(self.get_posting(tokens[i + 2]))
                        i += 2
                    #AND operation without NOT-W1
                    else:
                        p2 = self.get_posting(tokens[i + 1])
                        i += 1

                    if Intermediate_Result is None:
                        Intermediate_Result = p1

                    processingCost += min(len(Intermediate_Result), len(p2))
                    Intermediate_Result = self.Merging_AND_Operation(Intermediate_Result, p2)

                elif tokens[i].upper() == "OR":
                    #Initial query execution for the first time
                    if Intermediate_Result is None:
                        #OR operation with NOT-W1
                        if i - 2 >= 0 and tokens[i - 2] == "NOT":
                            p1 = self.NOT_Operation(self.get_posting(tokens[i - 1]))
                        #OR operation without NOT-W1
                        else:
                            p1 = self.get_posting(tokens[i - 1])

                    #Query execution with intermediate result
                    #OR operation with NOT-W1
                    if tokens[i + 1] == "NOT" and i + 2 < len(tokens):
                        p2 = self.NOT_Operation(self.get_posting(tokens[i + 2]))
                        i += 2
                    #OR operation without NOT-W1
                    else:
                        p2 = self.get_posting(tokens[i + 1])
                        i += 1

                    if Intermediate_Result is None:
                        Intermediate_Result = p1

                    processingCost += len(Intermediate_Result) + len(p2)
                    Intermediate_Result = self.Merging_OR_Operation(Intermediate_Result, p2)
                i += 1

        result = [i + 1 for i in Intermediate_Result]
        return (result, processingCost)

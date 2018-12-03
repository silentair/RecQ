'''the basic class of an RS'''
class Recommender_Base(object):
    def __init__(self):
        self.config = None
        self.dao = None
        self.name = None
        self.type = None

    def showRecommenderInfo(self):
        if self.config is None:
            print('current recommender has not been instantiated!')
        else:
            print('Algorithm name: '+self.name)
            print('Algorithm type: '+self.type)
            self.config.showConfig()

    def Training(self):
        pass

    def Testing(self):
        pass

    def __getRealOnValidation__(self):
        pass

    def __getPredOnValidation__(self,**params):
        pass

class Recommender_Rating(Recommender_Base):
    def __init__(self):
        super(Recommender_Rating,self).__init__()
        self.r_upper = None
        self.r_lower = None

        self.mae = None
        self.rmae = None


    def __ratingConfine__(self,rating):
        if self.r_upper is None or self.r_lower is None:
            print('paramter(s) rating upper bound or(and) lower bound missing!')
            raise ValueError

        if rating < self.r_lower:
            return self.r_lower
        elif rating > self.r_upper:
            return self.r_upper
        else:
            return rating
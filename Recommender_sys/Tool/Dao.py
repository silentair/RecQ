from Tool.File import FileIO
from Tool.Math4r import Normalize,Denormalize
import numpy as np
import random

class RatingDataset(object):
    def __init__(self,data):
        self.data = data
        self.datalen = len(data)

        self.users = []
        self.items = []
        self.user_num = 0 # user number
        self.item_num = 0 # item number

        self.user2id = {}
        self.item2id = {}
        self.id2user = {} # No.x user has name 'xx'
        self.id2item = {} # No.x item has name 'xx'

        self.users_RatedItem = {}   # users who rated item i and rating r
        self.items_RatedByUser = {} # items which rated by user u and rating r

        self.user_means = {} # mean values of users' ratings
        self.item_means = {} # mean values of items' ratings
        self.global_mean = 0

        self.__getUserItemInfo__()
        self.__getRatingInfo__()
        self.__getMeanInfo__()

    def __getUserItemInfo__(self):
        for uir in self.data:
            user = uir[0]
            item = uir[1]
            if user not in self.users:
                self.users.append(user)

            if item not in self.items:
                self.items.append(item)

        self.user_num = len(self.users)
        self.item_num = len(self.items)

        for n,u in enumerate(self.users):
            self.user2id[u] = n
            self.id2user[n] = u
        for n,i in enumerate(self.items):
            self.item2id[i] = n
            self.id2item[n] = i

    def __getRatingInfo__(self):
        for uir in self.data:
            tempDict = self.items_RatedByUser.get(uir[0],{})
            #if tempDict.__contains__(uir[1]):
                #print('user '+uir[0]+' has rated item '+uir[1]+' before, please check your data!')
                #raise KeyError
            tempDict[uir[1]] = float(uir[2])
            self.items_RatedByUser[uir[0]] = tempDict

            tempDict = self.users_RatedItem.get(uir[1],{})
            #if tempDict.__contains__(uir[0]):
                #print('item '+uir[1]+' has been rated by users '+uir[0]+' before, please check your data!')
                #raise KeyError
            tempDict[uir[0]] = float(uir[2])
            self.users_RatedItem[uir[1]] = tempDict

    def __getMeanInfo__(self):
        total = 0
        leng = 0
        for user in self.users:
            total_u2i = sum([v for v in self.items_RatedByUser[user].values()])
            leng_u2i = len(self.items_RatedByUser[user])

            if leng_u2i == 0:
                self.user_means[user] = 0
            else:
                self.user_means[user] = total_u2i / leng_u2i

        for item in self.items:
            total_i2u = sum([v for v in self.users_RatedItem[item].values()])
            leng_i2u = len(self.users_RatedItem[item])

            if leng_u2i == 0:
                self.item_means[item] = 0
            else:
                self.item_means[item] = total_i2u / leng_i2u

            total = total + total_i2u
            leng = leng + leng_i2u

        self.global_mean = total / leng

    def generateMatrix(self):
        ui_matrix = np.zeros([self.user_num,self.item_num]).tolist()
        is_rating = np.zeros([self.user_num,self.item_num]).tolist()
        for u,ir in self.items_RatedByUser.items():
            for i,r in ir.items():
                u_id = self.user2id[u]
                i_id = self.item2id[i]
                ui_matrix[u_id][i_id] = r
                is_rating[u_id][i_id] = 1

        return ui_matrix,is_rating

    # Notice: 
    # Normalize and Denormalize are only used in models,
    # not in Dao
    def generateNormalizedDateset(self,method='min-max'):
        if method == 'min-max':
            ratings = np.array([ float(data[2]) for data in self.data ])
            maxVal = np.max(ratings)
            minVal = np.min(ratings)
            normalized_rating = Normalize(ratings,maxVal,minVal,method=method)

            normalized_data = []
            for rating,data in zip(normalized_rating,self.data):
                data[-1] = str(rating)
                normalized_data.append(data)
            return RatingDataset(normalized_data),maxVal,minVal
        elif method == 'z-score':   
            ratings = np.array([ float(data[2]) for data in self.data ])
            mean = np.mean(ratings)
            var = np.var(ratings)
            normalized_rating = Normalize(ratings,mean,var,method=method)
            
            normalized_data = []
            for rating,data in zip(normalized_rating,self.data):
                data[-1] = str(rating)
                normalized_data.append(data)
            return RatingDataset(normalized_data),mean,var
        else:
            print('please check your choose the correct normalization method')
            raise ValueError

    def generateDenormalizedDateset(self,param1,param2,method='min-max'):
        if method == 'min-max':
            ratings = np.array([ float(data[2]) for data in self.data ])
            maxVal = param1
            minVal = param2
            denormalized_rating = Denormalize(ratings,maxVal,minVal,method=method)

            denormalized_data = []
            for rating,data in zip(normalized_rating,self.data):
                data[-1] = str(rating)
                denormalized_data.append(data)
            return RatingDataset(denormalized_data),maxVal,minVal
        elif method == 'z-score':     
            ratings = np.array([ float(data[2]) for data in self.data ])
            mean = np.mean(ratings)
            var = np.var(ratings)
            denormalized_rating = Denormalize(ratings,mean,var,method=method)
            
            denormalized_data = []
            for rating,data in zip(denormalized_rating,self.data):
                data[-1] = str(rating)
                denormalized_data.append(data)
            return RatingDataset(denormalized_data),mean,var
        else:
            print('please check your choose the correct normalization method')
            raise ValueError

class RationgDao(object):
    def __init__(self,config):
        self.data = FileIO.loadData(config)
        self.normalized = False
        try:
            if config.getParam('shuffle') == 'y':
                random.shuffle(self.data)
        except KeyError:
            pass
        try:
            if config.getParam('normalized') == 'y':
                self.normalized = True
        except KeyError:
            pass

        try:
            self.norm_method = config.getParam('norm_method')
        except KeyError:
            self.norm_method = 'min-max'

        try:
            self.batch_size = int(config.getParam('batch_size'))
        except KeyError:
            self.batch_size = 1024

        self.trainingSet = None
        self.testingSet = None
        self.validationSet = None

        self.users = []
        self.items = []
        self.user_num = 0
        self.item_num = 0

        self.user2id = {}
        self.item2id = {}
        self.id2user = {} # No.x user has name 'xx'
        self.id2item = {} # No.x item has name 'xx'

        self.__getUserItemInfo__()
        self.__getDataset__(config)

    def __getUserItemInfo__(self):
        for uir in self.data:
            user = uir[0]
            item = uir[1]
            if user not in self.users:
                self.users.append(user)
            if item not in self.items:
                self.items.append(item)

        self.user_num = len(self.users)
        self.item_num = len(self.items)

        for n,u in enumerate(self.users):
            self.user2id[u] = n
            self.id2user[n] = u
        for n,i in enumerate(self.items):
            self.item2id[i] = n
            self.id2item[n] = i

    def __getDataset__(self,config):
        perct_tra = float(config.getParam('training'))
        perct_tes = float(config.getParam('testing'))
        perct_val = float(config.getParam('validation'))

        if perct_tra+perct_tes > 1 or perct_tra <= 0 or perct_tes <= 0:
            print('please assure you split the data correctly!')
            raise ArithmeticError

        print('spliting data: ',perct_tra,'for training  '
                               ,perct_tes,'for testing  '
                               ,perct_val,'for validation')

        data_len = len(self.data)
        threshold1 = data_len*perct_tra
        threshold2 = threshold1+data_len*perct_tes

        trainingData = []
        testingData = []
        validationData = []

        for i in range(data_len):
            if i < threshold1:
                trainingData.append(self.data[i])
            elif i < threshold2:
                testingData.append(self.data[i])
            else:
                validationData.append(self.data[i])

        if trainingData != []:
            self.trainingSet = RatingDataset(trainingData)
        if testingData != []:
            self.testingSet = RatingDataset(testingData)
        if validationData != []:
            self.validationSet = RatingDataset(validationData)


    def generateBatches(self):
        if self.batch_size == 0:
            print('batch size is zero, no need to generate batch!')
            raise ValueError

        batches = []
        batch_idx = 0
        while batch_idx != -1:
            # out of range
            if batch_idx + self.batch_size > self.trainingSet.datalen:
                batch = self.trainingSet.data[batch_idx : ]
                batch_idx = -1
            else:
                batch = self.trainingSet.data[batch_idx : batch_idx + self.batch_size]
                batch_idx = batch_idx + self.batch_size

            batches.append(RatingDataset(batch))

        return batches
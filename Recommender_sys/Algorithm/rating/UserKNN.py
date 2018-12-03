'''User-based K-nearest Neighbors'''
import tensorflow as tf
import numpy as np
import time
from Tool.Math4r import MAE,RMAE,Denormalize,Similarity
from Tool.File import FileIO
from Recommender.basicalRS import Recommender_Base

class UserKnn(Recommender_Base):
    def __init__(self,config,dao):
        super(UserKnn,self).__init__()

        self.name = 'User_based_K_nearest_Neighbors'
        self.type = 'rating_based'
        print('initializing algorithm '+self.name+'...')
        self.config = config
        self.dao = dao
        try:
            self.neighbor_num = int(config.getParam('neighbor_num'))
            self.sim_method = config.getParam('sim_method')
            self.save_model = config.getParam('save_model')
            self.save_result = config.getParam('save_result')
            self.save_path = config.getParam('save_path')
        except KeyError:
            print('missing parameters please check you have set all parameters for algorithm '+self.name)
            raise KeyError

        if dao.normalized:
            self.trainingSet,self.normlized_param1,self.normlized_param2 = dao.trainingSet.generateNormalizedDateset(dao.norm_method)
        else:
            self.trainingSet = dao.trainingSet
        self.testingSet = dao.testingSet
        self.validationSet = dao.validationSet # generally, no need

    def Training(self):
        print('begin training algorithm '+self.name+'...')

        start = time.clock()
        self.__getUsersSimility__()
        end = time.clock()

        print('algorithm '+self.name+' training complete, using time',end-start,'s')

        if self.save_model == 'y':
            print('saving model...')
            self.__saveModel__()

    def Testing(self):
        self.usersSimilarity = self.__loadModel__('2018-11-28_20-23-25/User_based_K_nearest_Neighbors.txt')
        test_pred = np.zeros([self.testingSet.user_num,self.testingSet.item_num])

        for user in self.testingSet.users:
            uid = self.testingSet.user2id[user]
            if not self.usersSimilarity.__contains__(user):  # user has no neighor
                topNn = []
            else:
                topN = sorted(self.usersSimilarity[user].items(), key = lambda x:x[1],reverse=True)[:self.neighbor_num]  # neighbor = topN[i][0], similarity = topN[i][1]
                topNn = [topi[0] for topi in topN]
            
            for item in self.testingSet.items_RatedByUser[user].keys():
                iid = self.testingSet.item2id[item]
                if not self.trainingSet.users_RatedItem.__contains__(item):
                    test_pred[uid][iid] = self.__chooseMean__(user,item)
                    continue

                cor_users = list(set(self.trainingSet.users_RatedItem[item]) & set(topNn))
                if cor_users == []:
                    test_pred[uid][iid] = self.__chooseMean__(user,item)
                else:
                    sim_sum = 0
                    rating_sum = 0
                    for nei in cor_users:
                        sim_sum = sim_sum + self.usersSimilarity[user][nei]
                        rating_sum = rating_sum + self.usersSimilarity[user][nei]*self.trainingSet.items_RatedByUser[nei][item]
                    if sim_sum == 0:
                        test_pred[uid][iid] = 0
                    else:
                        test_pred[uid][iid] = rating_sum / sim_sum
        self.test_real,self.test_is_rating = self.testingSet.generateMatrix()

        if self.dao.normalized:
            test_pred = Denormalize(test_pred,self.normlized_param1,self.normlized_param2,self.dao.norm_method)
        
        self.test_pred = test_pred.tolist()

        real_result = np.array(self.test_real).flatten().tolist()
        pred_result = np.array(self.test_pred).flatten().tolist()
        is_rating = np.array(self.test_is_rating).flatten().tolist()

        pred = []
        real = []
        for i in range(len(is_rating)):
            if is_rating[i] == 1:
                pred.append(pred_result[i])
                real.append(real_result[i])
        print('testing complete')

        self.mae = MAE(np.array(pred),np.array(real))
        self.rmae = RMAE(np.array(pred),np.array(real))
        print('the result of testing:')
        print('MAE:',self.mae,'RMAE:',self.rmae)

        if self.save_result == 'y':
            print('saving testing result...')
            self.__saveTestResults__()

    def __chooseMean__(self,user,item):
        if self.trainingSet.item_means.__contains__(item):
            return self.trainingSet.item_means[item]
        elif self.trainingSet.user_means.__contains__(user):
            return self.trainingSet.user_means[user]
        else:
            return self.self.trainingSet.global_mean

    def __saveTestResults__(self):
        header = 'user\t'+'item\t'+'real\t'+'pred\n'
        content=[]
        content.append(header)
        for user in self.testingSet.users:
            for item in self.testingSet.items:
                test_uid = self.testingSet.user2id[user]
                test_iid = self.testingSet.item2id[item]

                if self.test_is_rating[test_uid][test_iid] == 1:
                    real = str(self.test_real[test_uid][test_iid])
                    pred = str(self.test_pred[test_uid][test_iid])
                    line = user+'\t'+item+'\t'+real+'\t'+pred+'\n'
                    content.append(line)
        content.append('MAE: '+str(self.mae)+'\nRMAE:'+str(self.rmae))

        tm = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
        txt = self.name+'_'+tm+'.txt'
        FileIO.writeFile(self.save_path+'results/',txt,content,'a')

    # noticed: if dictionary usersCoRatedItems does not have key user_i then user_i has zero neighbor
    def __getUsersSimility__(self):
        self.usersSimilarity = {} 
        for user_all in self.dao.users:
            print('calculate simility between user '+user_all+' and his neighobrs...')
            try:
                user_allRatedItems = [item for item in self.trainingSet.items_RatedByUser[user_all].keys()]
            except KeyError:
                continue

            all_to_tra = self.usersSimilarity.get(user_all,{})
            for user_tra in self.trainingSet.users:
                if user_tra == user_all:
                    continue
                if all_to_tra.__contains__(user_tra):
                    continue

                user_traRatedItems = [item for item in self.trainingSet.items_RatedByUser[user_tra].keys()]
                cor_items = list(set(user_allRatedItems) & set(user_traRatedItems))  # items rated by both user_all and user_tra
                '''
                # not deal with cold-start problem yet
                if len(cor_items) < 3:
                    continue
                '''
                if cor_items == []:
                    continue

                vector_all = np.array([self.trainingSet.items_RatedByUser[user_all][i] for i in cor_items])
                vector_tra = np.array([self.trainingSet.items_RatedByUser[user_tra][i] for i in cor_items])
                sim = Similarity(vector_all,vector_tra,self.sim_method)

                all_to_tra[user_tra] = sim
                self.usersSimilarity[user_all] = all_to_tra
                
                tra_to_all = self.usersSimilarity.get(user_tra,{})
                tra_to_all[user_all] = sim
                self.usersSimilarity[user_tra] = tra_to_all

    def __saveModel__(self,):
        tm = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))+'/'
        file_name = self.name+'.txt'
        FileIO.writeFile(self.save_path+'models/'+tm,file_name,str(self.usersSimilarity))

    def __loadModel__(self,name):
        model_path = self.save_path+'models/' + name
        if not FileIO.fileExists(model_path):
            print('file '+name+' does not exist!')
            raise IOError
        with open(model_path,'r') as f:
            content = f.read()
            usersSimilarity = eval(content)
        return usersSimilarity
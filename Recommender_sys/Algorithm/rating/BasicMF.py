'''Basic Matrix Factorization'''
import tensorflow as tf
import numpy as np
import time
from Tool.Math4r import MAE,RMAE,Denormalize
from Tool.File import FileIO
from Recommender.basicalRS import Recommender_Rating

class BasicMatrixFactorization(Recommender_Rating):
    def __init__(self,config,dao):
        super(BasicMatrixFactorization,self).__init__()

        self.name = 'Basic_Matrix_Factorization'
        self.type = 'rating_based'
        print('initializing algorithm '+self.name+'...')
        self.config = config
        self.dao = dao
        try:
            self.r_upper = float(config.getParam('ratingUpper'))
            self.r_lower = float(config.getParam('ratingLower'))
            self.factor_num = int(config.getParam('factor_num'))
            self.iteration_num = int(config.getParam('iteration_num'))
            self.regU = float(config.getParam('regU'))
            self.regV = float(config.getParam('regV'))
            self.lr = float(config.getParam('lr'))
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
        self.validationSet = dao.validationSet
        print('initializing complete')

    def Training(self):
        print('begin training algorithm '+self.name+'...')
        user_num = self.dao.user_num
        item_num = self.dao.item_num

        U = tf.Variable(tf.random_normal(shape=[user_num,self.factor_num],dtype=tf.float32),name='user_vectors')
        V = tf.Variable(tf.random_normal(shape=[item_num,self.factor_num],dtype=tf.float32),name='item_vectors')

        batch_userids = tf.placeholder(tf.int32,shape=[None])
        batch_itemids = tf.placeholder(tf.int32,shape=[None])
        ui_matrix = tf.placeholder(tf.float32)
        is_rating = tf.placeholder(tf.float32)

        U_train = tf.nn.embedding_lookup(U,batch_userids)
        V_train = tf.nn.embedding_lookup(V,batch_itemids)

        pred_rating = tf.matmul(U_train,tf.transpose(V_train))

        loss = tf.nn.l2_loss((ui_matrix - pred_rating)*is_rating)
        loss = loss + tf.multiply(self.regU,tf.nn.l2_loss(U))
        loss = loss + tf.multiply(self.regV,tf.nn.l2_loss(V))

        optimzer = tf.train.AdamOptimizer(self.lr).minimize(loss)

        saver = tf.train.Saver(max_to_keep=3)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # with validation
            if self.validationSet is not None:
                self.__getRealOnValidation__()

            start = time.clock()
            batches = self.dao.generateBatches()
            for step in range(self.iteration_num):
                for batch in batches:
                    bat_u = sorted([self.dao.user2id[u] for u in batch.users])
                    bat_i = sorted([self.dao.item2id[i] for i in batch.items])
                    ui_mat,is_rat = batch.generateMatrix()

                    sess.run(optimzer,feed_dict= { batch_userids:bat_u, batch_itemids:bat_i, ui_matrix:ui_mat, is_rating:is_rat })

                loss_ = sess.run(loss,feed_dict= { batch_userids:bat_u, batch_itemids:bat_i, ui_matrix:ui_mat, is_rating:is_rat })
                print('step: ',step+1,'/',self.iteration_num)
                print('loss:',loss_)
                # with validation
                if self.validationSet is not None:
                    self.__getPredOnValidation__(U,V)
                    mae = MAE(np.array(self.valid_real),np.array(self.valid_pred))
                    rmae = RMAE(np.array(self.valid_real),np.array(self.valid_pred))
                    print('MAE: ',mae,'  RMAE: ',rmae)
                print('\n')
            end = time.clock()

            print('algorithm '+self.name+' training complete, using time',end-start,'s')
            
            self.resU = sess.run(U)
            self.resV = sess.run(V)

            if self.save_model == 'y':
                tm = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))+'/'
                save_path = self.save_path+'models/'+tm+self.name+'.ckpt'
                print('saving model...')
                saver.save(sess,save_path)

    def Testing(self):
        print('begin testing algorithm '+self.name+'...')
        U = tf.constant(self.resU)
        V = tf.constant(self.resV)

        testing_userids = [self.dao.user2id[u] for u in self.testingSet.users]
        testing_itemids = [self.dao.item2id[i] for i in self.testingSet.items]

        U_test = tf.nn.embedding_lookup(U,testing_userids)
        V_test = tf.nn.embedding_lookup(V,testing_itemids)

        test_real,test_is_rating = self.testingSet.generateMatrix()
        tf_real = tf.constant(test_real)
        tf_is_rating = tf.constant(test_is_rating)
        tf_pred = tf.matmul(U_test,tf.transpose(V_test))

        with tf.Session() as sess:
            self.test_pred = sess.run(tf_pred)
            self.test_real = sess.run(tf_real)
            self.test_is_rating = sess.run(tf_is_rating)
        
        if self.dao.normalized:
            self.test_pred = Denormalize(self.test_pred,self.normlized_param1,self.normlized_param2,self.dao.norm_method)

        pred_result = self.test_pred.flatten().tolist()
        real_result = self.test_real.flatten().tolist()
        is_rating = self.test_is_rating.flatten().tolist()
        pred = []
        real = []
        for i in range(len(is_rating)):
            if is_rating[i] == 1:
                pred_confined = self.__ratingConfine__(pred_result[i])
                pred.append(pred_confined)
                real.append(real_result[i])
        print('testing complete')

        self.mae = MAE(np.array(pred),np.array(real))
        self.rmae = RMAE(np.array(pred),np.array(real))
        print('the result of testing:')
        print('MAE:',self.mae,'RMAE:',self.rmae)

        if self.save_result == 'y':
            print('saving testing result...')
            self.__saveTestResults__()

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
                    pred_confined = self.__ratingConfine__(self.test_pred[test_uid][test_iid])
                    pred = str(pred_confined)
                    line = user+'\t'+item+'\t'+real+'\t'+pred+'\n'
                    content.append(line)
        content.append('MAE: '+str(self.mae)+'\nRMAE:'+str(self.rmae))

        tm = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
        txt = self.name+'_'+tm+'.txt'
        FileIO.writeFile(self.save_path+'results/',txt,content,'a')

    def __getRealOnValidation__(self):
        self.validation_userids = sorted([self.dao.user2id[u] for u in self.validationSet.users])
        self.validation_itemids = sorted([self.dao.item2id[i] for i in self.validationSet.items])

        valid_real,valid_is_rating = self.validationSet.generateMatrix()
        tf_real = tf.constant(valid_real)

        with tf.Session() as sess:
            real_result = sess.run(tf_real)

        real_result = real_result.flatten().tolist()
        self.valid_is_rating = np.array(valid_is_rating).flatten().tolist()

        self.valid_real = []
        for i in range(len(self.valid_is_rating)):
            if self.valid_is_rating[i] == 1:
                self.valid_real.append(real_result[i])

    def __getPredOnValidation__(self,tf_u,tf_v):
        U_valid = tf.nn.embedding_lookup(tf_u,self.validation_userids)
        V_valid = tf.nn.embedding_lookup(tf_v,self.validation_itemids)

        valid_pred = tf.matmul(U_valid,tf.transpose(V_valid))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer()) # for tf_u and tf_v
            pred_result = sess.run(valid_pred)
        
        if self.dao.normalized:
            pred_result = Denormalize(pred_result,self.normlized_param1,self.normlized_param2,self.dao.norm_method)

        pred_result = pred_result.flatten().tolist()

        self.valid_pred = []
        for i in range(len(self.valid_is_rating)):
            if self.valid_is_rating[i] == 1:
                pred_confined = self.__ratingConfine__(pred_result[i])
                self.valid_pred.append(pred_confined)
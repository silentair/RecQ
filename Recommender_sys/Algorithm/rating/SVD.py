'''Singular Value Decomposition'''
import tensorflow as tf
import numpy as np
import time
from Tool.Math4r import MAE,RMAE,Denormalize
from Tool.File import FileIO
from Recommender.basicalRS import Recommender_Base

class SingularValueDecomposition(Recommender_Base):
    def __init__(self,config,dao):
        super(SingularValueDecomposition,self).__init__()

        self.name = 'Singular_Value_Decomposition'
        self.type = 'rating_based'
        print('initializing algorithm '+self.name+'...')
        self.config = config
        self.dao = dao
        try:
            self.threshhold = float(config.getParam('threshhold'))
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

        self.__decompose__()
        self.__getFactorNum__()
        print('initializing complete')

    def __decompose__(self):
        ui_matrix,_ = self.trainingSet.generateMatrix()
        # svd
        self.u,self.s,self.v = np.linalg.svd(ui_matrix)

    def __getFactorNum__(self):
        sigma = self.s
        sigma_pow = sigma**2
        threshhold = np.sum(sigma_pow) * self.threshhold
        # dimension to reduction
        sigma_sum_k = 0
        k = 0
        for i in sigma:
            sigma_sum_k = sigma_sum_k + i**2
            k = k + 1
            if sigma_sum_k >= threshhold:
                break
        self.k = k

    def Training(self):
        print('begin training algorithm '+self.name+'...')
        user_num = self.dao.user_num
        item_num = self.dao.item_num

        U = tf.Variable(tf.random_normal(shape = [user_num, self.k]))
        V = tf.Variable(tf.random_normal(shape = [item_num, self.k]))
        U_bias = tf.Variable(tf.random_normal(shape = [user_num]))
        V_bias = tf.Variable(tf.random_normal(shape = [item_num]))
    
        batch_userids = tf.placeholder(tf.int32,shape=[None])
        batch_itemids = tf.placeholder(tf.int32,shape=[None])
        ui_matrix = tf.placeholder(tf.float32)
        is_rating = tf.placeholder(tf.float32)
        global_mean = tf.placeholder(tf.float32)
    
        U_train = tf.nn.embedding_lookup(U, batch_userids)
        V_train = tf.nn.embedding_lookup(V, batch_itemids)
        U_bias_train = tf.nn.embedding_lookup(U_bias, batch_userids)
        V_bias_train = tf.nn.embedding_lookup(V_bias, batch_itemids)
    
        pred_rating = tf.matmul(U_train, tf.transpose(V_train))
        pred_rating = tf.add(tf.transpose(pred_rating), U_bias_train)
        pred_rating = tf.add(tf.transpose(pred_rating), V_bias_train)
        pred_rating = tf.add(pred_rating, global_mean)
    
        loss_rat = tf.nn.l2_loss((ui_matrix - pred_rating)*is_rating)
        loss_reg_u = tf.multiply(self.regU,tf.nn.l2_loss(U)) + tf.multiply(self.regU,tf.nn.l2_loss(U_bias))
        loss_reg_v = tf.multiply(self.regV,tf.nn.l2_loss(V)) + tf.multiply(self.regV,tf.nn.l2_loss(V_bias))
    
        loss = loss_rat + loss_reg_u + loss_reg_v
    
        optimizer_U = tf.train.AdamOptimizer(self.lr).minimize(loss,var_list=[U,U_bias])
        optimizer_V = tf.train.AdamOptimizer(self.lr).minimize(loss,var_list=[V,V_bias])
    
        saver = tf.train.Saver(max_to_keep=3)

        # with validation
        if self.validationSet is not None:
            self.__getRealOnValidation__()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            start = time.clock()
            batches = self.dao.generateBatches()
            for step in range(self.iteration_num):
                for batch in batches:
                    bat_u = sorted([self.dao.user2id[u] for u in batch.users])
                    bat_i = sorted([self.dao.item2id[i] for i in batch.items])
                    ui_mat,is_rat = batch.generateMatrix()
                    glb_mean = batch.global_mean

                    sess.run(optimizer_U, feed_dict={ batch_userids:bat_u, batch_itemids:bat_i, ui_matrix:ui_mat, is_rating:is_rat, global_mean:glb_mean })
                    sess.run(optimizer_V, feed_dict={ batch_userids:bat_u, batch_itemids:bat_i, ui_matrix:ui_mat, is_rating:is_rat, global_mean:glb_mean })
    
                loss_ = sess.run(loss,feed_dict= { batch_userids:bat_u, batch_itemids:bat_i, ui_matrix:ui_mat, is_rating:is_rat, global_mean:glb_mean })
                print('step: ',step+1,'/',self.iteration_num)
                print('loss:',loss_)
    
                # with validation
                if self.validationSet is not None:
                    self.__getPredOnValidation__(U,V,U_bias,V_bias)
                    mae = MAE(np.array(self.valid_real),np.array(self.valid_pred))
                    rmae = RMAE(np.array(self.valid_real),np.array(self.valid_pred))
                    print('MAE: ',mae,'  RMAE: ',rmae)
                print('\n')
            end = time.clock()

            print('algorithm '+self.name+' training complete, using time',end-start,'s')
            
            self.resU = sess.run(U)
            self.resV = sess.run(V)
            self.resU_bias = sess.run(U_bias)
            self.resV_bias = sess.run(V_bias)

            if self.save_model == 'y':
                tm = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))+'/'
                save_path = self.save_path+'models/'+tm+self.name+'.ckpt'
                print('saving model...')
                saver.save(sess,save_path)

    def Testing(self):
        print('begin testing algorithm '+self.name+'...')
        U = tf.constant(self.resU)
        V = tf.constant(self.resV)
        U_bias = tf.constant(self.resU_bias)
        V_bias = tf.constant(self.resV_bias)

        testing_userids = sorted([self.dao.user2id[u] for u in self.dao.testingSet.users])
        testing_itemids = sorted([self.dao.item2id[i] for i in self.dao.testingSet.items])

        U_test = tf.nn.embedding_lookup(U,testing_userids)
        V_test = tf.nn.embedding_lookup(V,testing_itemids)
        U_bias_test = tf.nn.embedding_lookup(U_bias,testing_userids)
        V_bias_test = tf.nn.embedding_lookup(V_bias,testing_itemids)

        test_real,test_is_rating = self.testingSet.generateMatrix()
        tf_real = tf.constant(test_real)
        tf_is_rating = tf.constant(test_is_rating)
        tf_pred = tf.matmul(U_test,tf.transpose(V_test))
        tf_pred = tf.add(tf.transpose(tf_pred), U_bias_test)
        tf_pred = tf.add(tf.transpose(tf_pred), V_bias_test)
        tf_pred = tf.add(tf_pred, self.testingSet.global_mean)

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

    def __getPredOnValidation__(self,tf_u,tf_v,tf_u_b,tf_v_b):
        U_valid = tf.nn.embedding_lookup(tf_u,self.validation_userids)
        V_valid = tf.nn.embedding_lookup(tf_v,self.validation_itemids)
        U_bias_valid = tf.nn.embedding_lookup(tf_u_b,self.validation_userids)
        V_bias_valid = tf.nn.embedding_lookup(tf_v_b,self.validation_itemids)

        valid_pred = tf.matmul(U_valid, tf.transpose(V_valid))
        valid_pred = tf.add(tf.transpose(valid_pred), U_bias_valid)
        valid_pred = tf.add(tf.transpose(valid_pred), V_bias_valid)
        valid_pred = tf.add(valid_pred, self.validationSet.global_mean)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer()) # for tf_u and tf_v
            pred_result = sess.run(valid_pred)
        
        if self.dao.normalized:
            pred_result = Denormalize(pred_result,self.normlized_param1,self.normlized_param2,self.dao.norm_method)

        pred_result = pred_result.flatten().tolist()

        self.valid_pred = []
        for i in range(len(self.valid_is_rating)):
            if self.valid_is_rating[i] == 1:
                self.valid_pred.append(pred_result[i])
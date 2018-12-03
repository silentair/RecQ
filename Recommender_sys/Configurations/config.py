import os
class Config(object):
    def __init__(self,path):
        self.config = {}
        self.__getConfig__(path)

    def __getConfig__(self,path):
        if not os.path.exists(path):
            print('config file does not exist!')
            raise IOError
        
        print('Reading config file from ' + path)

        with open(path) as f:
            for idx,line in enumerate(f):
                if line.strip() != '':
                    param,value = line.strip().split('=')
                    if param == '' or value == '':
                        print('config file does not in the correct format! Error Line:',idx)
                        raise ValueError
                    self.config[param] = value

    def hasKey(self,key):
        return self.config.__contains__(key)

    def getParam(self,key):
        if not self.hasKey(key):
            print('parameter ' + key + ' does not exist')
            raise KeyError

        return self.config[key]

    def showConfig(self):
        if self.config:
            print('here is you configuration:')
            for k,v in self.config.items():
                print(k+': '+v)
        else:
            print('configuration does not set yet!')
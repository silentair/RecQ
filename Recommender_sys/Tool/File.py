import os

class FileIO(object):
    def __init__(self):
        pass

    @staticmethod
    def fileExists(path):
        if os.path.exists(path):
            return True
        else:
            return False

    @staticmethod
    def writeFile(path,name,content,op = 'w'):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path+name,op) as f:
            f.writelines(content)

    @staticmethod
    def deleteFile(path):
        if os.path.exists(path):
            remove(path)

    @staticmethod
    def loadData(config):
        path = config.getParam('data_path')
        if not os.path.exists(path):
            print('data does not exist!')
            raise IOError

        try:
            order = config.getParam('column_order').strip().split()
            if sorted(order) != [ str(i) for i in range(len(order)) ]:
                print('parameter column_order is in an incorrect format!')
                raise ValueError
        except KeyError:
            order = [0,1,2]

        print('loading data...')

        with open(path) as f:
            row_data = f.read().split()

        tup_len = len(order)
        data_len = len(row_data) // tup_len
        data = []
        for i in range(data_len):
                data.append([ row_data[i*tup_len+int(o)] for o in order ])

        return data
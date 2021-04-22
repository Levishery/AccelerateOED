import numpy as np
import json
with open('../Dataset/5o_type1.out') as reader:
    data_ = []
    for i in range(192):
        a = reader.readline()
    for i in range(47896):
        data_dic = {}
        w = reader.readline()
        if w[-7] != ']':
            w = w[:-1] + reader.readline()
        w = w[:-6]
        w = np.fromstring(w[1:-1], dtype=np.float, sep=' ')
        data_dic['w'] = w.tolist()
        if len(w) != 5:
            raise Exception("error reading w.")

        # a_upper = np.zeros([7, 7])
        a_upper = np.zeros([5, 5])
        # for j in range(7):
        for j in range(5):
            a = reader.readline()
            if a[-2] != ']' and a[-2] != 'r':
                a = a[:-1] + reader.readline()
            if j == 6:
                a = a[:-6]
            else:
                a = a[:-1]
            a = a[1:]
            a_upper[j, :] = np.fromstring(a[1:-1], dtype=np.float, sep=' ')
        data_dic['a_upper'] = a_upper.tolist()

        # a_lower = np.zeros([7, 7])
        a_lower = np.zeros([5, 5])
        # for j in range(7):
        for j in range(5):
            a = reader.readline()
            if a[-2] != ']' and a[-2] != 'r':
                a = a[:-1] + reader.readline()
            if j == 6:
                a = a[:-6]
            else:
                a = a[:-1]
            a = a[1:]
            a_lower[j, :] = np.fromstring(a[1:-1], dtype=np.float, sep=' ')
        data_dic['a_lower'] = a_lower.tolist()

        for j in range(5):
            if a_upper[j][j] != 0 or a_lower[j][j] != 0:
                raise Exception("error reading a.")

        for j in range(3):
            MOCU1 = reader.readline()
            if MOCU1[0] == 'M':
                MOCU1 = float(MOCU1[14:-1])
                break
        data_dic['MOCU1'] = MOCU1
        MOCU2 = float(reader.readline()[14:-1])
        data_dic['MOCU2'] = MOCU2
        data_dic['mean_MOCU'] = (MOCU1 + MOCU2)/2
        data_.append(data_dic)
        a = reader.readline()
    jsObj = json.dumps(data_)
    fileObject = open('../Dataset/5o_type1.json', 'w')
    fileObject.write(jsObj)
    fileObject.close()










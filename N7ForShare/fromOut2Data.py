import numpy as np
import json
with open('make_data2.out') as reader:
    data_ = []
    for i in range(998):
        a = reader.readline()
    for i in range(10000):
        data_dic = {}
        w = reader.readline()
        w = w[:-1] + reader.readline()
        w = w[:-6]
        w = np.fromstring(w[1:-1], dtype=np.float, sep=' ')
        data_dic['w'] = w.tolist()

        a_upper = np.zeros([7, 7])
        for j in range(7):
            a = reader.readline()
            a = a[:-1] + reader.readline()
            if j == 6:
                a = a[:-6]
            else:
                a = a[:-1]
            a = a[1:]
            a_upper[j, :] = np.fromstring(a[1:-1], dtype=np.float, sep=' ')
        data_dic['a_upper'] = a_upper.tolist()

        a_lower = np.zeros([7, 7])
        for j in range(7):
            a = reader.readline()
            a = a[:-1] + reader.readline()
            if j == 6:
                a = a[:-6]
            else:
                a = a[:-1]
            a = a[1:]
            a_lower[j, :] = np.fromstring(a[1:-1], dtype=np.float, sep=' ')
        data_dic['a_lower'] = a_lower.tolist()

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
    fileObject = open('data_7_oscillators_5.json', 'w')
    fileObject.write(jsObj)
    fileObject.close()










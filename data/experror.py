import demjson

fileObject = open('data/data_7_oscillators.json', 'r')
data2 = fileObject.read()
data2 = demjson.decode(data2)

fileObject = open('data/data_7_oscillators_1.json', 'r')
data1 = fileObject.read()
data1 = demjson.decode(data1)

data = data1 + data2

SSE = 0
for i in range(len(data)):
    x = (data[i]['MOCU1'] - data[i]['mean_MOCU'])^2
    SSE = SSE + x
MSE = SSE/len(data)

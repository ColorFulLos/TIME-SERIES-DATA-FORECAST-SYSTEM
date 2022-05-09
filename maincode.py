import pandas as pd
from numpy import *
import matplotlib.pyplot as plt

timecount=0
def loadDataSet(file):#读取训练集函数

    dataLabel = []
    row, col = file.shape
    dataMatrix = file.drop([file.columns[col - 1]], axis=1)#将因变量删掉，组成自变量集合
    for i in range(row):
         dataLabel.append(float(file.iloc[i,col-1]))#组成因变量集合
    matLabel = mat(dataLabel).transpose()
    return dataMatrix, matLabel,file

def loadExamSet(file,index):#测试机切割函数 index为读取的测试集个数
    dataLabel = []
    row, col = file.shape
    file2 = file.drop(labels = range(0,row-int(index)),axis=0)#删除训练集部分
    row, col = file2.shape
    for i in range(row):
         dataLabel.append(float(file2.iloc[i,col-1]))
    dataMatrix = file2.drop([file2.columns[col - 1]], axis=1)
    return dataMatrix,dataLabel,file2

def loadExamSet2(file):#给预测集添加因变量（只是为了符合拟合函数输入格式，无具体用途）
    dataLabel = []
    row, col = file.shape
    for i in range(row):
         dataLabel.append(0)#赋值全0
    file1=column_stack((file,dataLabel))
    return file,dataLabel,file1

def loadExamSet3(file):#给预测集添加因变量（只是为了符合拟合函数输入格式，无具体用途）
    dataLabel = []
    row, col = file.shape
    for i in range(row):
         dataLabel.append(i)#赋值为i
    file1=column_stack((file,dataLabel))
    return file,dataLabel,file1

def Solving(file):#将二维二分型设置二次拟合自变量集
    dataLabel = []
    dataMatrix2 = []

    row,col=file.shape

    for i in range(row):
        dataLabel.append(float(file.iloc[i, col - 1]))
    matLabel = mat(dataLabel).transpose()

    for i in range(row):#将X1,X2拓展成X1,X2,X1*X1,X2*X2,X1*X2
        dataMatrix2.append([float(file.iloc[i,0]),float(file.iloc[i,1]),float(file.iloc[i,0]*file.iloc[i,0]),float(file.iloc[i,1]*file.iloc[i,1]),float(file.iloc[i,0])*file.iloc[i,1]])
    return dataMatrix2, matLabel, file

def Solvingtest(file):#预测集的二次拟合计算，具体计算方法同上
    dataLabel = []
    dataMatrix2 = []

    row,col=file.shape

    for i in range(row):
        dataLabel.append(float(file[i, col - 1]))
    matLabel = mat(dataLabel).transpose()

    for i in range(row):
        dataMatrix2.append([float(file[i,0]),float(file[i,1]),float(file[i,0]*file[i,0]),float(file[i,1]*file[i,1]),float(file[i,0])*file[i,1]])
    return dataMatrix2, matLabel, file

def Solving2(dataMatrix,matLabel):#分组处理函数，将输入的自变量集和因变量集合进行四等分
    dataMatrix2 = column_stack((dataMatrix, matLabel))
    dataMatrix2 = dataMatrix2[lexsort(dataMatrix2.T)]
    dataMatrix2 = dataMatrix2[0,:,:]
    row,col=dataMatrix2.shape
    dataLabel3=[]
    dataLabel4 = []
    dataLabel5 = []
    dataLabel6 = []
    a=int(row/4)
    b=int(row/2)
    c=int(row*3/4)
    dataMatrix6 = delete(dataMatrix2,range(0,c),axis=0)
    dataMatrixtemp = delete(dataMatrix2,range(0,b),axis=0)
    dataMatrix5 = delete(dataMatrixtemp,range(c-b,row-b),axis=0)
    dataMatrixtemp2 = delete(dataMatrix2,range(b,row),axis=0)
    dataMatrix4 = delete(dataMatrixtemp2,range(0,a),axis=0)
    dataMatrix3 = delete(dataMatrix2,range(a,row),axis=0)
    row3,col3=dataMatrix3.shape
    for i in range(row3):
        dataLabel3.append(float(dataMatrix3[i,col3-1]))
    dataLabel3= mat(dataLabel3).transpose()
    dataMatrix3 = delete(dataMatrix3,col3-1,axis=1)
    row4,col4 = dataMatrix4.shape
    for i in range(row4):
        dataLabel4.append(float(dataMatrix4[i,col4-1]))
    dataLabel4 = mat(dataLabel4).transpose()
    dataMatrix4 = delete(dataMatrix4, col4 - 1, axis=1)
    row5, col5 = dataMatrix5.shape
    for i in range(row5):
        dataLabel5.append(float(dataMatrix5[i,col5-1]))
    dataLabel5 = mat(dataLabel5).transpose()
    dataMatrix5 = delete(dataMatrix5, col5 - 1, axis=1)
    row6, col6 = dataMatrix6.shape
    for i in range(row6):
        dataLabel6.append(float(dataMatrix6[i,col6-1]))
    dataLabel6 = mat(dataLabel6).transpose()
    dataMatrix6 = delete(dataMatrix6, col6 - 1, axis=1)
    return dataMatrix3,dataMatrix4,dataMatrix5,dataMatrix6,dataLabel3,dataLabel4,dataLabel5,dataLabel6

def Solving3(flag,matLabel):#边界提取函数，flag为输入的数据组属于哪个区间，matlabel是因变量集合
    result=[]
    row, col = matLabel.shape
    w = [0 for x in range(0, row)]
    for i in range(row - 1):
        w[i] = matLabel[i, 0]
    matLabel2 = pd.qcut(w, [0,0.0004,0.25,0.5,0.75,1])#将因变量分为4组（0-0.0004不算一组）
    if(flag==1):
         x=matLabel2[int(row/5)].left#提取第一区间左边界，下同
         result.append(x)
         y=matLabel2[int(row/5)].right#提取第一区间右边界，下同
         result.append(y)
    elif (flag== 2):
         x=matLabel2[int(row/3)].left
         result.append(x)
         y=matLabel2[int(row/3)].right
         result.append(y)
    elif  (flag== 3):
         x = matLabel2[int(row*2/3)].left
         result.append(x)
         y = matLabel2[int(row * 2 / 3)].right
         result.append(y)
    else:
        x = matLabel2[int(row*9/10)+1].left
        result.append(x)
        y = matLabel2[int(row * 9 / 10) + 1].right
        result.append(y)
    result=array(result)#将结果变为array格式
    result=result.reshape(-1,2)#转置
    return result

def BoxCut(matLabel):#连续型变量的分组因变量置0置1函数

    row,col=matLabel.shape
    w = pd.Series([0 for x in range(0, row)])
    for i in range(row-1):
        w[i]=matLabel[i,0]
    matLabel2=pd.qcut(w.rank(method='first'),[0,0.25,1],labels=[1,0])#将因变量进行分割，然后将其中的0-25%设置为1，75%为0。
    matLabel3=pd.qcut(w.rank(method='first'),[0,0.25,0.5,1],labels=[2,1,0])#26-50%为1
    row=len(matLabel3)
    for i in range(row):
        if (matLabel3[i]==2):
            matLabel3[i]=0
    matLabel4=pd.qcut(w.rank(method='first'),[0,0.5,0.75,1],labels=[2,1,0])#51-75%为1
    row = len(matLabel4)
    for i in range(row):
        if (matLabel4[i] == 2):
            matLabel4[i] = 0
    matLabel5=pd.qcut(w.rank(method='first'),[0,0.75,1],labels=[0,1])#76%-100%为1
    return matLabel2,matLabel3,matLabel4,matLabel5#返回分别有四分之一数据为1，其余为0的四个因变量数据集

def stocGraAscent1(dataMatrix, matLabel):#逻辑回归计算函数
    w={}
    matMatrix = mat(dataMatrix)
    from sklearn.linear_model import LogisticRegression
    LR = LogisticRegression( random_state=0, max_iter=5000)#最大拟合次数5000，随机数种子为0，其余为默认
    LR.fit(matMatrix,matLabel)
    y_predict = LR.predict(matMatrix)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(matLabel, y_predict)#计算准确率
    print(accuracy)
    LR.intercept_  # 截距
    row,col=LR.coef_.shape
    theta0 = LR.intercept_
    for i in range(col):
        w[i+1]=LR.coef_[0][i]

    w[0]=theta0

    return w,accuracy

def stocGraAscent2(dataMatrix, matLabel):#线性回归计算函数
    w = {}
    matMatrix = mat(dataMatrix)
    from sklearn.linear_model import LinearRegression
    LR_multi = LinearRegression()  # 建立模型
    LR_multi.fit(matMatrix, matLabel)  # 训练模型
    y_predict_multi = LR_multi.predict(matMatrix)
    from sklearn.metrics import r2_score
    r2_score_multi = r2_score(matLabel, y_predict_multi)
    print(r2_score_multi)
    LR_multi.intercept_  # 截距
    row, col = LR_multi.coef_.shape
    theta0 = LR_multi.intercept_
    for i in range(col):
        w[i + 1] = LR_multi.coef_[0][i]

    w[0] = theta0
    return w

def stocGraAscent3(dataMatrix, matLabel):#逻辑回归计算函数2（用于逻辑回归计算函数准确率偏低的时候）
    w={}
    matMatrix = mat(dataMatrix)
    from sklearn.linear_model import LogisticRegression
    LR = LogisticRegression( random_state=0, max_iter=5000,solver='liblinear',penalty='l1')#与1不同的是，penalty设置为L1正则化，损失函数迭代器为liblinear
    LR.fit(matMatrix,matLabel)
    y_predict = LR.predict(matMatrix)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(matLabel, y_predict)
    print(accuracy)
    LR.intercept_  # 截距
    row,col=LR.coef_.shape
    theta0 = LR.intercept_
    for i in range(col):
        w[0] = theta0
        w[i+1]=LR.coef_[0][i]

    w[0]=theta0

    return w,accuracy

def draw(weight,file):#训练集散点图绘画函数
    x0List = []
    y0List = []
    x1List = []
    y1List = []
    minvalue=file.iloc[0,0]
    maxvalue=file.iloc[0,0]
    row, col = file.shape
    for i in range(row):
        if minvalue>= file.iloc[i,0]:
            minvalue=file.iloc[i,0]
        if maxvalue <= file.iloc[i, 0]:
            maxvalue = file.iloc[i, 0]
        if file.iloc[i,2]== 0:
            x0List.append(float(file.iloc[i,0]))
            y0List.append(float(file.iloc[i,1]))
        else:
            x1List.append(float(file.iloc[i,0]))
            y1List.append(float(file.iloc[i,1]))


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x0List, y0List, s=10, c='red')#用不同的颜色将两种点分开
    ax.scatter(x1List, y1List, s=10, c='green')

    xList = []
    yList = []

    x = arange(minvalue-1, maxvalue+1, 0.1)#x轴范围
    for i in arange(len(x)):
        xList.append(x[i])

    y = (-weight[0] - weight[1] * x) / weight[2]#y的取值计算

    for j in arange(y.shape[0]):
        yList.append(y[j])

    ax.plot(xList, yList)
    plt.xlabel('x1')
    plt.ylabel('x2')
    # plt.show()
    plt.savefig("temp.png")

def draw1(weight, file):#二次拟合训练集散点图绘画函数
        x0List = []
        y0List = []
        x1List = []
        y1List = []
        minvalue = file.iloc[0, 0]
        maxvalue = file.iloc[0, 0]
        for i in range(len(file)):
            if minvalue >= file.iloc[i, 0]:
                minvalue = file.iloc[i, 0]
            if maxvalue <= file.iloc[i, 0]:
                maxvalue = file.iloc[i, 0]
            if file.iloc[i, 2] == 0:
                x0List.append(float(file.iloc[i, 0]))
                y0List.append(float(file.iloc[i, 1]))
            else:
                x1List.append(float(file.iloc[i, 0]))
                y1List.append(float(file.iloc[i, 1]))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x0List, y0List, s=10, c='red')
        ax.scatter(x1List, y1List, s=10, c='green')

        xList = []
        yList = []

        x = arange(minvalue - 1, maxvalue + 1, 0.01)

        for i in arange(len(x)):
            xList.append(x[i])
            a = weight[4]
            b = weight[5] * x[i] + weight[2]
            d = x[i] * x[i]
            c = weight[0] + weight[1] * x[i] + weight[3] * d
            y = (-b + sqrt(abs(b * b - 4 * a * c))) / (2 * a)#计算y值
            yList.append(y)

        ax.plot(xList, yList)
        plt.xlabel('x1')
        plt.ylabel('x2')
        #plt.show()
        plt.savefig("temp.png")

def Draw2(file,ww):#折线图绘画函数
    fig = plt.figure()
    row,col=file.shape
    row1,col1=ww.shape
    x1 = arange(0,row,1)
    y1 = file.iloc[:,col - 1]
    x2 = arange(row-row1,row,1)
    y2 = ww
    plt.plot(x1, y1,linewidth=3)#用不同粗细的线分开训练集数据和测试集数据
    plt.plot(x2, y2,linewidth=0.5)
    #plt.show()
    plt.savefig("temp.png")

def drawtest(weight,file):#预测集散点图绘画
    x0List = []
    y0List = []
    x1List = []
    y1List = []
    minvalue=file[0,0]
    maxvalue=file[0,0]
    row, col = file.shape
    for i in range(row):
        if minvalue>= file[i,0]:
            minvalue=file[i,0]
        if maxvalue <= file[i, 0]:
            maxvalue = file[i, 0]
        if file[i,2]== 0:
            x0List.append(float(file[i,0]))
            y0List.append(float(file[i,1]))
        else:
            x1List.append(float(file[i,0]))
            y1List.append(float(file[i,1]))


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x0List, y0List, s=10, c='red')
    ax.scatter(x1List, y1List, s=10, c='green')

    xList = []
    yList = []

    x = arange(minvalue-1, maxvalue+1, 0.1)
    for i in arange(len(x)):
        xList.append(x[i])

    y = (-weight[0] - weight[1] * x) / weight[2]

    for j in arange(y.shape[0]):
        yList.append(y[j])

    ax.plot(xList, yList)
    plt.xlabel('x1')
    plt.ylabel('x2')
    # plt.show()
    plt.savefig("temp1.png")

def draw1test(weight, file):#二次拟合预测集散点图绘画
        x0List = []
        y0List = []
        x1List = []
        y1List = []
        minvalue = file[0, 0]
        maxvalue = file[0, 0]
        row, col = file.shape
        for i in range(row):
            if minvalue >= file[i, 0]:
                minvalue = file[i, 0]
            if maxvalue <= file[i, 0]:
                maxvalue = file[i, 0]
            if file[i, 2] == 0:
                x0List.append(float(file[i, 0]))
                y0List.append(float(file[i, 1]))
            else:
                x1List.append(float(file[i, 0]))
                y1List.append(float(file[i, 1]))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x0List, y0List, s=10, c='red')
        ax.scatter(x1List, y1List, s=10, c='green')

        xList = []
        yList = []

        x = arange(minvalue - 1, maxvalue + 1, 0.01)

        for i in arange(len(x)):
            xList.append(x[i])
            a = weight[4]
            b = weight[5] * x[i] + weight[2]
            d = x[i] * x[i]
            c = weight[0] + weight[1] * x[i] + weight[3] * d
            y = (-b + sqrt(abs(b * b - 4 * a * c))) / (2 * a)
            yList.append(y)

        ax.plot(xList, yList)
        plt.xlabel('x1')
        plt.ylabel('x2')
        # plt.show()
        plt.savefig("temp1.png")

def Draw2test(file,ww):#预测集折线图绘画

        fig = plt.figure()
        row1, col1 = ww.shape
        x1 = arange(0, row1, 1)
        y1 = ww
        plt.plot(x1, y1, linewidth=2)
        # plt.show()
        plt.savefig("temp1.png")

def Test(dataMatrix,weight,matLabel):#二分型数据测试集计算函数
    w={}

    matMatrix = mat(dataMatrix)
    m,n = matMatrix.shape
    wei = mat(zeros((n, 1)))
    ww = mat(zeros((m, 1)))

    for count in range(len(weight)-1):
        wei[count,0]=weight[count+1]
    for i in range(m):
        w[i]=(dot(matMatrix[i],wei)) + weight[0]
        ww[i]=w[i]
        if w[i]>=0.5:#阈值分类
            ww[i]=1
        else:
            ww[i]=0
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(matLabel, ww)
    print(accuracy)
    dataMatrix=column_stack((dataMatrix,matLabel,ww))

    return dataMatrix,accuracy

def Test2(dataMatrix,weight,matLabel):#连续型数据线性回归测试集计算函数
    w={}

    matMatrix = mat(dataMatrix)
    m,n = matMatrix.shape
    wei = mat(zeros((n, 1)))
    ww = mat(zeros((m, 1)))
    for count in range(len(weight)-1):
        wei[count,0]=weight[count+1]
    for i in range(m):
        w[i]=(dot(matMatrix[i],wei)) + weight[0]
        ww[i]=w[i]
    from sklearn.metrics import mean_squared_error, r2_score
    mean_squared_error_multi = mean_squared_error(matLabel, ww)
    print(mean_squared_error_multi)
    dataMatrix=column_stack((dataMatrix,matLabel,ww))

    return dataMatrix,ww

def Test3(dataMatrix,weight1,weight2,weight3,weight4,matLabel):#判断连续性数据属于哪个区间的函数
    w1 = {}
    w2 = {}
    w3 = {}
    w4 = {}
    matMatrix = mat(dataMatrix)
    m, n = matMatrix.shape
    wei1 = mat(zeros((n, 1)))
    wei2 = mat(zeros((n, 1)))
    wei3 = mat(zeros((n, 1)))
    wei4 = mat(zeros((n, 1)))
    ww = mat(zeros((m, 1)))
    www = mat(zeros((4, 1)))

    for count in range(len(weight1)-1):
        wei1[count, 0] = weight1[count + 1]
        wei2[count, 0] = weight2[count + 1]
        wei3[count, 0] = weight3[count + 1]
        wei4[count, 0] = weight4[count + 1]
    for i in range(m):
        w1[i] = (dot(matMatrix[i], wei1)) + weight1[0]
        w2[i] = (dot(matMatrix[i], wei2)) + weight2[0]
        w3[i] = (dot(matMatrix[i], wei3)) + weight3[0]
        w4[i] = (dot(matMatrix[i], wei4)) + weight4[0]
        www[0]=w1[i]
        www[1]=w2[i]
        www[2]=w3[i]
        www[3]=w4[i]
        if(max(www)==w1[i]):
            ww[i]=1
        elif(max(www)==w2[i]):
            ww[i]=2
        elif(max(www)==w3[i]):
            ww[i]=3
        else:
            ww[i]=4
    dataMatrix = column_stack((dataMatrix, matLabel, ww))

    return dataMatrix, ww

# if __name__ == '__main__':
#         # data=pd.read_csv('seattleWeather.csv')
#         # dataMatrix, matLabel, file, row = loadDataSet(data)
#         # weight = stocGraAscent1(dataMatrix, matLabel)
#         # #draw(weight, file)
#         # testdata,matLabel2,file2=loadExamSet(data,row)
#         # finalmatrix=Test(testdata,weight,matLabel2)
#         # print(finalmatrix)
#         #
#         # dataMatrix2,matLabel3,file3 = Solving(data)
#         # weight2 = stocGraAscent1(dataMatrix2,matLabel3)
#         # dataMatrix3,matLabel4,file4 = Solving(file2)
#         # finalmatrix2=Test(dataMatrix3,weight2,matLabel4)
#         # print(finalmatrix2)
#         # draw1(weight2,file3)
#
#         data = pd.read_csv('weather.csv')
#
#         dataMatrix, matLabel, file, row = loadDataSet(data)
#         weight = stocGraAscent2(dataMatrix, matLabel)
#         testdata, matLabel2, file2 = loadExamSet(data, row)
#         finalmatrix,ww = Test2(testdata, weight, matLabel2)
#         print(finalmatrix)
#         Draw2(data,ww)
#
#         # data = pd.read_csv('weather.csv')
#         # dataMatrix, matLabel, file, row = loadDataSet(data)
#         # matLabel2,matLabel3,matLabel4,matLabel5=BoxCut(matLabel)
#         # weight1 = stocGraAscent1(dataMatrix, matLabel2.astype('int'))
#         # weight2 = stocGraAscent1(dataMatrix, matLabel3.astype('int'))
#         # weight3 = stocGraAscent1(dataMatrix, matLabel4.astype('int'))
#         # weight4 = stocGraAscent1(dataMatrix, matLabel5.astype('int'))
#         # testdata, matLabeltest, file2 = loadExamSet(data, row)
#         # finalmatrix,ww = Test3(testdata, weight1,weight2,weight3,weight4, matLabeltest)
#         # Matrix1,Matrix2,Matrix3,Matrix4,Label1,Label2,Label3,Label4=Solving2(dataMatrix,matLabel)
#         # matLabel11, matLabel12, matLabel13, matLabel14 = BoxCut(Label1)
#         # weight11 = stocGraAscent1(Matrix1, matLabel11.astype('int'))
#         # weight12 = stocGraAscent1(Matrix1, matLabel12.astype('int'))
#         # weight13 = stocGraAscent1(Matrix1, matLabel13.astype('int'))
#         # weight14 = stocGraAscent1(Matrix1, matLabel14.astype('int'))
#         # matLabel21, matLabel22, matLabel23, matLabel24 = BoxCut(Label2)
#         # weight21 = stocGraAscent1(Matrix2, matLabel21.astype('int'))
#         # weight22 = stocGraAscent1(Matrix2, matLabel22.astype('int'))
#         # weight23 = stocGraAscent1(Matrix2, matLabel23.astype('int'))
#         # weight24 = stocGraAscent1(Matrix2, matLabel24.astype('int'))
#         # matLabel31, matLabel32, matLabel33, matLabel34 = BoxCut(Label3)
#         # weight31 = stocGraAscent1(Matrix3, matLabel31.astype('int'))
#         # weight32 = stocGraAscent1(Matrix3, matLabel32.astype('int'))
#         # weight33 = stocGraAscent1(Matrix3, matLabel33.astype('int'))
#         # weight34 = stocGraAscent1(Matrix3, matLabel34.astype('int'))
#         # matLabel41, matLabel42, matLabel43, matLabel44 = BoxCut(Label4)
#         # weight41 = stocGraAscent1(Matrix4, matLabel41.astype('int'))
#         # weight42 = stocGraAscent1(Matrix4, matLabel42.astype('int'))
#         # weight43 = stocGraAscent1(Matrix4, matLabel43.astype('int'))
#         # weight44 = stocGraAscent1(Matrix4, matLabel44.astype('int'))
#         # result={}
#         # count=len(ww)
#         # flag1=0
#         # flag2=0
#         # flag3=0
#         # flag4=0
#         # for i in range(count):
#         #    if(ww[i]==1):
#         #        Matrix,www=Test3(testdata[i:i+1].values,weight11,weight12,weight13,weight14,matLabeltest[i])
#         #        result[i]=Solving3(www,Label1)
#         #        flag1=flag1+1
#         #    elif (ww[i] == 2):
#         #          Matrix,www = Test3(testdata[i:i+1].values, weight21, weight22, weight23, weight24, matLabeltest[i])
#         #          flag2=flag2+1
#         #          result[i] = Solving3(www, Label2)
#         #    elif (ww[i] == 3):
#         #          Matrix,www = Test3(testdata[i:i+1].values, weight31, weight32, weight33, weight34, matLabeltest[i])
#         #          flag3=flag3+1
#         #          result[i] = Solving3(www, Label3)
#         #    else:
#         #        Matrix,www = Test3(testdata[i:i+1].values, weight41, weight42, weight43, weight44, matLabeltest[i])
#         #        flag4 = flag4 + 1
#         #        result[i] = Solving3(www, Label4)
#         # print(result)

from tkinter import *
import time
import tkinter.filedialog
from tkinter.simpledialog import *
from typing import Any, Union
from tkinter import scrolledtext
import maincode
import pandas as pd
from numpy import *
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
set_printoptions(threshold=inf) # threshold
set_printoptions(suppress=True)
root= Tk()#封面设计
root.title('时序数据预测系统')
root.geometry('900x600')
lb = Label(root,text='欢迎使用时序数据预测系统',\
        bg='#d3fbfb',\
        fg='black',\
        font=('华文新魏',32),\
        width=25,\
        height=2,\
        relief=RIDGE)#封面抬头
lb.pack()


def gettime():#获取时间函数
      timestr = time.strftime("%H:%M:%S") # 获取当前的时间并转化为字符串
      lb2.configure(text=timestr)   # 重新设置标签文本
      root.after(1000,gettime) # 每隔1s调用函数 gettime 自身获取时间
lb2 =Label(root,text='',fg='black',font=("黑体",20),relief=RIDGE)
lb2.pack()
gettime()
def xz():#训练集选择函数
    global filename1
    global file1
    global count
    global col
    filename1=tkinter.filedialog.askopenfilename()
    if filename1 != '':
         lb3.config(text='您选择的训练集文件是'+filename1)
    else:
         lb3.config(text='您没有选择任何文件')
    file1 = pd.read_csv(filename1)
    dataMatrix1, matLabel1, file = maincode.loadDataSet(file1)
    count, col = dataMatrix1.shape
    print(filename1)

def xz2():#预测集选择函数
    global filename2
    global file2
    filename2=tkinter.filedialog.askopenfilename()
    if filename2 != '':
         lb5.config(text='您选择的预测集文件是'+filename2)
    else:
         lb5.config(text='您没有选择任何文件')
    file2 = pd.read_csv(filename2)
    print(filename2)

lb3 = Label(root,text='')
lb3.pack()
btn=Button(root,text='选择您的训练数据库',fg='black',font=("黑体",15),relief=RIDGE,command=xz)
btn.pack()
lb5 = Label(root,text='')
lb5.pack()
btn2=Button(root,text='选择您的预测数据库', fg='black',font=("黑体",15),relief=RIDGE,command=xz2)
btn2.pack()

global flag
flag=3
global count2
count2=0
global ttype
ttype=0
global flag1
flag1=3


# def Mysel():
#     global flag
#     dic = {0: '二维二分型', 1: '多维二分型', 2: '多维连续型'}
#     s = "您选了" + dic.get(var.get()) + "项"
#     lb4.config(text=s)
#     flag=var.get()
# lb4 = Label(root)
# lb4.pack()
# var = IntVar()
# rd1 = Radiobutton(root, text="二维二分型", fg='black',font=("黑体",20),relief=RIDGE,variable=var, value=0, command=Mysel)
# rd1.bind('<1>',Mysel())
# rd1.pack()
# rd2 = Radiobutton(root, text="多维二分型", fg='black',font=("黑体",20),relief=RIDGE,variable=var, value=1, command=Mysel)
# rd2.bind('<1>',Mysel())
# rd2.pack()
# rd3 = Radiobutton(root, text="多维连续型", fg='black',font=("黑体",20),relief=RIDGE,variable=var, value=2, command=Mysel)
# rd3.bind('<1>',Mysel())
# rd3.pack()

# def finalflag(flag):
#     global flag1
#     if flag==0:
#         flag1=0
#     elif flag==1:
#           flag1=1
#     else:flag1=2

def newwindow3():#预测集展示窗口
    global file
    global file2
    dataMatrix1, matLabel1, file = maincode.loadDataSet(file1)
    count, col = dataMatrix1.shape
    if col == 2:  # 数据集的类型自动判断
        if matLabel1[0] == 0 or matLabel1[0] == 1:
            flag1 = 0
        else:
            flag1 = 2
    else:
        if matLabel1[0] == 0 or matLabel1[0] == 1:
            flag1 = 1
        else:
            flag1 = 2
    winNew2 = Toplevel(root)
    winNew2.geometry('1300x800')
    winNew2.title('调试结果')
    btJianCe = Button(winNew2, text='     查看检测组     ', font=("黑体", 20), fg='red', relief=RIDGE, command=newwindow2)
    btJianCe.place(relx=0.2, rely=0.1)
    btJianCe = Button(winNew2, text='     预测组    ', font=("黑体", 20), fg='black', relief=RIDGE, command=DISABLED)
    btJianCe.place(relx=0.7, rely=0.1)
    if flag1 == 0:#二维二分型
        weight1, accuracy1 = maincode.stocGraAscent1(dataMatrix1, matLabel1)
        testdata, matLabel2,file3 = maincode.loadExamSet2(file2)
        finalmatrix,accuracy2 = maincode.Test(testdata, weight1, matLabel2)
        Txt1 = scrolledtext.ScrolledText(winNew2)
        Txt1.place(relx=0.55, rely=0.2)
        finalmatrix = delete(finalmatrix,2,axis=1)
        Txt1.insert("insert", finalmatrix)#文本框
        maincode.drawtest(weight1, finalmatrix)
        img_open = Image.open('temp1.png')#散点图
        img_png = ImageTk.PhotoImage(img_open)
        label1 = Label(winNew2)
        label1.configure(image=img_png)
        label1.image = img_png  # keep a reference!
        label1.place(relx=0.05, rely=0.2)
        label1.pack
        if accuracy1<0.9:#二次拟合
            def Twice(self):
                dataMatrix2, matLabel2, file3 = maincode.Solving(file1)
                weight2, accuracy2 = maincode.stocGraAscent1(dataMatrix2, matLabel2)
                testdata, matLabel2,file4 = maincode.loadExamSet2(file2)
                dataMatrix3, matLabel4, file5 = maincode.Solvingtest(file4)
                finalmatrix2,accuracy2 = maincode.Test(dataMatrix3, weight2, matLabel4)
                finalmatrix2 = delete(finalmatrix2, -2, axis=1)
                Txt1 = scrolledtext.ScrolledText(winNew2)
                Txt1.place(relx=0.55, rely=0.2)
                Txt1.insert("insert", finalmatrix2)#文本框
                maincode.draw1test(weight2, finalmatrix)
                img_open = Image.open('temp1.png')#散点图
                img_png = ImageTk.PhotoImage(img_open)
                label1 = Label(winNew2)
                label1.configure(image=img_png)
                label1.image = img_png  # keep a reference!
                label1.place(relx=0.05, rely=0.2)

            btTwice = Button(winNew2, text='     二次拟合     ', font=("黑体", 20), fg='black', relief=RIDGE,
                             command=DISABLED)
            btTwice.place(relx=0.1, rely=0.8)
            btTwice.bind('<1>', Twice)
    elif flag1 == 1:#多维二分型
        weight1, accuracy1 = maincode.stocGraAscent1(dataMatrix1, matLabel1)
        testdata, matLabel2, file3 = maincode.loadExamSet2(file2)
        finalmatrix,accuracy2 = maincode.Test(testdata, weight1, matLabel2)
        Txt1 = scrolledtext.ScrolledText(winNew2)
        Txt1.place(relx=0.5, rely=0.2)
        finalmatrix = delete(finalmatrix, -2, axis=1)
        Txt1.insert("insert", finalmatrix)
        maincode.drawtest(weight1, finalmatrix)
        img_open = Image.open('2.png')
        img_png = ImageTk.PhotoImage(img_open)
        label1 = Label(winNew2)
        label1.configure(image=img_png)
        label1.image = img_png  # keep a reference!
        label1.place(relx=0.05, rely=0.2)
        label1.pack
    else:#连续型
        weight1 = maincode.stocGraAscent2(dataMatrix1, matLabel1)
        testdata, matLabel2, file3 = maincode.loadExamSet3(file2)
        finalmatrix, ww1 = maincode.Test2(testdata, weight1, matLabel2)
        maincode.Draw2test(file, ww1)
        img_open = Image.open('temp1.png')
        img_png = ImageTk.PhotoImage(img_open)
        label1 = Label(winNew2)
        label1.configure(image=img_png)
        label1.image = img_png  # keep a reference!
        label1.place(relx=0.01, rely=0.2)
        label1.pack
        dataMatrix, matLabel, file = maincode.loadDataSet(file1)
        matLabel6, matLabel3, matLabel4, matLabel5 = maincode.BoxCut(matLabel)
        # 下面将数据集4等分，算出4个权值
        weight1, accuracy1 = maincode.stocGraAscent1(dataMatrix, matLabel6.astype('int'))
        weight2, acc2 = maincode.stocGraAscent1(dataMatrix, matLabel3.astype('int'))
        weight3, acc3 = maincode.stocGraAscent1(dataMatrix, matLabel4.astype('int'))
        weight4, acc4 = maincode.stocGraAscent1(dataMatrix, matLabel5.astype('int'))
        testdata, matLabeltest, file2 = maincode.loadExamSet2(file2)
        finalmatrix, ww = maincode.Test3(testdata, weight1, weight2, weight3, weight4, matLabeltest)
        Matrix1, Matrix2, Matrix3, Matrix4, Label1, Label2, Label3, Label4 = maincode.Solving2(dataMatrix, matLabel)
        # 下面将数据集16等分，算出16个权值
        matLabel11, matLabel12, matLabel13, matLabel14 = maincode.BoxCut(Label1)
        weight11, acc11 = maincode.stocGraAscent1(Matrix1, matLabel11.astype('int'))
        weight12, acc12 = maincode.stocGraAscent1(Matrix1, matLabel12.astype('int'))
        weight13, acc13 = maincode.stocGraAscent1(Matrix1, matLabel13.astype('int'))
        weight14, acc14 = maincode.stocGraAscent1(Matrix1, matLabel14.astype('int'))
        matLabel21, matLabel22, matLabel23, matLabel24 = maincode.BoxCut(Label2)
        weight21, acc21 = maincode.stocGraAscent1(Matrix2, matLabel21.astype('int'))
        weight22, acc22 = maincode.stocGraAscent1(Matrix2, matLabel22.astype('int'))
        weight23, acc23 = maincode.stocGraAscent1(Matrix2, matLabel23.astype('int'))
        weight24, acc24 = maincode.stocGraAscent1(Matrix2, matLabel24.astype('int'))
        matLabel31, matLabel32, matLabel33, matLabel34 = maincode.BoxCut(Label3)
        weight31, acc31 = maincode.stocGraAscent1(Matrix3, matLabel31.astype('int'))
        weight32, acc32 = maincode.stocGraAscent1(Matrix3, matLabel32.astype('int'))
        weight33, acc33 = maincode.stocGraAscent1(Matrix3, matLabel33.astype('int'))
        weight34, acc34 = maincode.stocGraAscent1(Matrix3, matLabel34.astype('int'))
        matLabel41, matLabel42, matLabel43, matLabel44 = maincode.BoxCut(Label4)
        weight41, acc41 = maincode.stocGraAscent1(Matrix4, matLabel41.astype('int'))
        weight42, acc42 = maincode.stocGraAscent1(Matrix4, matLabel42.astype('int'))
        weight43, acc43 = maincode.stocGraAscent1(Matrix4, matLabel43.astype('int'))
        weight44, acc44 = maincode.stocGraAscent1(Matrix4, matLabel44.astype('int'))
        result = {}
        count = len(ww)
        for i in range(count):#进行区间判断
            if (ww[i] == 1):
                Matrix, www = maincode.Test3(testdata[i:i + 1].values, weight11, weight12, weight13, weight14,
                                             matLabel3[i])
                result[i] = maincode.Solving3(www, Label1)
            elif (ww[i] == 2):
                Matrix, www = maincode.Test3(testdata[i:i + 1].values, weight21, weight22, weight23, weight24,
                                             matLabel3[i])
                result[i] = maincode.Solving3(www, Label2)
            elif (ww[i] == 3):
                Matrix, www = maincode.Test3(testdata[i:i + 1].values, weight31, weight32, weight33, weight34,
                                             matLabel3[i])
                result[i] = maincode.Solving3(www, Label3)
            else:
                Matrix, www = maincode.Test3(testdata[i:i + 1].values, weight41, weight42, weight43, weight44,
                                             matLabel3[i])
                result[i] = maincode.Solving3(www, Label4)
        Txt1 = scrolledtext.ScrolledText(winNew2)
        Txt1.place(relx=0.5, rely=0.2)
        for i in range(count):#在文本框中输出结果
          Txt1.insert("insert", i+1)
          Txt1.insert("insert", '   ')
          Txt1.insert('insert',ww1[i],'')
          Txt1.insert("insert",'   ')
          Txt1.insert("insert", result[i])
          Txt1.insert(INSERT, '\n')

def newwindow2():#训练集展示窗口
    global flag1
    global count
    global file
    winNew2 = Toplevel(root)
    winNew2.geometry('1300x800')
    winNew2.title('调试结果')
    dataMatrix1, matLabel1, file = maincode.loadDataSet(file1)
    count, col = dataMatrix1.shape
    if col == 2:  # 数据集的类型自动判断
        if matLabel1[0] == 0 or matLabel1[0] == 1:
            flag1 = 0
        else:
            flag1 = 2
    else:
        if matLabel1[0] == 0 or matLabel1[0] == 1:
            flag1 = 1
        else:
            flag1 = 2
    if flag1 == 0:#二维二分型
        weight1, accuracy1 = maincode.stocGraAscent1(dataMatrix1, matLabel1)
        maincode.draw(weight1,file)
        testdata,matLabel2,file2=maincode.loadExamSet(file1,ttype)
        finalmatrix,accuracy2=maincode.Test(testdata,weight1,matLabel2)
        img_open = Image.open('temp.png')
        img_png = ImageTk.PhotoImage(img_open)
        label1 = Label(winNew2)
        label1.configure(image=img_png)
        label1.image = img_png  # keep a reference!
        label1.place(relx=0.1, rely=0.2)
        label1.pack
        Txt1 = scrolledtext.ScrolledText(winNew2)
        Txt1.place(relx=0.6, rely=0.2)
        Txt1.insert("insert", finalmatrix)
        label2 = Label(winNew2, text=('精准度为', accuracy2), fg='black', font=("黑体", 23), relief=RIDGE)
        label2.place(relx=0.6, rely=0.9)
        if accuracy1 < 0.9:
            def Twice(self):#二次拟合
                dataMatrix2,matLabel2,file3=maincode.Solving(file1)
                weight2,accuracy2=maincode.stocGraAscent1(dataMatrix2,matLabel2)
                maincode.draw1(weight2,file3)
                dataMatrix3,matLabel4,file4 = maincode.Solving(file2)
                finalmatrix2,accuracy3=maincode.Test(dataMatrix3,weight2,matLabel4)
                Txt1 = scrolledtext.ScrolledText(winNew2)
                Txt1.place(relx=0.6, rely=0.2)
                finalmatrix2=delete(finalmatrix2,range(0,-3),axis=1)
                Txt1.insert("insert", finalmatrix2)

                label2 = Label(winNew2, text=('  精准度为       ', accuracy2), fg='black', font=("黑体", 28), relief=RIDGE)
                label2.place(relx=0.1, rely=0.9)
                img_open = Image.open('temp.png')
                img_png = ImageTk.PhotoImage(img_open)
                label1 = Label(winNew2)
                label1.configure(image=img_png)
                label1.image = img_png  # keep a reference!
                label1.place(relx=0.1, rely=0.2)
                label1.pack
                label4 = Label(winNew2, text=('  精准度为       ', accuracy3), fg='black', font=("黑体", 28), relief=RIDGE)
                label4.place(relx=0.6, rely=0.9)
            btTwice = Button(winNew2, text='     二次拟合     ', font=("黑体", 20), fg='black', relief=RIDGE,
                              command=DISABLED)
            btTwice.place(relx=0.1, rely=0.8)
            btTwice.bind('<1>',Twice)
    elif flag1==1:#多维二分型
        weight1, accuracy1 = maincode.stocGraAscent1(dataMatrix1, matLabel1)
        testdata, matLabel2, file2 = maincode.loadExamSet(file1, ttype)
        finalmatrix,acccc= maincode.Test(testdata, weight1, matLabel2)
        img_open = Image.open('2.png')
        img_png = ImageTk.PhotoImage(img_open)
        label1 = Label(winNew2)
        label1.configure(image=img_png)
        label1.image = img_png  # keep a reference!
        label1.place(relx=0.1, rely=0.2)
        label1.pack
        Txt1 = scrolledtext.ScrolledText(winNew2)
        Txt1.place(relx=0.5, rely=0.2)
        finalmatrix2 = delete(finalmatrix, range(0, -3), axis=1)
        Txt1.insert("insert", finalmatrix2)
        label2 = Label(winNew2, text=('精准度为', acccc), fg='black', font=("黑体", 23), relief=RIDGE)
        label2.place(relx=0.6, rely=0.9)
    else:#连续型
        weight1 = maincode.stocGraAscent2(dataMatrix1, matLabel1)
        testdata, matLabel2, file2 = maincode.loadExamSet(file1, ttype)
        finalmatrix, ww = maincode.Test2(testdata, weight1, matLabel2)
        maincode.Draw2(file, ww)
        img_open = Image.open('temp.png')
        img_png = ImageTk.PhotoImage(img_open)
        label1 = Label(winNew2)
        label1.configure(image=img_png)
        label1.image = img_png  # keep a reference!
        label1.place(relx=0.01, rely=0.2)
        label1.pack
        dataMatrix, matLabel, file = maincode.loadDataSet(file1)
        matLabel6, matLabel3, matLabel4, matLabel5 = maincode.BoxCut(matLabel)
        weight1, accuracy1 = maincode.stocGraAscent1(dataMatrix, matLabel6.astype('int'))
        weight2, acc2 = maincode.stocGraAscent1(dataMatrix, matLabel3.astype('int'))
        weight3, acc3 = maincode.stocGraAscent1(dataMatrix, matLabel4.astype('int'))
        weight4, acc4 = maincode.stocGraAscent1(dataMatrix, matLabel5.astype('int'))
        testdata, matLabeltest, filelocal = maincode.loadExamSet2(file2)
        finalmatrix, ww = maincode.Test3(testdata, weight1, weight2, weight3, weight4, matLabeltest)
        Matrix1, Matrix2, Matrix3, Matrix4, Label1, Label2, Label3, Label4 = maincode.Solving2(dataMatrix, matLabel)
        matLabel11, matLabel12, matLabel13, matLabel14 = maincode.BoxCut(Label1)
        weight11, acc11 = maincode.stocGraAscent1(Matrix1, matLabel11.astype('int'))
        weight12, acc12 = maincode.stocGraAscent1(Matrix1, matLabel12.astype('int'))
        weight13, acc13 = maincode.stocGraAscent1(Matrix1, matLabel13.astype('int'))
        weight14, acc14 = maincode.stocGraAscent1(Matrix1, matLabel14.astype('int'))
        matLabel21, matLabel22, matLabel23, matLabel24 = maincode.BoxCut(Label2)
        weight21, acc21 = maincode.stocGraAscent1(Matrix2, matLabel21.astype('int'))
        weight22, acc22 = maincode.stocGraAscent1(Matrix2, matLabel22.astype('int'))
        weight23, acc23 = maincode.stocGraAscent1(Matrix2, matLabel23.astype('int'))
        weight24, acc24 = maincode.stocGraAscent1(Matrix2, matLabel24.astype('int'))
        matLabel31, matLabel32, matLabel33, matLabel34 = maincode.BoxCut(Label3)
        weight31, acc31 = maincode.stocGraAscent1(Matrix3, matLabel31.astype('int'))
        weight32, acc32 = maincode.stocGraAscent1(Matrix3, matLabel32.astype('int'))
        weight33, acc33 = maincode.stocGraAscent1(Matrix3, matLabel33.astype('int'))
        weight34, acc34 = maincode.stocGraAscent1(Matrix3, matLabel34.astype('int'))
        matLabel41, matLabel42, matLabel43, matLabel44 = maincode.BoxCut(Label4)
        weight41, acc41 = maincode.stocGraAscent1(Matrix4, matLabel41.astype('int'))
        weight42, acc42 = maincode.stocGraAscent1(Matrix4, matLabel42.astype('int'))
        weight43, acc43 = maincode.stocGraAscent1(Matrix4, matLabel43.astype('int'))
        weight44, acc44 = maincode.stocGraAscent1(Matrix4, matLabel44.astype('int'))
        result = {}
        count = len(ww)
        for i in range(count):
            if (ww[i] == 1):
                Matrix, www = maincode.Test3(testdata[i:i + 1].values, weight11, weight12, weight13, weight14,
                                             matLabel3[i])
                result[i] = maincode.Solving3(www, Label1)
            elif (ww[i] == 2):
                Matrix, www = maincode.Test3(testdata[i:i + 1].values, weight21, weight22, weight23, weight24,
                                             matLabel3[i])
                result[i] = maincode.Solving3(www, Label2)
            elif (ww[i] == 3):
                Matrix, www = maincode.Test3(testdata[i:i + 1].values, weight31, weight32, weight33, weight34,
                                             matLabel3[i])
                result[i] = maincode.Solving3(www, Label3)
            else:
                Matrix, www = maincode.Test3(testdata[i:i + 1].values, weight41, weight42, weight43, weight44,
                                             matLabel3[i])
                result[i] = maincode.Solving3(www, Label4)
        Txt1 =scrolledtext.ScrolledText(winNew2)
        Txt1.place(relx=0.5, rely=0.2)
        accc = 0
        for i in range(count):
          Txt1.insert("insert",file2.iloc[i,-1],'')
          Txt1.insert("insert", '   ')
          Txt1.insert("insert", result[i])
          Txt1.insert(INSERT, '\n')
          if file2.iloc[i,-1]<=result[i][0,1] and file2.iloc[i,-1]>=result[i][0,0]:
            accc+=1
        accuracy1=accc/count


    btJianCe=Button(winNew2,text='     检测组     ',font=("黑体",20),fg='black',relief=RIDGE,command=DISABLED)
    btJianCe.place(relx=0.2,rely=0.1)
    btJianCe = Button(winNew2, text='     查看预测组    ', font=("黑体", 20), fg='red', relief=RIDGE, command=newwindow3)
    btJianCe.place(relx=0.7, rely=0.1)

    label2=Label(winNew2,text=('精准度为',accuracy1),fg='black',font=("黑体",23),relief=RIDGE)
    label2.place(relx=0.1,rely=0.9)

def newwindow():#选择测试机组数窗口
    global count
    global ttype
    winNew = Toplevel(root)
    winNew.geometry('500x300')
    winNew.title('选择您的调试数据数目')
    lb2 = Label(winNew, text=('共有',count,'组数据'),fg='black',font=("黑体",18),relief=RIDGE)
    lb2.place(relx=0.1, rely=0.2)
    lb3 =Label(winNew, text='您希望使用多少组数据进行调试检验',fg='black',font=("黑体",18),relief=RIDGE)
    lb3.place(relx=0.1, rely=0.4)

    lbb = Label(winNew, text='')
    lbb.pack()
    def show(event):#滑纽
        global ttype
        s = '滑块的取值为' + str(var.get())
        lbb.config(text=s)
        ttype=var.get()
    var=IntVar()
    scl = Scale(winNew, orient=HORIZONTAL, length=200, from_=0, to=count, label='请拖动滑块', tickinterval=0, resolution=1,
                variable=var)
    scl.bind('<1>', show)
    scl.pack()
    scl.place(relx=0.1,rely=0.6)

    btSure = Button(winNew, text='确定', font=("黑体",10),command=newwindow2)
    btSure.place(relx=0.3, rely=0.9)
    btSure.bind('<1>')
    btClose = Button(winNew, text='返回', font=("黑体",10),command=winNew.destroy)
    btClose.place(relx=0.7, rely=0.9)


btn1 = Button(root, text='开始',fg='black',font=("黑体",20),relief=RIDGE,command=newwindow)

btn1.place(relx=0.4, rely=0.65, relwidth=0.2, relheight=0.1)
btn1.pack


root.mainloop()



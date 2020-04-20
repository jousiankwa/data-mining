import numpy as np
import pandas as pd


data=pd.read_csv('iris.csv')
data=np.array(data)
data=np.mat(data[:,0:4])
#数据长度
length=len(data)
#通过核函数在输入空间计算核矩阵
k=np.mat(np.zeros((length,length)))
for i in range(0,length):
    for j in range(i,length):
        k[i,j]=(np.dot(data[i],data[j].T))**2
        k[j,i]=k[i,j]
name=range(length)
test=pd.DataFrame(columns=name,data=k)
print('核矩阵\n',test)


len_k=len(k)
#中心化核矩阵
I=np.eye(len_k)
one=np.ones((len_k,len_k))
A=I-1.0/len_k*one
centered_k=np.dot(np.dot(A,k),A)
test=pd.DataFrame(columns=name,data=centered_k)
print('居中化核矩阵\n',test)


#标准化核矩阵
W_2=np.zeros((len_k,len_k))
for i in range(0,len_k):
    W_2[i,i]=k[i,i]**(-0.5)
normalized_k=np.dot(np.dot(W_2,k),W_2)
test=pd.DataFrame(columns=name,data=normalized_k)
print('规范化核矩阵\n',test)


#标准化中心化核矩阵
W_3=np.zeros((len_k,len_k))
for i in range(0,len_k):
    W_3[i,i]=centered_k[i,i]**(-0.5)
normalized_centered_k=np.dot(np.dot(W_3,centered_k),W_3)
test=pd.DataFrame(columns=name,data=normalized_centered_k)
print('居中规范化核矩阵\n',test)

#计算每个输入向量的特征函数φ
fai=np.mat(np.zeros((length,10)))
for i in range(0,length):
    for j in range(0,4):
        fai[i,j]=data[i,j]**2
    for m in range(0,3):
        for n in range(m+1,4):
            j=j+1
            fai[i,j]=2**0.5*data[i,m]*data[i,n]
name_f=range(10)
test=pd.DataFrame(columns=name_f,data=fai)
print('计算φ\n',test)


#通过φ计算核矩阵
k_f=np.mat(np.zeros((length,length)))
for i in range(0,length):
    for j in range(i,length):
        k_f[i,j]=(np.dot(fai[i],fai[j].T))
        k_f[j,i]=k_f[i,j]
test=pd.DataFrame(columns=name,data=k_f)
print('通过φ计算核矩阵\n',test)


#中心化φ
rows=fai.shape[0]
cols=fai.shape[1]
centered_fai=np.mat(np.zeros((rows,cols)))
for i in range(0,cols):
    centered_fai[:,i]=fai[:,i]-np.mean(fai[:,i])
test=pd.DataFrame(columns=name_f,data=centered_fai)
print('居中φ\n',test)


#通过中心化φ计算中心化的核函数
k_cf=np.mat(np.zeros((length,length)))
for i in range(0,length):
    for j in range(i,length):
        k_cf[i,j]=(np.dot(centered_fai[i],centered_fai[j].T))
        k_cf[j,i]=k_cf[i,j]
test=pd.DataFrame(columns=name,data=k_cf)
print('通过居中φ计算居中核矩阵\n',test)


#规范化φ
normalized_fai=np.mat(np.zeros((rows,cols)))
for i in range(0,len(fai)):
    temp=np.linalg.norm(fai[i])
    normalized_fai[i]=fai[i]/np.linalg.norm(fai[i])
test=pd.DataFrame(columns=name_f,data=normalized_fai)
print('规范化φ\n',test)


#通过规范化φ计算规范化的核函数
k_nf=np.mat(np.zeros((length,length)))
for i in range(0,length):
    for j in range(i,length):
        k_nf[i,j]=(np.dot(normalized_fai[i],normalized_fai[j].T))
        k_nf[j,i]=k_nf[i,j]
test=pd.DataFrame(columns=name,data=k_nf)
print('通过规范化φ计算规范化核矩阵\n',test)


#计算居中规范化φ
normalized_centered_fai=np.mat(np.zeros((rows,cols)))
for i in range(0,len(fai)):
    temp=np.linalg.norm(fai[i])
    normalized_centered_fai[i]=centered_fai[i]/np.linalg.norm(centered_fai[i])
test=pd.DataFrame(columns=name_f,data=normalized_centered_fai)
print('居中规范化φ\n',test)


#通过居中规范化φ计算剧中规范化核矩阵
kc_nf=np.mat(np.zeros((length,length)))
for i in range(0,length):
    for j in range(i,length):
        kc_nf[i,j]=(np.dot(normalized_centered_fai[i],normalized_centered_fai[j].T))
        kc_nf[j,i]=kc_nf[i,j]
test=pd.DataFrame(columns=name,data=kc_nf)
print('通过居中规范化φ计算居中规范化核矩阵\n',test)
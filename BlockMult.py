# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import re
import numpy as np
from pyspark import SparkContext 


sc = SparkContext(appName="inf553")

# Map Input
matA = sc.textFile('file-A.txt')
matB = sc.textFile('file-B.txt')

# translate str to int
def trans_k(m):
    tem = re.search('\((.*,.*)\),\[', m).group(1).split(",")
    l = []
    for i in tem:
        l.append(int(i))
    return tuple(l)

def trans_v(m):
    tem1 = re.search('\[(\(.*\))\]', m).group(1).split("),(")
    l1 = []
    new_list = [re.findall('\d', i) for i in tem1]
    for i in new_list:
        t1 = []
        for j in i:
            t1.append(int(j))
        t1 = tuple(t1)
        l1.append(t1)
    return l1

matA_map_out = matA.map(lambda x:(trans_k(x),trans_v(x)))
matB_map_out = matB.map(lambda x:(trans_k(x),trans_v(x)))


# translate to ((i, 1), ('A', k, Aik)), ..., ((i, b), ('A', k, Aik))
def LeftMapper(x):
    LM = []
    for K in range(1,4):
        tem = ((x[0][0],K),("A",x[0][1],x[1]))
        LM.append(tem)
    return LM
               
def RightMapper(x):
    RM = []
    for K in range(1,4) :
        tem = ((K,x[0][1]),("B",x[0][0],x[1]))
        RM.append(tem)
    return RM


redA_in = matA_map_out.flatMap(LeftMapper)
redB_in = matB_map_out.flatMap(RightMapper)



# combine A AND B by key
#red_tep = redA_in.join(redB_in).filter(lambda x: x[1][0][1] == x[1][1][1])

# group by key to get reduce input
#red_input = red_tep.groupByKey().map(lambda x: (x[0], list(x[1])))

redA_in_list = matA_map_out.flatMap(LeftMapper).collect()
redB_in_list = matB_map_out.flatMap(RightMapper).collect()

# extend to one list
redA_in_list.extend(redB_in_list)
red_in = sc.parallelize(redA_in_list)

# define func to do block multiplication--- def a null matrix and replace the value
def baby_matrix(x):
    L = [[0,0],
        [0,0]]
    for i in x[1][2]:
        L[i[0]-1][i[1]-1]=i[2]
    x[1][2]=L
    return list(x)


def block_multi(M):
    L1=[]
    for i in M:
        for j in M:
            if i[1]==j[1] and i[0] < j[0]:            # i[0]<j[0] means A != B
                L1.append(np.dot(i[2],j[2]))
    if L1 == []:
        L1 = [[0,0],[0,0]]                           #replace null with 0 in order to tanslate to list
        return L1
    else:
        return(sum(L1).tolist())


semi_result = red_in.map(baby_matrix).groupByKey().mapValues(block_multi)

#func to remap the the list of lists to required format (1,1,2).....
def indexing(a):
    x = []
    for i in range(0,2):
        if i == 0:
            x.append(a[i])
        elif i == 1:
            y = []
            if a[1][0][0] != 0:
                y.append([1,1,a[1][0][0]])
            if a[1][0][1] != 0:
                y.append([1,2,a[1][0][1]])
            if a[1][1][0] != 0:
                y.append([2,1,a[1][1][0]])
            if a[1][1][1] != 0:
                y.append([2,2,a[1][1][1]])
            x.append(y)
    return x


result = semi_result.map(lambda x: indexing(x)).sortBy(lambda a: a[0]).filter(lambda x: x[1] !=[]).collect()


# write to a file
with open ('Output.txt','w') as file:
   for x in result:
     file.write(str(x[0]) + ',' + str(x[1]) + '\n')









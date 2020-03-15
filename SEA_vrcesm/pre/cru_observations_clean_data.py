#import libraries
import numpy as np
import pandas as pd

#set up CRU pre observations directory filename
obsdir = "/Users/Harryliyi/Documents/Research/obsdataset/CRU/pre/"
obsfname = "pre.1704241136.clean.dtb"

#define country name
country = "THAILAND"

#define inital year and end year
iniyear = 1995
endyear = 2004

#read observations from CRU
fopen = open(obsdir+obsfname,"r")
linereader = fopen.readlines()

#define function to split data
def datasplit(str):
    tempstr = [str[0:4]]
    tempstr.extend([ str[start:start+5] for start in range(4, 60, 5) ])
    return tempstr

#read data
lineno = 0
iscountry = False
predata = np.array([],dtype=float)
stndata = []
currpre = np.array([],dtype=float)
currstn = []
while lineno < len(linereader):
    if len(linereader[lineno][64:].strip(' '))>1:
        #print(len(linereader))
        linelist = linereader[lineno].split()
        #print(linelist)
    else:
        #print(linereader[lineno])
        linelist = datasplit(linereader[lineno])
        #print(linelist)
    #print(linelist)
    #print(len(linelist))
    if len(linereader[lineno][64:].strip(' '))==1:
        if iscountry:
            idxyear = int(linelist[0])
            if idxyear==iniyear:
                currpre = np.append(currpre,np.asfarray(np.array(linelist[1:]),float))
            if (idxyear>iniyear) and (idxyear<endyear):
                currpre = np.append(currpre,np.asfarray(np.array(linelist[1:]),float))
            if idxyear==endyear:
                currpre = np.append(currpre,np.asfarray(np.array(linelist[1:]),float))
                if (len(currpre)==(endyear-iniyear+1)*12) and not(np.any(currpre ==-9999)):
                    predata = np.concatenate((predata,currpre))
                    stndata.append(currstn)
    if len(linereader[lineno][64:].strip(' '))>1:
        print("Current station is "+" ".join(str(x) for x in linelist[4:-3])+" in "+linelist[-3])
        if linelist[-3] == country:
            iscountry = True
            currpre = np.array([],dtype=float)
            currstn = linelist
        else:
            iscountry = False
            currpre = np.array([],dtype=float)
            currstn = []
        lineno = lineno + 1
    #print(currpre)
    #print(iscountry)
    lineno = lineno + 1
                    
predata = np.reshape(predata,(len(stndata),12*(endyear-iniyear+1)))
print((len(predata)))   
print(len(stndata))
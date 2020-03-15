#import libraries
import numpy as np
import pandas as pd

#set up CRU pre observations directory filename
obsdir = "/scratch/d/dylan/harryli/gpcdata/obsdataset/SA_OBS/SACA_blend_rr/"
stnsum = "stations.txt"

#define country name
countries = ["Thailand","Vietnam","Cambodia","Myanmar"]
countryids = ["TH","VN","KH","MM"]

#define inital year and end year
iniyear = 1979
endyear = 2005

#create a function to convert degree/minute/second to decimal
def deg2dec(x):
    xlist =x.split(":")
    if xlist[0][0]=="-":
        xdec = float(xlist[0])-float(xlist[1])/60.-float(xlist[2])/60./60.
    else:
        xdec = float(xlist[0])+float(xlist[1])/60.+float(xlist[2])/60./60.
    return xdec

#find the stn ids
fopen = open(obsdir+stnsum,"r")
linereader = fopen.readlines()[19:]

stnids = np.array([],dtype=int)
stnnames = np.array([])
countrynames = np.array([])
stnlats = np.array([],dtype=float)
stnlons = np.array([],dtype=float)

for lineno in range(len(linereader)):
    linelist = linereader[lineno].split(",")
    #print(linelist)
    #print(" ".join(x for x in linelist[1].split()))
    if linelist[2] in countryids:
        stnids = np.append(stnids,int(linelist[0]))
        stnnames = np.append(stnnames," ".join(x for x in linelist[1].split()))
        countrynames = np.append(countrynames,countries[countryids.index(linelist[2])])
        stnlats = np.append(stnlats,deg2dec(linelist[3]))
        stnlons = np.append(stnlons,deg2dec(linelist[4]))
    print("Current station "+linelist[0]+" is "+" ".join(x for x in linelist[1].split())+" in "+linelist[2]+" at "+str(deg2dec(linelist[3]))+"/"+str(deg2dec(linelist[4])))


saobspre = np.array([],dtype=float)

print("Totally "+str(len(stnids))+" stations are found. Their information is shown in following:")
for idx in range(len(stnids)):
    print("station "+str(idx+1)+"/"+str(len(stnids))+" is: "+stnnames[idx]+" in "+countrynames[idx]+" at "+str(stnlats[idx])+"/"+str(stnlons[idx])+" with an station id "+str(stnids[idx]).zfill(6))
    
    #open file from each station
    obsfname = "RR_STAID"+str(stnids[idx]).zfill(6)+".txt"
    fopen = open(obsdir+obsfname,"r")
    linereader = fopen.readlines()[21:]
    
    #read data from station between given years
    lineno = 0
    currpre = np.array([],dtype=float)
    missingcount = 0.
    while int(linereader[lineno].split(",")[2][0:4])<=endyear:
        #print(linereader[lineno].split(",")[2])
        if (int(linereader[lineno].split(",")[2][0:4])>=iniyear) and (linereader[lineno].split(",")[2][4:8]!="0229"):
            currpre = np.append(currpre,float(linereader[lineno].split(",")[3]))
            if (int(linereader[lineno].split(",")[4])==9) or (int(linereader[lineno].split(",")[4])==1):
                missingcount = missingcount + 1.
                currpre[-1] = np.NaN
        lineno = lineno +1
    
    #record the data if it cover the whole temperal range and no missing data 
    if (len(currpre)==365*(endyear-iniyear+1)):
        print("Current missing ratio is "+str(missingcount/365/(endyear-iniyear+1)))
        saobspre = np.concatenate((saobspre,currpre))
    

saobspre = np.reshape(saobspre,(len(stnids),365*(endyear-iniyear+1)))
print(np.shape(saobspre))

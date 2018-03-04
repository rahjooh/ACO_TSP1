import random
import operator
import numpy as np
import datetime
import math
#np.seterr(divide='ignore', invalid='ignore')

def importDS(path, FromLine, ToLine):
    f1 = open(path, 'r')
    ds = []
    for i, line in enumerate(f1):
        if i >= FromLine and i <= ToLine:
            str1 = line.replace('\n', '').split(',')
            for l in str1:
                str2 = l.split(' ')
                str3 = []
                for ll in str2:
                    if ll != '' and ll != ' ':
                        str3.append(ll)
                ds.append(str3)
    return ds
def getDistanceLatLong(ds):
    ds1 = np.float64(ds)
    ds2 = np.empty([len(ds), len(ds)])

    DictDist = {}
    for i in range(ds2.shape[0]):
        for j in range(i, ds2.shape[0]):
            if j < i: continue
            if j == i: ds2[i][j] = 0.0; continue
            lat1 = ds1[i][1]
            lat2 = ds1[j][1]
            lon1 = ds1[i][2]
            lon2 = ds1[j][2]
            dlon = lon2 - lon1
            dlat = lat2 - lat1

            a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            Base = 6371 * c
            ds2[i][j] = Base
            DictDist[str(int(ds1[i][0])).strip() + '|' + str(int(ds1[j][0])).strip()] = ds2[i][j]
            DictDist[str(int(ds1[j][0])).strip() + '|' + str(int(ds1[i][0])).strip()] = ds2[i][j]
            DictDist[str(int(ds1[i][0])).strip() + '|' + str(int(ds1[i][0])).strip()] = 0
            # print('i=' + str(i) + '  j=' + str(j) + ' lat1=' + str(lat1) + '  lat2=' + str(lat2) + '  lon1=' + str(lon1) + '  lon2=' + str(lon2) + '  res='+            str(ds2[i][j]))
    return ds2, DictDist
"""Α, α > Άλφα (a’l-phah) > as in “cat”
Β, β > Βήτα (vee’-tah) > as in “vote”
Γ, γ > Γάμμα (ga’-mah) > soft sound as in “yet”
Δ, δ > Δέλτα (the’-ltah) > as in “the”
Ε, ε > Έψιλον (e’h-psee-lon) > as in “set”
Ζ, ζ > Ζήτα (zee’-tah) > as in “zoo”
Η, η > Ήτα (ee’-tah) > as in “sit”
Θ, θ > Θήτα (thee’-tah) > as in “think”
Ι, ι > Ιότα (yo’-tah) > as in “sit”
Κ, κ > Κάππα (ka’h-pah) > as in “kite”
Λ, λ > Λάμδα (la’m-thah) > as in “lion”
Μ, μ > Μι (mee’) > as in “me”
Ν, ν > Νι (nee’) > as in “net”
Ξ, ξ > Ξι (xee’) > as in “taxI”
Ο, ο > Όμικρον (aw’-mee-kron) > as in “pot”
Π, π > Πι (pee’) > as in “pot”
Ρ, ρ > Ρο (raw’) > as in “raw”
Σ, σ, ς > Σίγμα (see’-gmah) > as in “set”
Τ, τ > Ταυ (ta’ph) > as in “toe”
Υ, υ > Ύψιλον (ee’-psee-lon) > as in “sit”
Φ, φ > Φι (phee’) > as in “fish”
Χ, χ > Χι (hee’) > hard h as in “Loch”
Ψ, ψ > Ψι (psee’) > as in “tops”
Ω, ω > Ωμέγα (oh-me’h-ga) > as in “pot”"""
def roulette_selection(weights):
    # sort the weights in ascending order
    sorted_indexed_weights = sorted(enumerate(weights), key=operator.itemgetter(1));
    indices, sorted_weights = zip(*sorted_indexed_weights);
    # calculate the cumulative probability
    tot_sum = sum(sorted_weights)
    prob = [x / tot_sum for x in sorted_weights]
    cum_prob = np.cumsum(prob)
    # select a random a number in the range [0,1]
    random_num = random.random()

    for index_value, cum_prob_value in zip(indices, cum_prob):
        if random_num < cum_prob_value:
            return index_value

ds1 = importDS('ali535.tsp', 7, 541)
destlist, destdict = getDistanceLatLong(ds1)
#print(roulette_selection([1,2,6,4,3,7,20,1000]))

m = 5   #tedad morcheh
Q=100
alpha = 2        #Zarib tavani Tao
Beta =1           #Zarib tavani eta
Roh = 0.02         #Darsad Tabkhir


def ACO_MetaHeuristic(roh,alfa,beta,tedadMorche,DistanceDict,DistanceMatrix):
    ncity = destlist.shape[0]
    tau = (1 - np.diag(np.ones(ncity)))
    delta_tau = (1 - np.diag(np.ones(ncity)))

    f1 = open ('monitoring.csv','w')
    bool =True
    oldcurrent = {}
    current = {}
    visited = {}

    for i in range(m):
        current[i] = int(np.random.choice(ncity, 1, replace=False))
        visited[i] = [current[i]]
        #print(str(visited[i]) + '    ' + str(type(visited[i])))

    while bool:
        t =1
        str1 = ''
        for w in range(ncity-1):

            for i in range(m):

                # mohasebe ehtamal masir ha
                p = [] ;
                zigma= 0
                for k in range(ncity):
                    if (destlist[current[i]][k])> 0 :
                        zigma += (tau[current[i]][k]**alfa)*(destlist[current[i]][k]**beta)
                    elif(destlist[k][current[i]])> 0: zigma += (tau[k][current[i]]**alfa)*(destlist[k][current[i]]**beta)
                    else: zigma+=0

                for j in range(ncity):
                    if destlist[current[i]][j] > 0 :
                        eta = 100.0/float(destlist[current[i]][j])
                    elif destlist[j][current[i]] > 0 :
                        eta = 100.0 / float(destlist[j][current[i]])
                    else:eta=0.0

                    if j in visited[i]:p.append(0)
                    else:
                        if zigma == 0: print('err : destlist[' +current[str(i)] + '][' + str(j) + ']' + str(
                            destlist[current[i]][k]) + '    tau[' + str(i) + '][' + str(j) + '] = ' + str(
                            tau[i][j]) + '      current[' + str(i) + ']=' + str(current[i]))
                        if j==534 and current[i]==534 : p.append(0)
                        else: p.append((tau[i][j]**alfa)*(eta**beta)/zigma)

                #entekhab masir ba roulette wheel
                oldcurrent[i] = current[i]
                counter=0
                while counter<100 and sum(p)!=0 and(oldcurrent[i] == current[i] or type(current[i])!= int):current[i] = roulette_selection(p) ; counter+=1
                visited[i].append(current[i])
                if current[i] == None and counter >99:
                    for row5 in range(len(p)) :
                        if p[row5]>0 :
                            current[i]=p[row5]
                if current[i] == None: break

                visited[i].append(current[i])
                if len(visited[i]) < ncity : t = 0
                #update delta_tao
                if destlist[oldcurrent[i]][current[i]] > 0 :
                    delta_tau[oldcurrent[i]][current[i]] += 1.0 / float(destlist[oldcurrent[i]][current[i]])
                    delta_tau[current[i]][oldcurrent[i]] += 1.0 / float(destlist[oldcurrent[i]][current[i]])
                elif destlist[current[i]][oldcurrent[i]] > 0 :
                    delta_tau[oldcurrent[i]][current[i]] += 1.0 / float(destlist[current[i]][oldcurrent[i]])
                    delta_tau[current[i]][oldcurrent[i]] += 1.0 / float(destlist[current[i]][oldcurrent[i]])
                elif oldcurrent[i] != current[i]:
                    print(' internal err : fasele city '+str(oldcurrent[i])+'  ta city'+ str(current[i])+' shode 0 !!!')

            #updating tau : tabkhir + formon rizi(delta_tau)
            for row1 in range(tau.shape[0]):
                for row2 in range(tau.shape[1]):
                    tau[row1][row2] = (1-roh)*tau[row1][row2] + delta_tau[row1][row2]
                    tau[row2][row1] = (1-roh)*tau[row2][row1] + delta_tau[row2][row1]
            if w % 50 == 0: str1 = ' ' + str1
            if w<100 :str1 = '█'+str1
            elif w>100 and w<200 : str1 = str1[:-1]
            elif w>200 and w<300 :str1 = '█'+str1
            elif w>300 and w<400 :str1 =str1[:-1]
            else :str1 ='█'+str1

            print(str1,end='')
            print(str(w/ncity*100)[0:5]+'%')
        bool = False            #print(str(w) + '    ' +str(i) + '     ' +str(visited[i]) )

    print('100%')
    print('yes')
    old = -1
    best = 99999999999;
    for i in range(m):
        cost = 0

        for row in visited[i]:
            if old == -1 : old = row
            else:
                cost += DistanceDict[str(old+1) + '|' + str(row+1)]/12
        if cost < best :
            best = cost
            bestpath = visited[i]

    print(best)
    print(bestpath)

    bestpath = []
    oldcurrent1 = 0
    current1 = 0
    bestpath.append(current1)
    cost = 0.0
    topvalue = 0
    topvalueindex = -1
    if current1 != None :
        for i in range(ncity-1):
            for row in range(tau[current1]):
                if topvalue < tau[current1][row] :
                    topvalue = tau[current1][row]
                    topvalueindex = row
            oldcurrent1 = current1
            current1 = topvalueindex
            bestpath.append(topvalueindex)
            cost += DistanceDict[str(oldcurrent1)+'|'+str(current1)]

    print(cost)
    print(bestpath)







ACO_MetaHeuristic(Roh,alpha,Beta,m,destdict,destlist)
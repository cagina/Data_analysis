import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import math
from numpy.ma.core import get_object_signature
from pandas import DataFrame as df


def isint(value):
    try:
        int(value)
        return int(value)
    except:
        return False
def __eq__(self, other):
    return self.Value == other.Value


data = pd.read_csv('http://download.geonames.org/export/dump/countryInfo.txt', sep="\t", skiprows=50)

df = pd.DataFrame(data)

df.drop(df.tail(1).index, inplace=True)

df.rename(columns={'#ISO': 'CountryCode'}, inplace=True)

del df['ISO3']
del df['ISO-Numeric']
del df['fips']
del df['EquivalentFipsCode']
del df['Postal Code Format']
del df['Postal Code Regex']
del df['geonameid']

df['Continent'] = df['Continent'].replace(np.NaN, 'NA')

df1 = pd.read_csv('data.csv', skiprows=1, encoding="ISO-8859-1")

df1 = pd.DataFrame(df1)
df1.drop(df1.tail(1).index, inplace=True)
df1.drop([col for col in df1.columns if "Unnamed" in col], axis=1, inplace=True)
df1 = df1.dropna()
df1["Country"] = df1["Country"].str.strip()

df1["Country"] = df1["Country"].replace("Venezuela (Bolivarian Republic of)", "Venezuela")
df1["Country"] = df1["Country"].replace("The former Yugoslav Republic of Macedonia", "Macedonia")
df1["Country"] = df1["Country"].replace("Hong Kong, China (SAR)", "Hong Kong")
df1["Country"] = df1["Country"].replace("Korea (Republic of)", "South Korea")

df2 = pd.merge(df1, df, on='Country', how='inner', indicator=True)
df2.set_index('Country', inplace=True)

print(df1.tail())


a=[]

for column in df2.columns[1:]:

    if isint(column)>=1990 and isint(column)<=2015:
        a.append(column)

df2["GDP ortalama"]=df2.loc[:, a].mean(axis=1)




df2['GDP ortalama'].plot(kind='hist',bins=100)

plt.xlabel('ortalama')

plt.show()

dede=df2.nlargest(10, 'GDP ortalama')
objects=[]
for country in dede.index.get_level_values(0):
    objects.append(country)

print(len(objects))
y_pos = np.arange(len(objects))
performance = [10,9,8,7,6,5,4,3,2,1]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('GDP Ortalama')
plt.title('GDP Ortalaması en yüksek 10 ülke')
plt.show()


yuzolcumoranı=df2['Population']/df2['Area(in sq km)']
dede=yuzolcumoranı.nsmallest(10).index.get_level_values(0)

print(dede)
y_pos = np.arange(len(dede))
performance = [10,9,8,7,6,5,4,3,2,1]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, dede)
plt.ylabel('KBDYA Ortalama')
plt.title('Kişi başına düşen yüzey alanını')
plt.show()




for kıtalar in pd.unique(df2["Continent"]):
    orda_olanlar=df2.loc[df2['Continent'] == kıtalar].index.get_level_values(0)
    orda=pd.DataFrame(orda_olanlar)
    orda["GDP ortalama"]=df2["GDP ortalama"][orda_olanlar].values
    orda.plot.bar(x=orda_olanlar)
    plt.title(kıtalar+' kıtasındaki ülkelerin GDP Ortalaması')
    plt.show()






degerler=[]
orda=pd.DataFrame(pd.unique(df2["Continent"]))
for kıtalar in pd.unique(df2["Continent"]):
    orda_olanlar=df2.loc[df2['Continent'] == kıtalar].index.get_level_values(0)
    orda2 = df2["Population"][orda_olanlar].values.sum()
    degerler.append(orda2)


orda["Toplam Nüfüs"]=degerler
orda = orda.rename(columns={ 0: 'Kıtalar'})
orda.set_index('Kıtalar', inplace=True)
orda.plot(kind="pie",subplots=True)
plt.show()



bolunme=str(df2["neighbours"]["Turkey"]).split(',')
bolunme.append("TR")
degerler=[]
ülke=[]
for codlar in bolunme:
    ülkeler = df2.loc[df2['CountryCode'] == codlar].index.get_level_values(0)
    deger=df2["GDP ortalama"][ülkeler].values
    ülke.append(ülkeler)
    degerler.append(deger)
orda=pd.DataFrame(ülke)
orda["GDP ortalama"]=degerler
orda = orda.rename(columns={0: 'Ülkeler'})
orda=orda.dropna()
orda.set_index('Ülkeler', inplace=True)
orda=orda.astype(float)
orda.plot.bar()
plt.show()

if deger is 0.0:
        deger = df2["GDP ortalama"][df2.loc[sayac]].values



orda=pd.DataFrame(df2.index.tolist())

ana_deger=[]
sayac=-1
diz=[]
for ülke_komusları in df2["neighbours"].values:
    ülkekendiGdp=0.0
    sayac+=1
    bolunme = str(ülke_komusları).split(',')
    degerler = []
    deger=0.0
    komsular=len(bolunme)
    for codlar in bolunme:
        ülkeler = df2.loc[df2['CountryCode'] == codlar].index.get_level_values(0)
        deger1 = df2["GDP ortalama"][ülkeler].values
        if deger1:
            deger=deger+deger1
        else:
            komsular=komsular-1

    if deger is 0.0:
        deger = df2["GDP ortalama"][df2.index[sayac]]

    if komsular :
        deger=deger/komsular
    ülkekendiGdp=df2["GDP ortalama"][df2.index[sayac]]
    deger=deger-ülkekendiGdp
    deger=deger.__abs__()
    ana_deger.append(deger)


orda["GDP ortalama"] = ana_deger
orda = orda.rename(columns={0: 'Ülkeler'})
orda.set_index('Ülkeler', inplace=True)
orda=orda.astype(float)
orda=orda.nlargest(20,'GDP ortalama')
print(orda)




plt.scatter(df2['Area(in sq km)'],df2['Population'],s=20,c='r', edgecolors='k',alpha=0.5)
plt.xlabel('Area (in sq km')
plt.ylabel('Population')
plt.title("Ülke nüfusu - yüzey ölçümü dağılım grafiği")
plt.show()


df2['HDI Rank (2015)'] = df2['HDI Rank (2015)'].apply(pd.to_numeric, errors='coerce')
kaka = df2.nsmallest(20,'HDI Rank (2015)')


hdi = []
for a in kaka['HDI Rank (2015)']:
      hdi.append(a)

      gdp = []
for b in kaka['GDP ortalama']:
      gdp.append(b)

      size_gdp = []
for each in gdp:
       size_gdp.append(each/100)



kitalar = [1,2,1,1,1,3,1,1,1,4,4,3,2,1,1,3,3,3,1,1]

plt.scatter(x=hdi,y=gdp,s=size_gdp,c=kitalar,edgecolors='k',alpha=0.5)
plt.xlabel('HDI Rank (2015)')
plt.ylabel('GDP Ortalama')
plt.title("İnsani gelişmişlik düzey - GDP arasındaki ilişki ")
plt.show()

        

gdp_fark = df2.loc[:,'1990'].sub(df2.loc[:,'2015']).abs().nlargest(15)
print(gdp_fark)

gdp_fark.plot(kind='bar')
plt.show()

        

sa_ulkeler = df2.loc[df2['Continent'] == 'SA']
sa_ulkeler.loc[:,'1990':'2015'].plot.line()
plt.title("1990'dan 2015'e kadar olan GDP değişimi")
plt.show()









diller=[]
for ülke_dilleri in df2["Languages"].values:
         bolunme = str(ülke_dilleri).split(',')
         for dil in bolunme:
              dil=dil.split('-')
              if dil[0] not in  diller:
                 diller.append(dil[0])


butun_yerler=[]

for dil in diller:
    sayac=-1
    yer=[]
    for dilin_yeri in df2["Languages"].values:
        sayac+=1
        if(dilin_yeri.__contains__(dil)) :
            yer.append(sayac)
    butun_yerler.append(yer)


orda=pd.DataFrame(diller)

konuslan_toplam_nufus=[]
konusuldugu_sayısı=[]
for ülke_konusanlar in butun_yerler:
    nüfüs=0
    sayısı=0
    for ülke in ülke_konusanlar:
        sayısı+=1
        nüfüs+=df2["Population"][ülke]

    konuslan_toplam_nufus.append(nüfüs)
    konusuldugu_sayısı.append(sayısı)




orda["Konusan Toplam Ülke"] = konusuldugu_sayısı
orda = orda.rename(columns={0: 'Diller'})
orda.set_index("Diller", inplace=True)

orda.plot.bar()
plt.show()

print(orda)

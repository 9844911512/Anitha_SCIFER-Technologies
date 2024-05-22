import pandas as pd
import numpy as np

s1=[10,20,30,40,50]
data=pd.Series(s1)
print(data)

s2=[30000,60000,90000,120000]
data=pd.Series(s2)
print(data)

n=np.arange(0,70)
data=np.Series(n)
print(data)

marks={"sec1":60,"sec2":90,"sec3":45,"sec4":75}
h=pd.Series(marks)
print(h)




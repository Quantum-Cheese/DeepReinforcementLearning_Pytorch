import pandas as pd
import numpy as np

scores=np.array([[1,-1.2],[2,-3.4],[3,3.6]])


df=pd.DataFrame(data=scores,columns=['episode','score'])

print(df)

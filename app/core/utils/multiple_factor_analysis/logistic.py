import pandas as pd
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Binomial

def logistic(df):
    columns = df.columns
    model1=glm(f'{columns[0]}~{"+".join(columns[1:])}',data=df,family=Binomial()).fit()
    return model1.summary()

# filepath1=r'C:\Users\admin\PycharmProjects\Multivariate_analysis\HYPOG_com数据填充.xlsx'
# df1=pd.read_excel(filepath1)


# model1=glm('Y~X1+X2+X3+X4+X5+X6',data=df1,family=Binomial()).fit()
# print(model1.summary())





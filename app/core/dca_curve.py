import sys
import pandas as pd
from rpy2 import robjects

data_name = sys.argv[1]
dca_img = f'results/{data_name}/plots/dcaCurve.png'
dcaData_path = f'results/{data_name}/logging/step_1_data_cleaning.csv'
data = pd.read_csv(dcaData_path)
columns = f'{data.columns[0]}~{"+".join(data.columns[1:])}'

rscripts = \
    f'''
            if (!require("rmda")) install.packages("rmda")
            library(rmda)
            png(filename='{dca_img}')

            dcaData <- read.csv(file = '{dcaData_path}', fileEncoding = "UTF-8-BOM")

            dcaCurve <- decision_curve(
                {columns},
                data = dcaData,
                family = binomial(link = 'logit'),
                thresholds = seq(0, 1, by=0.01),
                confidence.intervals = 0.95,
                study.design = 'cohort',
                bootstraps = 10
            )

            plot_decision_curve(dcaCurve, curve.names ="dcaCurve")
            dev.off()
    '''

try:
    robjects.r(rscripts)
except:
    ...

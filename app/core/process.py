'''

{'var-ID': 'quan', 'var-入院时间': 'quan', 'var-当前状态': 'binary', 'var-下一状态': 'mult_order', 'var-出生年月日': 'quan', 
    'var-死亡年月日': 'quan', 'var-手动进展日期': 'quan', 'var-进展情况': 'mult_order', 'var-是否进展': 'binary', 'var-入院年龄': 'quan', 'var-用药种数': 'quan', 'var-住院天数': 'quan', 'var-上次出院距离': 'quan', 'var-是否医保': 'binary', 'var-住院次数': 'quan', 'var-分期判断': 'mult_order', 'var-是否合并症': 'binary', 'var-高
血压': 'binary', 'var-合并症个数': 'mult_order', 'var-手术情况': 'binary', 'var-转移部位数量': 'mult_order', 'var-远处转移数量': 'mult_order', 'var-是否淋巴转移': 'binary', 'var-是否远处转移': 'binary', 'var-不良反应数量': 'mult_order', 'var-骨髓抑制': 'binary', 'var-疼痛': 'binary', 'var-肝肾功异常': 'binary', 'var-化疗方案整理': 'mult_order', 'var-癌种': 'mult_order', 'var-性别': 'binary', 'var-效用值': 'quan', 'var-共计': 'quan', 'Impute Methods': ['ISVD', 'Imput'], 'Sampling Methods': ['SSMO', 'BSMO'], 'Selection Methods': ['RCV', 'Cat'], 'Train Methods': ['Random Forest', 'AdaBoost', 'Gradient Boosting']}

'''
class DataProcesser:
    
    def __init__(self, form_config, form_data):
        ...
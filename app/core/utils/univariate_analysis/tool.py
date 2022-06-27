def format_float(data: float, n: int, float_=False):
    """
    给定数据，保留n位小数
        例子： format(2.358755, '.5f') ，结果是 2.35875，正确四舍五入值是 2.35876
    由于计算机最后一位如果是5，实际内存中的数据是4999999...，所以会导致四舍五入不正确

    float_=False   该方法返回的数据类型是 str
    float_=True  该方法返回的数据类型是 float
    """
    digit = len(str(data).split('.')[-1])  # 计算小数位数
    if n == digit - 1:  # 保留的位数正好比总的小数位数少一位，则在后面补1
        data = float(str(data) + '1')
    data = format(data, f'.{n}f')
    if float_:
        data = float(data)
    return data

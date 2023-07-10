import math


# hàm tính correlation bằng tay
def cor(x, y):
    # trung binh cong x va y
    mean_x = sum(x) / float(len(x))
    mean_y = sum(y) / float(len(y))
    #
    sub_x = [i - mean_x for i in x]
    sub_y = [i - mean_y for i in y]
    numerator = sum([sub_x[i] * sub_y[i] for i in range(len(sub_x))])
    # denominator = len(x) - 1
    # cov = numerator / denominator

    covx = math.sqrt(sum(i ** 2 for i in sub_x))
    covy = math.sqrt(sum(i ** 2 for i in sub_y))
    # print(covx)
    # print(covy)
    result = numerator / (covx * covy)
    return result
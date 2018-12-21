import sys
sys.path.append("/home/darth/projects/master-thesis-code")
import numpy
import myopt

import pyximport; pyximport.install()
import segmentation

data = segmentation.load_data()[:1000]

def f(theta):
    return segmentation.Model(alpha = theta[0], p_c = theta[1]).fit(data).final_log_P_data.item()

bounds = [
        myopt.Float(0.1, 5000),
        myopt.Float(0.05, 0.95)
        ]

result = myopt.bo_maximize(f, bounds, n_iter=50)

with open("result.pkl", "wb") as file:
    import pickle
    result.opt_fun = None
    pickle.dump(result, file)

print(result)

import ipdb
ipdb.set_trace()

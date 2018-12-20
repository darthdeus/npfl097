import sys
sys.path.append("/home/darth/projects/master-thesis-code")
import numpy
import myopt
import segmentation
import ipdb

data = segmentation.load_data()[:1000]

def f(theta):
    return segmentation.fit(data, theta[0], theta[1])[0].item()

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

ipdb.set_trace()


# ▽f ≝ (∂f/∂x, ∂f/∂y)

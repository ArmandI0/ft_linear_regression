import json
import tools as tls
import pandas as pd
import numpy as np

def isNum(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def predict(entry,theta):
    x = float(entry)
    return theta[0] * x + theta[1]

def main():
    try:
        try:
            np.set_printoptions(suppress=True, precision=2)
            theta = open('model/theta.json', 'r')
            thetaContent = json.load(theta)
            theta = np.array([thetaContent['theta_1'],
                            thetaContent['theta_0']])
            
        except Exception as e:
            theta = np.array([0,0])
        entry = input('Provide the mileage : ')        
        if isNum(entry):
            print(predict(entry, theta))
        else:
            print('Invalid mileage')
    except Exception as e:
        print(e)


if __name__=="__main__":
	main()
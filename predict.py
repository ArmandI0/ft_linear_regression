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
            theta = open('model/theta.json', 'r')
            thetaContent = json.load(theta)
            theta = np.array([thetaContent['theta_1'],
                            thetaContent['theta_0']])
            
        except Exception as e:
            theta = np.array([0,0])
        print(theta)
        entry = input('give a mileage : ')        
        if isNum(entry):
            print(predict(entry, theta))
        else:
            print('give a valid entry')
    except Exception as e:
        print(e)


if __name__=="__main__":
	main()
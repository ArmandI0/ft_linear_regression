import tools as tls
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def normaliseData(data):
	# X_norm = (X - X.mean()) / X.std()
	mean = data.mean()
	std = data.std()
	result = (data - mean) / std 
	return result

fig, axs = plt.subplots(2, 2)
np.set_printoptions(suppress=True, precision=2)

dataSet = tls.load_csv('data.csv')
# print(dataSet)
xi = dataSet['km'].values.reshape(-1, 1) # -1 selectionne automatiquement le bon nombres dans la colone
yi = dataSet['price'].values.reshape(-1, 1)
# Reshape les datas pour les mettres sous formes de matrice
x = normaliseData(dataSet['km']).values.reshape(-1, 1) # -1 selectionne automatiquement le bon nombres dans la colone
y = normaliseData(dataSet['price']).values.reshape(-1, 1)
# Ajouter une colones de 1 a la matrice x
X = np.hstack((x, np.ones(x.shape)))


# Calcul les y theoriques:
def model(X, theta):
	# y = ax + b -> model de la regression (equation d'une droite)
	# Produit matriciel revient a faire une boucle avec result = X[0] * theta[0] + X[1] * theta[1]

	result = X.dot(theta)
	print("resultat du modele", result)

	return result


def grad(X, y, theta):
	m = len(y)
	result = 1/m * X.T.dot(model(X, theta) - y )
    # Tracé des points de données
	axs[1][1].scatter(x, model(X, theta), c='r', label='Prédictions')  # Points prédits en rouge
	axs[1][1].scatter(x, y, c='b', label='Données réelles')  # Points réels en bleu
	predictions = model(X, theta)

    # Tracé de la ligne de régression
	axs[1][1].plot(x, model(X, theta), 'r--', label='Ligne de régression')  # Ligne pointillée rouge
	for i in range(len(x)):
		axs[1][1].plot([x[i], x[i]], [y[i], predictions[i]], 'g--', alpha=0.5)
    
	print('gradient = ', result)

	#Retourne le gradiant -> la pente 'a' et l'ordonee a l'origine 'b' result[0] = 'a' / result[1] = 'b'
	return result


def costFunction(X, y, theta):
	m = len(y)
	return 1/(2*m) * np.sum((model(X,theta) - y) ** 2)

def gradiantDescent(X, y, theta, learningRate, iterations,ax, x):
	try:
		costHistory = np.zeros(iterations)
		for i in range(0, iterations):
			theta = theta - learningRate * grad(X, y, theta)
			ax.plot(x, model(X, theta))
			costHistory[i] = costFunction(X, y, theta)
		return theta, costHistory
	except Exception as e:
		print(e)
		return

def main():
	try:
		# Pr print que deux ch


		# Definition de theta
		# theta = np.zeros(2).reshape(-1,1)
		theta = np.array([-0.5, -0.5]).reshape(-1,1)

		# Calcul de la fonction cout (Erreur quadratique moyenne)


		iterations = 1
		finalTheta, costHistory = gradiantDescent(X, y, theta, 0.1, iterations, axs[0][0], x)
		print(finalTheta)

		predict = model(X, finalTheta)

		axs[0][0].scatter(x, y, label='Je sais pas encore', c='b')
		axs[0][0].plot(x, predict, c='r')
		axs[0][1].plot(range(0, iterations), costHistory)
		# plt.figure(2)
		# plt.scatter(xi, yi, c='r')
		axs[0][0].set_xlabel('km')
		axs[0][0].set_ylabel('price')
		plt.show()
	except Exception as e:
		print(e)
if __name__=="__main__":
	main()
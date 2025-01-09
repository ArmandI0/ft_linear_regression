
### Modele lineaire

```
h(x) = θ₀ + θ₁x
```

### Fonction Coût (Mean Squared Error)

```
J(θ) = (1/2m) Σ(h(x⁽ᵢ⁾) - y⁽ᵢ⁾)²
```
```Python
def costFunction(X, y, theta):
    m = len(y)
    return 1/(2*m) * np.sum((model(X,theta) - y)**2)
```

- m : nombre d'exemples d'entraînement "len(y)"
- h(x) : prédiction du modèle
- y : valeur réelle

### Gradient
Cette fonction calcule la dérivée partielle de la fonction coût par rapport à θ :
```
∂J(θ)/∂θ = (1/m) X^T·(X·θ - y)
```

```Python
def grad(X, y, theta):
    m = len(y)
    return 1/m * X.T.dot(model(X, theta) - y)
```

### Descente de Gradient

```
θ = θ - α·∂J(θ)/∂θ
```
- α : learning rate (taux d'apprentissage)
- ∂J(θ)/∂θ : gradient
```Python
def gradiantDescent(X, y, theta, learningRate, iterations):
        theta = theta - learningRate * grad(X, y, theta)
```
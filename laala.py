import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Leer archivos CSV
capacidad_por_saco = pd.read_csv('capacidad_por_saco.csv', header=None).values.flatten()
costo_saco = pd.read_csv('costo_saco.csv', header=None).values
tiempo_demora = pd.read_csv('tiempo_demora.csv', header=None).values.flatten()
kilos_fruta = pd.read_csv('kilos_fruta.csv', header=None).values.flatten()
precio_venta = pd.read_csv('precio_venta.csv', header=None).values
capital_inicial = pd.read_csv('capital_inicial.csv', header=None).values.item()
cantidad_cuadrantes = pd.read_csv('cantidad_cuadrantes.csv', header=None).values.item()

# Definir parámetros
J = len(capacidad_por_saco)
K = cantidad_cuadrantes
T = costo_saco.shape[1]

# Crear modelo
model = gp.Model('Plantación')

# Variables de decisión
X = model.addVars(J, K, T, vtype=GRB.BINARY, name='X')
Y = model.addVars(J, K, T, vtype=GRB.BINARY, name='Y')
I = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name='I')
U = model.addVars(J, T, vtype=GRB.INTEGER, lb=0, name='U')
W = model.addVars(J, T, vtype=GRB.INTEGER, lb=0, name='W')

# Función objetivo
model.setObjective(I[T-1], GRB.MAXIMIZE)

# Restricciones
# Activación sembrado
for j in range(J):
    for k in range(K):
        for t in range(T):
            model.addConstr(gp.quicksum(Y[j, k, l] for l in range(t, min(t + tiempo_demora[j], T))) >= tiempo_demora[j] * X[j, k, t])

# Solo un sembrado por cuadrante
for k in range(K):
    for t in range(T):
        model.addConstr(gp.quicksum(Y[j, k, t] for j in range(J)) <= 1)

# Inventario de dinero
for t in range(1, T):
    model.addConstr(I[t] == I[t-1] - gp.quicksum(costo_saco[j, t] * W[j, t] for j in range(J)) + gp.quicksum(X[j, k, t-tiempo_demora[j]] * kilos_fruta[j] * precio_venta[j, t] for j in range(J) for k in range(K) if t >= tiempo_demora[j]))

# Condición borde inventario dinero
model.addConstr(I[0] == capital_inicial - gp.quicksum(costo_saco[j, 0] * W[j, 0] for j in range(J)))

# Inventario semillas
for j in range(J):
    for t in range(1, T):
        model.addConstr(U[j, t] == U[j, t-1] + capacidad_por_saco[j] * W[j, t] - gp.quicksum(X[j, k, t] for k in range(K)))

# Condición borde semillas
for j in range(J):
    model.addConstr(U[j, 0] == capacidad_por_saco[j] * W[j, 0] - gp.quicksum(X[j, k, 0] for k in range(K)))

# Terminar cosecha antes de volver a cosechar
for j in range(J):
    for k in range(K):
        for t in range(T-1):
            model.addConstr(1 - X[j, k, t] >= gp.quicksum(X[j, k, l] for l in range(t+1, min(t + tiempo_demora[j], T))))


# Optimizar modelo
model.optimize()

# Imprimir resultados en consola
if model.status == GRB.OPTIMAL:
    print("Solución óptima:")
    print(f'Dinero final: {I[T-1].X} pesos')

    # Calcular cantidad de plantaciones por terreno
    plantaciones_por_terreno = [0] * K
    for k in range(K):
        for t in range(T):
            #plantaciones_por_terreno = sum(X[j, k, t].X for j in range(J))
            for j in range(J):
                if X[j, k, t].X > 0.5:
                    plantaciones_por_terreno[k] += 1

    print("Cantidad de veces que se plantó en cada terreno en los 11 meses")
    for k in range(K):
        print(f"Terreno {k+1}: {plantaciones_por_terreno[k]} veces")

    # Crear tabla para la visualización
    tabla = np.zeros((K, T))
    for k in range(K):
        for t in range(T):
            for j in range(J):
                if X[j, k, t].X > 0.5:
                    tabla[k, t] = j

    # Visualizar la tabla usando matplotlib
    fig, ax = plt.subplots()
    cax = ax.matshow(tabla, cmap='Blues')

    # Poner etiquetas en los ejes
    ax.set_xticks(np.arange(T))
    ax.set_yticks(np.arange(K))
    ax.set_xticklabels([f'Mes {i+1}' for i in range(T)])
    ax.set_yticklabels([f'Terreno {i+1}' for i in range(K)])
   
    # Rotar etiquetas del eje x
    plt.xticks(rotation=90)

    # Mostrar valores en las celdas
    for i in range(K):
        for j in range(T):
            c = int(tabla[i, j])
            ax.text(j, i, str(c), va='center', ha='center')

    plt.colorbar(cax)
    plt.title('Tabla de Siembra')
    plt.show()
else:
    print('No se encontró una solución óptima.')
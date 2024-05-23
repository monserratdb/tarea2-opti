import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import seaborn as sns

def read_data():
    # Leer datos desde los archivos CSV
    capacidad_por_saco = pd.read_csv('capacidad_por_saco.csv', header=None).squeeze().values
    costo_saco = pd.read_csv('costo_saco.csv', header=None).values
    tiempo_demora = pd.read_csv('tiempo_demora.csv', header=None).squeeze().values
    kilos_fruta = pd.read_csv('kilos_fruta.csv', header=None).squeeze().values
    precio_venta = pd.read_csv('precio_venta.csv', header=None).values
    capital_inicial = pd.read_csv('capital_inicial.csv', header=None).squeeze()
    cantidad_cuadrantes = pd.read_csv('cantidad_cuadrantes.csv', header=None).squeeze()

    return capacidad_por_saco, costo_saco, tiempo_demora, kilos_fruta, precio_venta, capital_inicial, cantidad_cuadrantes

def solve_model():
    α, c, θ, λ, β, γ, K = read_data()

    J = len(α)  # Número de tipos de semillas
    T = c.shape[1]  # Número de meses
    K = int(K)  # Número de cuadrantes

    # Crear modelo
    model = gp.Model("campo_frutal")

    # Variables de decisión
    X = model.addVars(J, K, T, vtype=GRB.BINARY, name="X")
    Y = model.addVars(J, K, T, vtype=GRB.BINARY, name="Y")
    I = model.addVars(T, vtype=GRB.CONTINUOUS, name="I")
    U = model.addVars(J, T, vtype=GRB.INTEGER, name="U")
    W = model.addVars(J, T, vtype=GRB.INTEGER, name="W")

    # Función objetivo
    model.setObjective(I[T-1], GRB.MAXIMIZE)

    # Restricciones
    for j in range(J):
        for k in range(K):
            for t in range(T):
                # Activación sembrado
                model.addConstr(gp.quicksum(Y[j, k, l] for l in range(t, min(t+θ[j], T))) >= θ[j] * X[j, k, t])

    for k in range(K):
        for t in range(T):
            # Solo un sembrado por cuadrante
            model.addConstr(gp.quicksum(Y[j, k, t] for j in range(J)) <= 1)

    for t in range(1, T):
        # Inventario de dinero
        model.addConstr(I[t] == I[t-1] - gp.quicksum(c[j, t] * W[j, t] for j in range(J)) +
                        gp.quicksum(X[j, k, t-θ[j]] * λ[j] * β[j, t] for j in range(J) for k in range(K) if t-θ[j] >= 0))
    
    # Condición borde inventario dinero
    model.addConstr(I[0] == γ - gp.quicksum(c[j, 0] * W[j, 0] for j in range(J)))

    for j in range(J):
        for t in range(1, T):
            # Inventario de semillas
            model.addConstr(U[j, t] == U[j, t-1] + α[j] * W[j, t] - gp.quicksum(X[j, k, t] for k in range(K)))
        
        # Condición borde semillas
        model.addConstr(U[j, 0] == α[j] * W[j, 0] - gp.quicksum(X[j, k, 0] for k in range(K)))

    for j in range(J):
        for k in range(K):
            for t in range(T-1):
                # Terminar cosecha antes de volver a cosechar
                model.addConstr(1 - X[j, k, t] >= gp.quicksum(X[j, k, l] for l in range(t+1, min(t+θ[j], T))))

    # Restricciones adicionales
    for t in range(T):
        model.addConstr(I[t] >= 0)

    for j in range(J):
        for t in range(T):
            model.addConstr(U[j, t] >= 0)
            model.addConstr(W[j, t] >= 0)

    # Optimizar el modelo
    model.optimize()

    # Imprimir resultados
    if model.status == GRB.OPTIMAL:
        print("-------------------------------")
        print(f"Valor óptimo: {I[T-1].X} pesos")

        # Calcular cantidad de plantaciones por terreno
        plantaciones_por_terreno = [0] * K
        for k in range(K):
            for t in range(T):
                plantings = sum(X[j, k, t].X for j in range(J) if X[j, k, t].X > 0)
                if plantings > 0:
                    plantaciones_por_terreno[k] += 1

        print("Cantidad de veces que se plantó en cada terreno:")
        for k in range(K):
            print(f"Terreno {k+1}: {plantaciones_por_terreno[k]}")

        # Crear la tabla de plantaciones
        calendario = [[0 for _ in range(T)] for _ in range(K)]
        for k in range(K):
            for t in range(T):
                for j in range(J):
                    if X[j, k, t].X > 0:
                        calendario[k][t] = j+1
        
        df_calendario = pd.DataFrame(calendario, columns=[f"Mes {t+1}" for t in range(T)], index=[f"Terreno {k+1}" for k in range(K)])

        # Usar matplotlib para visualizar la tabla en blanco y negro
        plt.figure(figsize=(12, 6))
        sns.heatmap(df_calendario, annot=True, fmt="d", cmap="Greys", cbar=False)
        plt.title('Calendario de Plantaciones')
        plt.xlabel('Meses', fontsize=12, color='red')
        plt.ylabel('Terrenos', fontsize=12, color='red')
        plt.show()
    else:
        print("No se encontró una solución óptima")

# Solucionar modelo
solve_model()

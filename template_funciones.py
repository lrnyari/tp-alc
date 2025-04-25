#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 12:49:51 2025

@author: usuario
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Para leer archivos
import geopandas as gpd # Para hacer cosas geográficas
import seaborn as sns # Para hacer plots lindos
import seaborn.objects as so
import networkx as nx # Construcción de la red en NetworkX
import scipy

# Leemos el archivo, retenemos aquellos museos que están en CABA, y descartamos aquellos que no tienen latitud y longitud
museos = gpd.read_file('https://raw.githubusercontent.com/MuseosAbiertos/Leaflet-museums-OpenStreetMap/refs/heads/principal/data/export.geojson')
barrios = gpd.read_file('https://cdn.buenosaires.gob.ar/datosabiertos/datasets/ministerio-de-educacion/barrios/barrios.geojson')


visitantes1 = pd.read_csv('https://raw.githubusercontent.com/lrnyari/tp-alc/main/visitas.txt') #convierte el archivo visitantes.txt en un dataframe
visitantes2 = visitantes1.rename(columns={'3.866000000000000000e+03': 'w'}) #cambia el nombre de la columna 3.866000000000000000e+03 por w
nueva_fila = pd.DataFrame({'w': [3866]}) #creo otro dataframe, que tiene solo una columna y una fila, con la columna de nombre w y con el dato del numero 3866
w = pd.concat([nueva_fila, visitantes2], ignore_index=True) #le agrego a visitantes 2, arriba de todo, la fila de nueva_fila 

# Armamos el gráfico para visualizar los museos
fig, ax = plt.subplots(figsize=(10, 10))
barrios.boundary.plot(color='gray',ax=ax)
museos.plot(ax=ax)

# En esta línea:
# Tomamos museos, lo convertimos al sistema de coordenadas de interés, extraemos su geometría (los puntos del mapa), 
# calculamos sus distancias a los otros puntos de df, redondeamos (obteniendo distancia en metros), y lo convertimos a un array 2D de numpy
D = museos.to_crs("EPSG:22184").geometry.apply(lambda g: museos.to_crs("EPSG:22184").distance(g)).round().to_numpy()

#%%
def construye_adyacencia(D,m): 
    # Función que construye la matriz de adyacencia del grafo de museos
    # D matriz de distancias, m cantidad de links por nodo
    # Retorna la matriz de adyacencia como un numpy.
    D = D.copy()
    l = [] # Lista para guardar las filas
    for fila in D: # recorriendo las filas, anexamos vectores lógicos
        l.append(fila<=fila[np.argsort(fila)[m]] ) # En realidad, elegimos todos los nodos que estén a una distancia menor o igual a la del m-esimo más cercano
    A = np.asarray(l).astype(int) # Convertimos a entero
    np.fill_diagonal(A,0) # Borramos diagonal para eliminar autolinks
    return(A)


#%%
def calculaLU(matriz):
    # matriz es una matriz de NxN
    # Retorna la factorizaci√≥n LU a traves de una lista con dos matrices L y U de NxN.
    matriz = matriz.astype(np.float64) # Para operar con más precisión.

    n = matriz.shape[0] # n = dimensión de la matriz.

    # Si hay algún 0 en la diagonal de la matriz, imprime un mensaje de error y devuelve las matrices Identidad y matriz.
    for p in range(0, len(matriz)):
        if matriz[p][p] == 0:
            Identidad = np.eye(len(matriz))
            print("Error: hay por lo menos un cero en la diagonal.")
            return (Identidad, matriz)

    # Si la matriz es de 1x1, L=(1) y U=(a11).
    if n == 1:
        L = 1
        U = matriz[0, 0]
        return (L, U)

    U = np.zeros_like(matriz) # Creamos U, una matriz de 0s con la dimensión de la matriz.
    U[0, :] = matriz[0, :]  # Primera fila de U es igual a la primera fila de la matriz.

    L = np.eye(n) # Creamos L como la identidad nxn.
    L[1:, 0] = matriz[1:, 0] / U[0, 0] # Primera columna de L sin el primer elemento.

    # Generamos una submatriz para aplicar recursión:
    # A la matriz original le sacamos la primera fila y columna y le restamos la primera columna de la matriz L (sin el primer elemento)
    # por la primera fila de la matriz U (sin el primer elemento).
    matriz_recursiva = matriz[1:, 1:] - L[1:, 0].reshape(n - 1, 1) @ U[0, 1:].reshape(1, n - 1)

    L_recu, U_recu = calculaLU(matriz_recursiva) # Generamos el resto de los valores de L y U de forma recursiva.
    L[1:, 1:] = L_recu
    U[1:, 1:] = U_recu

    return (L, U)

#%%



#%%
#Función que recibe una matriz y devuelve otra matriz transpuesta de A
#transpuesta=que las filas sean las columnas y las columnas las filas
def trans(A):
    filas = len(A) #longitud de las filas
    columnas = len(A[0]) #longitud de las columnas
    AA = [] #matriz vacia
    for j in range(columnas):
        nueva_fila = [] #una fila vacia
        for i in range(filas):
            nueva_fila.append(A[i][j]) #le pongo lo de las columnas a las filas
        AA.append(nueva_fila) #pongo la nueva fila en una nueva matriz
    Aarray= np.asarray(AA) #convierto lista a numpy
    return Aarray

#K es una matriz de ceros que tiene en su diagonal la suma de las columnas de A
def LaK(A):
  #Funcion que recibe una matriz A y devuelve una matriz K
    n = A.shape[0]  #pido el numero de filas
    K = np.zeros((n, n))  #matriz de ceros segun numero de filas
    for i in range(n):
        K[i, i] = np.sum(A[i, :])  #sumo cada columna y lo pongo en la diagonal
    return K

#como K es una matriz de ceros con solo numeros en la diagonal, su inversa sera
#la misma solo que los numeros  la diagonal ahora estaran elevados a la menos uno
def diagonalalamenos1(K):
  #Recibe una matriz K y devuelve una matriz Kinversa 
    Kinversa = K.copy() #copio K
    for i in range(K.shape[0]): 
        Kinversa[i, i] = 1 / K[i, i] #le pido que la diagonal este a la menos 1
    return Kinversa

def calcula_matriz_C(A):
  #Funcion que recibe una matriz A de adyacencia y genera la matriz C
  #Recibe una matriz A y devuelve una matriz C
  Atranspuesta = trans(A) #traspongo A
  Kinv = diagonalalamenos1(LaK(A)) #saco K inv
  C = Atranspuesta@Kinv #producto matricial entre Atranspuesta y Kinv
  return C



#%%
def calcula_pagerank(A,alfa):
    # Funci√≥n para calcular PageRank usando LU
    # A: Matriz de adyacencia
    # d: coeficientes de damping
    # Retorna: Un vector p con los coeficientes de page rank de cada museo
    C = calcula_matriz_C(A)
    N = A.shape[0] # Obtenemos el n√∫mero de museos N a partir de la estructura de la matriz A
    M = (N/alfa)*(np.eye(N) - (1-alfa)*C) #el dato de que es M, nos lo dice el enunciado del punto 1
    L, U = calculaLU(M) # Calculamos descomposici√≥n LU a partir de C y d
    b = np.ones(N) # Vector de 1s, multiplicado por el coeficiente correspondiente usando d y N.
    Up = scipy.linalg.solve_triangular(L,b,lower=True) # Primera inversi√≥n usando L
    p = scipy.linalg.solve_triangular(U,Up) # Segunda inversi√≥n usando U
    return p

#%%
def calcula_matriz_C_continua(D):
    # Funci√≥n para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C en versi√≥n continua
    D = D.copy()
    F = 1/D
    np.fill_diagonal(F,0) #como F en la diagonal me da infinito, reemplazo los infinitos por ceros
    KFinv = diagonalalamenos1(LaK(F)) # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de F
    C =  F@KFinv # Calcula C multiplicando Kinv y F
    return C



#%%
#hago una funcion que me da una matriz elevado al exponete (este no debe ser cero)
def elevar(matriz, exponente):
  #Retorna el resultado de multiplicar la matriz con si misma n veces
  res=matriz #matriz elevado a la uno
  for i in range(1, exponente):
    res@=matriz #y luego le multiplico a la matriz elevado a la uno lo que necesito para q este elevado a lo q yo quiero q este
  return res

#B es igual a la sumatoria de Ccontinua a la cero hasta Ccontinua elevado a cantidad_de_visitas -1 inclusive
def calcula_B(C,cantidad_de_visitas):
    # Recibe la matriz T de transiciones, y calcula la matriz B que representa la relación entre el total de visitas 
    #y el número inicial de visitantes suponiendo que cada visitante realizó cantidad_de_visitas pasos
    # C: Matirz de transiciones
    # cantidad_de_visitas: Cantidad de pasos en la red dado por los visitantes. Indicado como r en el enunciado
    # Retorna:Una matriz B que vincula la cantidad de visitas w con la cantidad de primeras visitas v
    B = np.eye(C.shape[0]) #B inicialmente es una matriz elevado a la cero, lo que seria igual a Ccontinua elevado a la cero
    for i in range(1, cantidad_de_visitas):
        B+= elevar(C, i)# Sumamos las matrices de transición para cada cantidad de pasos, osea le sumamos la Ccont elevado a la uno hasta la Ccont elevado a cantidad_de_visitas -1 inclusive
    return B


def dicPagerankSobreM(D, m, alpha):
  #Funcion que genera un diccionario con el pagerank de cada museo para una serie de valores de m
  #Recibe la matriz D, un float alpha y una lista de enteros m
  #Devuelve un diccionario con claves "museo" , "m" y "pgr". Los valores de cada clave son listas(enteros para museo y m, float64 para pagerank)
    diccionarioPagerank = {"museo":[], "m":[], "pgr":[]}
    for valor in m:
        A = construye_adyacencia(D, valor)
        pagerank = calcula_pagerank(A, alpha)
        for i in range(pagerank.shape[0]):
            diccionarioPagerank["museo"].append(i)
            diccionarioPagerank["m"].append(valor)
            diccionarioPagerank["pgr"].append(pagerank[i])
    return diccionarioPagerank

def dicPagerankSobreAlpha(D, m, alpha):
  #Funcion que genera un difccionario con el pagerank de cada museo para una esrie de valordes de alpha
  #Recibe la matriz D, un entero m y una lista de float alpha
  #Devuelve un diccionario con las claves "museo, "alpha", y "pgr".
  #Los valore de "museo" es una lista de enteros, los de "alpha" y "pgr" son listas de float. 
  diccionarioPagerank = {"museo":[], "alpha":[], "pgr":[]}
  A = construye_adyacencia(D, m)
  for valor in alpha:
    pagerank = calcula_pagerank(A, valor)
    for i in range(pagerank.shape[0]):
      diccionarioPagerank["museo"].append(i)
      diccionarioPagerank["alpha"].append(valor)
      diccionarioPagerank["pgr"].append(pagerank[i])
  return diccionarioPagerank
#%%

def prcongrafico(D, m, alpha):
  #Funcion que recibe la matriz de distancias D, un entero m y un float alpha
  #Calcula el pagerank de cada museo, imprime los 3 museos centrales y genera un gráfico sobre el mapa
  #donde el tamaño de cada nodo es proporcional a su pagerank
  adyacencia = construye_adyacencia(D, m)
  pagerank = calcula_pagerank(adyacencia, alpha)
  #Muestra el vector p ordenado para facilitar la visualización.
  prordenado = pd.Series(pagerank).sort_values(ascending=False)
  print("Museos más centrales: \n", prordenado.head(3))
  #display(prordenado.head(3))
  #Para el gráfico
  G = nx.from_numpy_array(adyacencia) # Construimos la red a partir de la matriz de adyacencia
# Construimos un layout a partir de las coordenadas geográficas
  G_layout = {i:v for i,v in enumerate(zip(museos.to_crs("EPSG:22184").get_coordinates()['x'],museos.to_crs("EPSG:22184").get_coordinates()['y']))}
  factor_escala = 1e4 # Escalamos los nodos 10 mil veces para que sean bien visibles
  fig, ax = plt.subplots(figsize=(10, 10)) # Visualización de la red en el mapa
  barrios.to_crs("EPSG:22184").boundary.plot(color='gray',ax=ax); # Graficamos Los barrios
  pr = np.random.uniform(pagerank)# Este va a ser su score Page Rank. Ahora lo reemplazamos con un vector al azar
  pr = pr/pr.sum() # Normalizamos para que sume 1
  Nprincipales = 3 # Cantidad de principales
  principales = np.argsort(pr)[-Nprincipales:] # Identificamos a los N principales
  labels = {n: str(n) if i in principales else "" for i, n in enumerate(G.nodes)} # Nombres para esos nodos
  nx.draw_networkx(G,G_layout,node_size = pr*factor_escala, ax=ax,with_labels=False); # Graficamos red
  nx.draw_networkx_labels(G, G_layout, labels=labels, font_size=6, font_color="k") # Agregamos los nombres
  return
#5)c

#Quiero resolver el sistema Bv=w con B=LU, osea LUv=w, siendo Uv = Y

#Sacamos L de B
#BL=calculaLU(B)[0]
#Sacamos U de B
#BU=calculaLU(B)[1]


#Resolvemos LY=w
#Y=scipy.linalg.solve_triangular(BL, w)

#Resolvemos Uv=Y
#v=scipy.linalg.solve_triangular(BU, Y)

#Y usamos esta libreria sumar el modulo de los datos de v
#vtotales=np.linalg.norm(v, ord=1)
#print("Visitantes Totales: ", vtotales)













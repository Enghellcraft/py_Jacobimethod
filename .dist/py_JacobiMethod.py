# TP Métodos Numéricos - 2023
# Alumna: Denise Martin

# Profesores: en las siguientes líneas estan comentadas todas las funciones con su explicación.
#           Si presiona el "play", podrá ver en la terminal toda la teoría escrita y los ejemplos
#           con la comprobación para cada caso: los que no pueden realizarse y los que sí. 
#           Para correr el programa es necesario tener instalado scipy y matplotlib. 

# Imports
import random
import numpy as np
import scipy.linalg as linalg 
import matplotlib.pyplot as plt
from sympy import jacobi

# Functions
## Square matrix
def my_square_matrix(A):
    A = np.array(A)
    # Revisa si el numero de Columnas es igual al de Filas
    return A.shape[0] == A.shape[1]

## Strictly dominant diagonal matrix
def my_diagonal_strictly_dominant(A):
    # Guarda en diagonal, todos los valores absolutos de la diagonalde la matriz 
    diagonal = np.diag(np.abs(A)) 
    # print(diagonal)
    # Guarda en sum la sumatoria de las filas menos el valor diagonal
    sum = np.sum(np.abs(A), axis=1) - diagonal 
    # print(sum)
    # Compara para cada valor diagonal si es mayor o igual al de la final a la que pertenece
    return (np.all(diagonal >= sum))

## Checks whether the determinant of the matrix is Zero
def my_determinant_is_not_zero(A):
    # print(linalg.det(A))
    return (linalg.det(A) != 0)

## Checks whether the matrix is inversible 
def my_has_inverse(A):
    # Se verifica su inversibilidad por determinante
    if my_determinant_is_not_zero(A) == False:
        print("  La matriz no es inversible")
        return False
    else:
        return True
    
## Inverts the matrix
def my_inverse(A):
    return linalg.inv(A)
                 
## Calculates a zero vector according to the given matrix size
def my_zero_vector(A):
    # Calcula una dimension de la matriz
    n = len(A)
    # Crea un vector de la dimension de la matriz * 1
    zerov = np.zeros(n*1)
    return zerov
                
## Calculates Jacobi
def my_jacobi(A, b, k, n):
    # Requiere un minimo de dos pasos para observar el proceso
    if (k <= 0):
        k = 2
    # No permite que las dimensiones de la matriz sean menores a 2*2
    if (n <= 0):
        n = 2
    print("  La Matriz A es:")
    print_matrix(A)
    print("                                                                                  ")
    # Compuba que la matriz sea cuadrada
    if(my_square_matrix(A)) == False:
        print("  La matriz no es cuadrada y se sugiere no continuar")
    else:
        # Comprueba que la diagonal sea estrictamente dominante
        if (my_diagonal_strictly_dominant(A)) == False:
            print("  La diagonal no es estrictamente dominante y puede no convergir")    
        else:
            if (my_determinant_is_not_zero(A)) == False:
                print("La determinante de la Matriz A es cero y no se puede continuar")
            else:
                    # Calcula D como la diagonal de A
                    D = np.diag(np.diag(A))
                    # Comprueba que la diagonal sea inversible
                    if (my_has_inverse(D)) == False:
                        # Como la determinante de la Matriz A es distinta de cero, esto deberia ser siempre verdadero
                        # Sin embargo como medida de precaución se agrega dentro de las validaciones
                        print("La matriz D no es inversible y no se puede continuar")
                    else:
                        # Calcula L como la matriz inferior estricta de A
                        L = np.tril(A, k=-1)
                        # Calcula L como la matriz superior estricta de A
                        U = np.triu(A, k=1)
                        print("  A = L + U + D, donde:")
                        print("  La Matriz L es:")
                        print_matrix(L)
                        print("  La Matriz U es:")
                        print_matrix(U)
                        print("  La Matriz D es:")
                        Da = np.array(D)
                        print_matrix(np.round(Da,1))
                        print("                                                                                  ")
                        if (my_has_inverse(D) == True):
                            print("  La Matriz inversa de D es:")
                            Dinv = my_inverse(D)
                            Dinva = np.array(Dinv)
                            print_matrix(Dinva)
                            print("  El Vector b es:")
                            print(b)
                            print("  La Sumatoria de L + U es: ")
                            LU = L + U
                            print_matrix(LU)
                            print("                                                                                  ")
                            # Calcula el primer paso de Jacobi: H
                            print("  Considerando: H = D^-1 * (L + U)")
                            H = my_H(Dinv, L, U)
                            print("  H es:")
                            print_matrix(H)
                            # Calcula si la norma de H es valida: || H ||2 < 1
                            if (linalg.norm(H) >= 1 ):
                                print("  La norma de H es inválida para continuar: por || H || >= 1                       ")
                                print(f"  La norma de H es: {linalg.norm(H):.1f}")
                            else:
                                # Calcula el segundo paso de Jacobi: v
                                print("  Considerando v = D^-1 * b")
                                ba = np.array(b)
                                v = np.dot(Dinv,b)
                                print(f"  v es: {print_str_matrix(v)}")
                                print("                                                                                  ")
                                print("  Para comenzar tomaremos el vector inicial Xo:")
                                x0 = my_zero_vector(A)
                                print(f"  X0 es: {print_str_matrix(x0)}")
                                print("  Consideraremos el error como la norma de la diferencia entre X0 y X1:           ")
                                print("  error = || x - x-1 ||                                                           ")
                                print("  Y una tolerancia de 1e-2                                                        ")
                                tol = 1e-2
                                e = 1000
                                print("                                                                                  ")
                                # Se guardan los valores de x para imprimirlos en el plot
                                solutions = [x0]
                                errors = []
                                for i in range(k):
                                    # Calcula la solución por Jacobi para la iteración
                                    x = my_x(H, v, x0)
                                    print(f"  En la Iteración {i+1} el vector resultante x es: {print_str_matrix(x)}, con un vector x0: {print_str_matrix(x0)}")
                                    solutions.append(x)
                                    # Verifica el error a partir de la segunda iteración
                                    e = linalg.norm(x - x0)
                                    errors.append(e)
                                    print(f"  El error es esta iteración {i+1} es de: {e:.2f}")
                                    if (e < tol):
                                        break
                                    
                                    # Reasigna el nuevo X0 como el X del paso anterior                                           
                                    x0 = x
                                print_matrix_graphics(solutions) 
                                print_error_graphics(errors)                 

## Calculates H
def my_H(dinv, L, U):
    LU = L + U
    H = -np.dot(dinv, LU)
    return H

## Calculates x
def my_x(H, v, x0):
    mul = np.dot(H, x0)
    final = mul + v
    return final
    
## Creates a matrix with n parameter
def my_generate_matrix(n):
    random_list = [random.randint(0, 9) for i in range(n*n)]
    matrix = np.reshape(random_list, (n, n))
    return matrix   
                
## Print matrix
def print_matrix(A):
    a = np.array(A)
    for row in a: 
        # Si el valor es int lo imprime como tal, si es un float lo trunca a un decimal
        print(*[f"{value:.1f}" if not value.is_integer() else f"{value:.0f}" for value in row])
        
def print_str_matrix(A):
    cadena = "" 
    for i in A: 
        if i.is_integer(): 
            cadena += str(int(i)) + " " 
        else: 
            cadena += "{:.1f} ".format(i) 
    return cadena

## Plot matrix
def print_matrix_graphics(sol):
    # Calcula la cantidad de Iteraciones para el eje Y
    iter = range(len(sol))
    # Configura la gráfica de los valores del vector solución en función del número de iteraciones
    plt.plot(iter, sol)
    plt.gca().set_facecolor('#e9edc9')
    plt.xlabel('Iteración')
    plt.ylabel('Solución x')
    plt.title('Solución del Sistema Linear usando Método Jacobi')
    plt.show() 
    
## Plot error
def print_error_graphics(es):
    # Calcula la cantidad de Iteraciones para el eje Y
    iter = range(len(es))
    # Configura la gráfica de los valores del vector solución en función del número de iteraciones
    plt.plot(iter, es)
    plt.gca().set_facecolor('#e9edc9')
    plt.xlabel('Iteración')
    plt.ylabel('Error')
    plt.title('Evolución de Error usando Método Jacobi')
    plt.show()    
  
# Dataset
non_square = [[1,2,3],[4,5,6]]

det_zero = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

diag_not_dominant = [[1, 2, 3], [4, 1, 6], [7, 8, 1]]

almost_doable_matrix = [[3, 2, 1],[1, 2, 1],[1, 2, 3]]

doable_matrix = [[5, 2, 1],[1, 6, 3],[2, 3, 7]]

doable_matrix_2 = [[2, 1],[1, 2]]

b3 = [1, 2, 3]

b2 = [1, 2]

# Prints
## null) Task + Pres
print("                                                                                  ")
print("**********************************************************************************")
print("*                 METODOS NUMERICOS - 2023 - TP METODO JACOBI                    *")
print("**********************************************************************************")
print("    • Alumna: Denise Martin                                                       ")
print("                                                                                  ")
print("**********************************************************************************")
print("*                                    CONSIGNA                                    *")
print("**********************************************************************************")
print("  Implementar un programa o script que lea un número natural n >= 1, una matriz de")
print("  números reales A de n x n, un vector b de n números reales y el vector v del    ")
print("  método de Jacobi con el fin de buscar una solución aproximada del sistema de    ")
print("  ecuaciones lineales Ax = b, y deberá hacer k iteraciones del método para este   ")
print("  sistema lineal mostrando los vectores que resulten en los distintos pasos.      ")
print("  Se informará cualquier error o eventualidad que impida calcular lo pedido.      ")
print("  La entrada y salida del programa podrán ser las estándares o bien archivos, a elección.")
print("  (En forma opcional, comparar la solución numérica obtenida con la solución mediante")
print("  algún método exacto, para lo cual se permite tomar este último  de un módulo o     ")
print("  biblioteca de métodos numéricos.)")

## I) Theory
print("                                                                                  ")
print("**********************************************************************************")
print("*                                      TEORIA                                    *")
print("**********************************************************************************")
print("  Jacobi es un método Iterativo, que mejora en cada paso la aproximación al valor solución.")
print("  La base del método consiste en construir una sucesión convergente donde el límite") 
print("  es precisamente la solución del sistema.                                         ")
print("  Dada una Matriz A, se debe expresarla como A = L + U + D, donde:                 ")
print("    • L es estrictamente triangular inferior,                                      ")
print("    • U es estrictamente triangular superior y                                     ")
print("    • D es la diagonal.                                                            ")
print("  Considerando entonces:                                                           ") 
print("    • H = -D^-1 (L+U),                                                             ") 
print("    • v = -D-1 * b                                                                 ") 
print("    • x(k+1) = H x(k) + v                                                          ")  
print("  A partir de la identidad x = -D-1 (L+U) x + D-1b se arma el esquema iterativo    ")
print("  siguiente, que dada una aproximación x(k) la “mejora” obteniendo x(k+1) :        ")
print("    • x(0) = un vector inicial cualquiera                                          ")
print("    • x(k+1) = -D^-1 (L+U) x(k) + D^-1 b                                           ")  
print("                                                                                   ")
print("  Requisitos para asegurar su convergencia:                                        ")
print("    • La matriz principal debe ser cuadrada (ya que no tiene una única diagonal definida).")
print("    • La diagonal debe ser estrictamente dominante (podría converger igualmente en caso contrario).")
print("    • La matriz D debe ser inversible para poder realizar las operaciones.        ")
print("    • La condición || H || < 1 se aplica a la matriz H (no a la matriz A)         ")
print("    • La determinante de la matriz A sea != 0.                                    ")
print("                                                                                  ")
print("  Convergencia:                                                                   ")
print("    • La convergencia no depende del vector inicial x(0)                          ")
print("    • Jacobi tiene O n**2 pasos para una aproximación (donde O es una constante)  ")
print("                                                                                  ")
print("  Error:                                                                          ")
print("    • e(k) = x(k) - x es el vector de error en el paso k                          ")       
print("                                                                                  ")


## II) Examples
print("                                                                                  ")
print("**********************************************************************************")
print("*                                    EJEMPLOS                                    *")
print("**********************************************************************************")
print("    • Se comprueba si la matriz es cuadrada:                                      ")
my_jacobi(non_square, b3, 6, 3)
print("                                 ************                                     ")
print("    • Se comprueba si la diagonal es estrictamente dominante:                     ")
my_jacobi(diag_not_dominant, b3, 6, 3)
print("                                 ************                                     ")
print("    • Se comprueba si la matriz tiene determinante = 0                            ")
my_jacobi(det_zero, b3, 6, 3)
print("                                 ************                                     ")
print("    • Se comprueba si la norma de H > 1:                                          ") 
my_jacobi(almost_doable_matrix, b3, 6, 3)
print("                                 ************                                     ")
print("    • Si cumple todos los requisitos, se emiten todos los valores y cálculos:     ") 
print("    Para iniciar asignaremos un K = 6 a una matriz de 3 * 3                       ") 
my_jacobi(doable_matrix, b3, 6, 3)
print("                                                                                  ") 
print("    Puede verse la tendencia a converger pero aun no llega a un resultado         ")
print("                                                                                  ")
print("    • Ahora la misma matriz pero con un K = 1000 para constatar que si converge   ") 
print("      El programa deberia terminar antes del paso 1000                            ") 
my_jacobi(doable_matrix, b3, 1000, 3)
print("                                                                                  ") 
print("    Puede verse la convergencia según la tolerancia el el paso 12 donde corta el programa.") 
print("                                                                                  ") 

## III) Extra Comparison by Calculator
def jacobi_method(A, b): 
    tol = 1e-2
    n = len(A) 
    x = np.zeros(n) 
    x_prev = np.zeros(n) 
    while True: 
        for i in range(n): 
            s = sum(A[i][j] * x_prev[j] for j in range(n) if j != i) 
            x[i] = (b[i] - s) / A[i][i] 
        if np.linalg.norm(x - x_prev) < tol:
            print([f"{num:.1f}" for num in x]) 
            return x 
        x_prev = np.copy(x) 

print("                                                                                  ")
print("   + Se puede verificar con calculadora:                                          ") 
jacobi_method(doable_matrix, b3)
print("   Donde excepto el segundo valor, concuerda en los otros                         ")
print("                                 ************                                     ")
print("    • Si cumple todos los requisitos, se emiten todos los valores y cálculos:     ") 
print("    Para iniciar asignaremos un K = 6 a una matriz de 2 * 2                       ") 
my_jacobi(doable_matrix_2, b2, 6, 2)
print("                                                                                  ") 
print("    Puede verse la tendencia a converger pero aun no llega a un resultado         ")
print("                                                                                  ")
print("    • Ahora la misma matriz pero con un K = 1000 para constatar que si converge   ") 
print("      El programa deberia terminar antes del paso 1000                            ") 
my_jacobi(doable_matrix_2, b2, 1000, 2)
print("                                                                                  ") 
print("    Puede verse la convergencia según la tolerancia el el paso 8 donde corta el programa.") 
print("                                                                                  ") 
print("   + Se puede verificar con calculadora:                                         ") 
jacobi_method(doable_matrix_2, b2)
print("   Donde concuerda en todos los valores                                           ")


## IV) Conclusions
print("                                                                                  ")
print("**********************************************************************************")
print("*                                  CONCLUSIONES                                  *")
print("**********************************************************************************")
print(" • El Método Jacobi es un método simple de implementar e incluso puede utilizarse,")
print("   como base para otros métodos iteraticos.                                       ") 
print("                                                                                  ")
print(" • Es un método lento de convergencia en los casos donde la diagonal no es        ")
print("   estrictamente dominante, ya que se incrementa el número de iteraciones para    ") 
print("   llegar al resultado.                                                          ") 
print("                                                                                  ")
print(" • La descomposicion de la matriz no requiere de la resolución de sistemas lineales,")
print("   lo cual es ventajoso frente a otros métodos exactos, sin embargo las iteraciones ") 
print("   si lo utilizan.                                                                ") 
print("                                                                                  ")
print(" • Aqui se muetra la varificación de requisitos necesarios para realizar Jacobi, previo ")
print("   a la resolución de la misma. Si pasa todas las verificaciones, se imprimen los    ")
print("   pasos siguientes con sus correspondientes verificaciones.                      ")
print("   Cabe destacar que se asegura su convergencia si pasan los requisitos, sin embargo")
print("   podría converger aún en caso de no cumnplirlos.                                  ")
print("   Es el caso de la diagonal estrictamente dominante donde se encuentra el mayor problema")
print("   ya que es difícil encontrar dichas matrices en la naturaleza.                  ")
print("                                                                                  ")
print(" • NOTA 1: En las líneas 26 y 29, están comentados los print para imprimir las operaciones")
print("         de la funcion que revisa si la diagonal es dominante. Puedes descomentarse ")
print("         para observar dicho comportamiento. ")
print("                                                                                  ")
print(" • NOTA 2: En la línea 35, está comentado el print para imprimir las determinante de ")
print("         la matriz que se está evaluando. Puede descomentarse para observar dicho comportamiento ")
print("                                                                                  ")
print(" • NOTA 3: En las líneas 171 a 175, se encuentra el método para generar una matriz  ")
print("         aleatoria según las dimendiones que se le provean. La idea original era  ")
print("         poder ingresar un N, en la función de my_jacobi y generar una matriz aleatoria")
print("         para evaluarla, sin embargo, luego de todas las verificaciones y ante la posibilidad")
print("         de generar varias matrices que no pasaran la lista de requisitos, decidí ")
print("         dejarlo funcional pero usable en el código actual. El parámetro n, que se ")
print("         agrega en Jacobi, es por ahora, meramente decorativo. ")
print("                                                                                  ")



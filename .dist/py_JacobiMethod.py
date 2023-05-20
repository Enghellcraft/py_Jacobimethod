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
                                print("  v es: ")
                                print(v)
                                print("                                                                                  ")
                                print("  Para comenzar tomaremos el vector inicial Xo:")
                                x0 = b
                                print("  X0 es: ")
                                print(x0)
                                print("  Consideraremos el error como la norma de la diferencia entre X0 y X1:           ")
                                print("  error = || x - x-1 ||                                                           ")
                                print("  Y una tolerancia de 1e-3                                                        ")
                                tol = 1e-3
                                print("                                                                                  ")
                                for i in range(k):
                                    # Calcula el vector solución de Jacobi para la iteración
                                    x = my_x(H, v, x0)
                                    print(f"  En la Iteración {i+1} el vector resultante x es: {print_matrix(x)}, con un vector x-1: {print(x0)}")
                                    # Verifica el error a partir de la segunda iteración
                                    if i < 0:
                                        # Calcula el error 
                                        e = linalg.norm(x - x0)
                                        print(f"  El error es esta iteración {i+1} es de: {e}")
                                    # Reasigna el nuevo X0 como el X del paso anterior
                                    if (e < tol):
                                        break
                                    x0 = x

                                return x, e

## Calculates H
def my_H(dinv, L, U):
    LU = L + U
    H = -np.dot(dinv, LU)
    return H

def my_x(H, v, x0):
    mul = np.dot(H, x0)
    final = mul + v
    
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

## Print matrix graphics
def print_matrix_graphics(x):
    plt.plot(x)
    plt.gca().set_facecolor('#e9edc9')
    plt.xlabel('Índice')
    plt.ylabel('Solución')
    plt.title('Solución del Sistema Linear usando Método LU')
    plt.show() 
  
# Dataset
non_square = [[1,2,3],[4,5,6]]

det_zero = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

diag_not_dominant = [[1, 2, 3], [4, 1, 6], [7, 8, 1]]

almost_doable_matrix = [[3, 2, 1],[1, 2, 1],[1, 2, 3]]

doable_matrix = [[5, 2, 1],[1, 6, 3],[2, 3, 7]]

b3 = [0, 0, 0]

b2 = [0, 0]

# Prints
## null) Task + Pres
print("                                                                                  ")
print("**********************************************************************************")
print("*                  METODOS NUMERICOS - 2023 - TP METODO LU                       *")
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
print("                                                                                  ")
print("    • Se comprueba si la diagonal es estrictamente dominante:                     ")
my_jacobi(diag_not_dominant, b3, 6, 3)
print("                                                                                  ")
print("    • Se comprueba si la matriz tiene determinante = 0                            ")
my_jacobi(det_zero, b3, 6, 3)
print("                                                                                  ")
print("    • Se comprueba si la norma de H > 1:                                          ") 
my_jacobi(almost_doable_matrix, b3, 6, 3)
print("                                                                                  ")
print("    • Si cumple todos los requisitos, se emiten todos los valores y cálculos:     ") 
my_jacobi(doable_matrix, b3, 6, 3)

## III) Comparación Extra
""" import autograd.numpy as autog
from autograd import jacobian

x = autog.array([5, 3], dtype=float)

def cost(x):
    return x[0]**2 / x[1] - autog.log(x[1])

jacobian_cost = jacobian(cost)

jacobian_cost(autog.array([x, x, x]))
 """

## IV) Conclusions
print("                                                                                  ")
print("**********************************************************************************")
print("*                                  CONCLUSIONES                                  *")
print("**********************************************************************************")
print(" • El Método de descomposición LU, también es es un método semi-numérico, cuyo    ")
print("   criterio de convergencia no es del todo limpio. Como ventaja no requiere       ") 
print("   aproximación inicial, sin embargo continúa siendo un método donde se resuelven ")
print("   dos sistemas lineales.                                                         ")
print("                                                                                  ")
print(" • Es un método inestable ya que si alguno o varios elementos de la diagonal principal ")
print("   son cero, se debe premultiplicar la matriz por alguna elemental de permutación, ") 
print("   para poder aplicar la factorización.                                           ") 
print("                                                                                  ")
print(" • Es un método no tan utilizado ya que requiere matrices n*n que no son siempre  ")
print("   frecuentes.                                                                     ") 
print("                                                                                  ")
print(" • Se utiliza principalmente por su facilidad en resolucionde matrices triangulares.")
print("                                                                                  ")
print(" • Aqui se muetra la varificación de requisitos necesarios para realuzar LU, previo ")
print("   a la resución de la misma. Si pasa todas las verificaciones, se imprime las    ")
print("   matrices para ver las posibilidades de operaciones. En este caso pasa por un método")
print("   genérico, que se describió en la teoría, pero hay que considerar que en caso de")
print("   aparecer unos en la matriz U, deben descontarse operaciones: para ello se uso una")
print("   funcion para contabilizar los unos en la matriz U  y en la matriz L, por separado")
print("   y descontar así esas operaciones despreciadas.                                  ")
print("                                                                                  ")
print(" • NOTA: Las líneas 24 a 36 sirven para calcular la inversa de la matriz.         ")
print("         Como exede lo pedido por el TP ha quedado en desuso pero es funcional.   ")
print("         Se considera que el hecho de tener un determinante distinto a cero de la ")
print("         matriz, ya comprueba que es posible hacer la inversion de la misma.     ")
print("         Puede utilizarse en caso de querer imprimir la inversa de la matriz en cuestión.")
print("                                                                                  ")



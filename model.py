

# ==============================================================================================
#                               IMPORTS
# ==============================================================================================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==============================================================================================
#                               INTERVALS
# ==============================================================================================

class Interval:

    def __init__(self, minimum: float, maximum: float) -> None:
        self.min = minimum
        self.max = maximum
        self.length = maximum - minimum

    def contains(self, x: float) -> bool:
        return self.min <= x and x <= self.max
    
    def __str__(self) -> str:
        return f"[{self.min}, {self.max}]"
    
    def __repr__(self) -> str:
        classname = self.__class__.__name__
        return f"{classname}( min = {self.min}, max = {self.max}, length = {self.length} )"
    
# ==============================================================================================
#                               MODEL
# ==============================================================================================

class Model:

    def __init__(self, precision: int, interval: Interval, rho: float = 0.5) -> None:
        self.nums_step = precision
        self.rho = rho
        self.interval = interval

        self.mesh = None
        self.innerMesh = None
        self.thickness = None

        self.exact = None

    def getThickness(self) -> float:
        return self.interval.length / (self.nums_step + 1)

    def setMesh(self):

        a = self.interval.min
        b = self.interval.max
        N = self.nums_step

        self.thickness = self.getThickness()

        self.mesh = np.linspace(a, b, N + 2)
        self.innerMesh = self.mesh[1:-1]

        return self
    
    def setExact(self, exact: callable):
        self.exact = exact
        return self
    
    def update(self, precision: int):
        self.nums_step = precision
        self.setMesh()
        return self
    
    def hat(self, i: int, x: float) -> float:
        assert 1 <= i <= self.nums_step
        x_im = self.mesh[i - 1]
        x_i = self.mesh[i]
        x_ip = self.mesh[i + 1]

        thickness = self.thickness

        up_slope = (x - x_im) / thickness
        down_slope = (x_ip - x) / thickness
        
        value = np.where((x >= x_im) & (x <= x_i), up_slope, 0)
        value = np.where((x > x_i) & (x <= x_ip), down_slope, value)

        return value
    
    def getRigidityMatrix(self) -> np.array:
        invThickness = 1 / self.thickness
        diag = np.ones(self.nums_step) * 2 * invThickness
        diag1 = - np.ones(self.nums_step - 1) * invThickness
        return np.diag( diag1, -1 ) + np.diag( diag, 0 ) + np.diag( diag1, 1 )
    
    def getRighHandSide(self, u: np.array) -> np.array:
        b = np.ones(self.nums_step)
        h = self.thickness
        innerMesh = self.innerMesh

        return b
    
    def getRighHandSide(self, u: np.array) -> np.array:
        b = np.zeros(self.nums_step)
        h = self.thickness
        mesh = self.mesh

        h2 = h * h

        u_ip = u[2:]
        u_i = u[1:-1]
        u_im = u[:-2]

        u_diff_m = u_i - u_im
        u_diff_p = u_ip - u_i

        sqrt_u_m = np.sqrt(h2 + u_diff_m ** 2)
        sqrt_u_p = np.sqrt(h2 + u_diff_p ** 2)

        middle_points = 0.5 * (mesh[1:] + mesh[:-1])

        b = u_diff_m / sqrt_u_m * middle_points[:-1] - u_diff_p / sqrt_u_p * middle_points[1:]

        return b
    
    def getGradient(self, u: np.array) -> callable:
        # A uBar = b
        # u(x) = ∑ uBar(i) * hat(i, x)
        
        A = self.getRigidityMatrix()
        B = self.getRighHandSide(u)
        uBar = np.linalg.solve( A, B )

        """
        def grad(x: float) -> float:
            # x can be a numpy array
            x = np.atleast_1d(x)

            hat_values = np.zeros((self.nums_step, len(x)))
            
            for i in range(self.nums_step): # may be optimizable for numpy
                hat_values[i] = self.hat(i + 1, x)
            
            return np.dot(uBar, hat_values)
        """
        
        N = self.nums_step

        uBar_ext = np.ones(N + 2)
        uBar_ext[1:-1] = uBar
        uBar_ext[0] = 0
        uBar_ext[-1] = 0

        return uBar_ext
    
    def gradientMethod(self, u0: np.array, tolerance: float = 1e-6, max_iter: int = 100) -> np.array:
        
        u = u0.copy()

        # to enter while loop
        norm_grad = float('inf')
        iter_index = 0

        while norm_grad > tolerance:
            iter_index += 1

            grad = self.getGradient(u)
            u = u - self.rho * grad

            norm_grad = np.linalg.norm(grad)

            #plt.plot(self.mesh, u)
            #plt.show()

            if iter_index > max_iter:
                return u, max_iter
            
        return u, iter_index

    def getInitialFunction(self, alpha: float, beta: float) -> np.array:

        # u0 in C

        a = self.interval.min
        b = self.interval.max
        length = self.interval.length

        def initial(x: float) -> float:
            return (beta - alpha) / length * x - (beta * a - alpha * b) / length

        return initial(self.mesh)
    
    def normLebesgue1(self, function: np.array) -> float:

        length = self.interval.length

        return np.sum(np.abs(function)) / length
    
    def normLebesgueInf(self, function: np.array) -> float:

        return np.max(np.abs(function))
    
    def normSobolev11(self, function: np.array) -> float:

        grad = self.getGradient(function)

        return self.normLebesgue1(function) + self.normLebesgue1(grad)
    
    def __str__(self) -> str:
        return f"""Model(
    precision = {self.nums_step},  # number of steps to build the mesh
    rho = {self.rho},  # constant step of gradient method
    interval = {self.interval},  # interval to build on
    hasMesh = {not self.mesh is None},  # is mesh set ?
    exact = {self.exact}  # exact solution (if set)
)"""
    
    def __repr__(self) -> str:
        classname = self.__class__.__name__
        return f"""{classname}(
    precision = {self.nums_step},  # number of steps to build the mesh
    rho = {self.rho},  # constant step of gradient method
    interval = {self.interval},  # interval to build on
    hasMesh = {not self.mesh is None},  # is mesh set ?
    exact = {self.exact}  # exact solution (if set)
)"""
    
# ==============================================================================================
#                               MODEL
# ==============================================================================================

class Problem:

    def __init__(self, exact: callable, interval: Interval, max_iter: int = 2000, tolerance: float = 1e-6) -> None:

        self.exact = exact
        if self.exact is None:
            raise ValueError("The exact function is not defined. An exact solution in this context is mandatory.")
        self.interval = interval

        self.alpha = exact(interval.min)
        self.beta = exact(interval.max)

        self.model = Model(
            128,
            interval
        ).setMesh().setExact(exact)

        self.max_iter = max_iter
        self.tolerance = tolerance

        self.omega = None


    def plotApprox(self, precision: int, save_path: str = None):

        self.model.update(precision)

        mesh = self.model.mesh

        u0 = self.model.getInitialFunction(self.alpha, self.beta)
        uapp, iterations = self.model.gradientMethod(u0, self.tolerance, self.max_iter)

        plt.plot(mesh, uapp, label = f"Solution approximée : en {iterations} itérations")
        plt.plot(mesh, u0, label = "Solution initiale")


        if self.model.exact:
            """
            uexa = self.model.exact(mesh)
            plt.plot(mesh, uexa, label = "Solution exacte")
            """
            x = np.linspace(self.interval.min, self.interval.max, 1024)
            uexa = self.model.exact(x)
            plt.plot(x, uexa, label = "Solution exacte")

        plt.legend()
        plt.grid(True)
        plt.title(f"Approximation avec N = {precision}: nombre de pas")

        if not save_path is None:
            plt.savefig(save_path)
        plt.show()

    def plotErrors(self, save_path: str = None):

        N_params = [16, 32, 64, 128, 256, 512]

        thickness_array = []

        errsL1 = []
        errsW11 = []
        errsLmax = []

        for N in N_params:
    
            self.model.update(N)

            thickness = self.model.getThickness()
            thickness_array.append(thickness)

            mesh = self.model.mesh

            u0 = self.model.getInitialFunction(self.alpha, self.beta)
            uapp, iterations = self.model.gradientMethod(u0, self.tolerance, self.max_iter)
            
            if self.model.exact is None:
                raise ValueError("The exact function is not defined. Please, use .setExact(function: callable) to define an exact solution.")

            uexa = self.model.exact(mesh)

            diff = uapp - uexa

            errL1 = self.model.normLebesgue1(diff)
            errW11 = self.model.normSobolev11(diff)
            errLmax = self.model.normLebesgueInf(diff)

            errsL1.append(errL1)
            errsW11.append(errW11)
            errsLmax.append(errLmax)

        # Affichage classique
        plt.plot(N_params, errsL1, label = "Erreur L1")
        plt.plot(N_params, errsLmax, label = "Erreur Linf")
        plt.plot(N_params, errsW11, label = "Erreur W11")
        plt.legend()
        plt.grid(True)
        plt.title(f"Évolution de l'erreur en fonction du nombre de pas")
        plt.show()

        plt.plot(thickness_array, errsL1, label = "Erreur L1")
        plt.plot(thickness_array, errsLmax, label = "Erreur Linf")
        plt.plot(thickness_array, errsW11, label = "Erreur W11")
        plt.legend()
        plt.grid(True)
        plt.title(f"Évolution de l'erreur en fonction de la finesse du maillage")
        plt.show()

        # Affichage loglog
        N_params_log = np.log(N_params)
        thickness_array_log = np.log(thickness_array)

        errsL1_log = np.log(errsL1)
        errsLmax_log = np.log(errsLmax)
        errsW11_log = np.log(errsW11)

        plt.plot(N_params_log, errsL1_log, label = "Erreur L1")
        plt.plot(N_params_log, errsLmax_log, label = "Erreur Linf")
        plt.plot(N_params_log, errsW11_log, label = "Erreur W11")
        plt.grid(True)
        plt.title(f"Évolution de l'erreur en fonction du nombre de pas en échelle log-log")
        plt.legend()

        if not save_path is None:
            plt.savefig(save_path)
            
        plt.show()

        plt.plot(thickness_array_log, errsL1_log, label = "Erreur L1")
        plt.plot(thickness_array_log, errsLmax_log, label = "Erreur Linf")
        plt.plot(thickness_array_log, errsW11_log, label = "Erreur W11")
        plt.grid(True)
        plt.title(f"Évolution de l'erreur en fonction de la finesse du maillage en échelle log-log")
        plt.legend()
        plt.show()

        slope_L1 = (errsL1_log[-1] - errsL1_log[-2]) / (thickness_array_log[-1] - thickness_array_log[-2])
        slope_Lmax = (errsLmax_log[-1] - errsLmax_log[-2]) / (thickness_array_log[-1] - thickness_array_log[-2])
        slope_W11 = (errsW11_log[-1] - errsW11_log[-2]) / (thickness_array_log[-1] - thickness_array_log[-2])

        print(f"Pente de l'erreur L1:   {slope_L1}")
        print(f"Pente de l'erreur Lmax: {slope_Lmax}")
        print(f"Pente de l'erreur W11:  {slope_W11}")

    def getInterpolatedFunction(self) -> callable:

        u0 = self.model.getInitialFunction(self.alpha, self.beta)
        uapp, iterations = self.model.gradientMethod(u0, self.tolerance, self.max_iter)

        mesh = self.model.mesh

        def u(x: float) -> float:
            index = np.searchsorted(mesh, x) - 1
            index = np.maximum(index, 0)
            value = Problem.interpolate(uapp[index], uapp[index + 1], Interval(mesh[index], mesh[index + 1]), x)
            return value
        
        return u
    
    def getSurfaceFunctionWith(self, function: callable) -> callable:
        
        def surface(x, y):
            r = np.sqrt(x * x + y * y)
            return function(r)
        
        return surface
    
    def setOmegaMesh(self, delta: float = 2.0, points: int = 256):

        radius_min = self.interval.min
        radius_max = self.interval.max

        half_side_length = radius_max + delta

        X = np.linspace(-half_side_length, half_side_length, points)
        Y = np.linspace(-half_side_length, half_side_length, points)

        X, Y = np.meshgrid(X, Y)

        condition = (radius_min**2 < X**2 + Y**2) & (X**2 + Y**2 < radius_max**2)

        X_filtered = X[condition]
        Y_filtered = Y[condition]
        
        self.omega = (X_filtered, Y_filtered)

        return self
    
    def showOmega(self) -> None:

        if self.omega is None:
            raise ValueError("Omega is not set. Please, use .setOmegaMesh().")
        
        X, Y = self.omega

        plt.figure(figsize=(6,6))
        plt.scatter(X, Y, color = 'blue', s = 1)
        plt.title('Ensemble Ω')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.show()

    def plotSurface(self, function: callable) -> None:

        if self.omega is None:
            raise ValueError("Omega is not set. Please, use .setOmegaMesh().")
        
        fig = plt.figure()

        ax = fig.add_subplot(111, projection = '3d')

        X, Y = self.omega
        Z = function(X, Y)

        ax.scatter(X, Y, Z, c = Z, cmap = "magma", marker = 'o')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

    @classmethod
    def lerp(cls, minimum: float, maximum: float, x: float) -> float:
        return (maximum - minimum) * x + minimum

    @classmethod
    def interpolate(cls, minimum: float, maximum: float, interval: Interval, x: float) -> float:
        a = interval.min
        length = interval.length
        y = (x - a) / length
        return cls.lerp(minimum, maximum, y)

    def __repr__(self) -> str:
        classname = self.__class__.__name__
        return f"""{classname}(
    exact = {self.exact},  # exact solution
    interval = {self.interval},  # interval to build on
    alpha, beta = {self.alpha, self.beta},  # edge conditions
    model = {self.model},  # model to approximate functions with a gradient and a finite elements method
    hasOmega = {not self.omega is None},  # is omega set ?
    max_iter = {self.max_iter},  # number of iterations the model can perform until it stops
    tolerance = {self.tolerance}  # error that we tolerate to stop the model and consider that we have convergence
)"""
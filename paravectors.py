import numpy as np
import matplotlib.pyplot as plt

epsilon = 1e-3

#region Classes
class Paravector:
    alpha: float
    theta: float
    beta: float

    def __init__(self, alpha, theta, beta):
        self.alpha = alpha 
        self.theta = theta
        self.beta = beta

    def as_local_x(self):
        a = np.tan(-self.beta) / self.alpha
        return lambda x: a * x * (x - self.alpha)

    def as_global_x(self, h, k):
        a = np.tan(-self.beta) / self.alpha

        def func(x):
            if np.abs(np.sin(self.theta)) < epsilon: # no rotation
                local_x = x - h
                local_y = a * local_x * (local_x - self.alpha)
                return k + local_y

            A = a * np.sin(self.theta)
            B = -(np.cos(self.theta) + a * self.alpha * np.sin(self.theta))
            C = x - h

            discriminant = B ** 2 - 4 * A * C
            if discriminant < 0:
                return None
            local_x = (-B - np.sqrt(discriminant)) / (2 * A)

            local_y = a * local_x * (local_x - self.alpha)
            y = k + local_x * np.sin(self.theta) + local_y * np.cos(self.theta)

            return y

        return func

    def as_local_y(self, h, k):
        a = np.tan(-self.beta - (np.pi / 2)) / self.alpha
        return lambda y: a * (y - k) * (y - k + self.alpha) + h

    def as_global_y(self, h, k):
        theta_prime = self.theta + (np.pi / 2)
        return lambda y: y * np.tan(theta_prime) + self.as_local_y(h, k)(y / np.cos(theta_prime))

class Chain:
    paravectors: list[Paravector]

    def __init__(self, paravectors):
        self.paravectors = paravectors

    def domain_end(self, h):
        hs = [h]
        for i in range(len(self.paravectors)):
            hs.append(hs[i] + self.paravectors[i].alpha * np.cos(self.paravectors[i].theta))
        return hs[-1]

    def as_function(self, h: float, k: float, x_compliant: bool = True):
        segment_starts_x = [h]
        segment_starts_y = [k]
        funcs = []

        for i, pv in enumerate(self.paravectors):
            if x_compliant:
                local_func = pv.as_global_x(0, 0)
            else:
                local_func = pv.as_global_y(0, 0)

            funcs.append(local_func)

            next_x = segment_starts_x[-1] + pv.alpha * np.cos(pv.theta)
            next_y = segment_starts_y[-1] + pv.alpha * np.sin(pv.theta)

            segment_starts_x.append(next_x)
            segment_starts_y.append(next_y)

        def func(x):
            for j in range(len(self.paravectors)):
                x_start = segment_starts_x[j]
                x_end = segment_starts_x[j + 1]

                if x_start <= x <= x_end:
                    local_x = x - x_start
                    local_y = funcs[j](local_x)
                    return segment_starts_y[j] + local_y

            return None

        return func


#endregion
#region Functions
def plot_local(upsilon: Paravector, path: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))

    x_end = upsilon.alpha
    y_end = 0
    func = upsilon.as_local_x()

    x_values = np.linspace(0, x_end, 100)
    y_values = [func(x) for x in x_values]

    ax.plot(x_values, y_values, label='upsilon')

    beta_x_end = np.cos(upsilon.beta)
    beta_y_end = np.sin(upsilon.beta)
    ax.axline((0, 0), (beta_x_end, beta_y_end), color='red', label='beta')

    ax.axline((0, 0), (x_end, y_end), color='green', label='span')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Paravector Local Visualization')
    ax.grid(True)
    ax.legend()

    ax.set_aspect('equal')

    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches='tight')

    plt.show()

def plot(upsilon: Paravector, path: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))

    x_start = 0
    x_end = upsilon.alpha * np.cos(upsilon.theta)
    y_start = 0
    y_end = upsilon.alpha * np.sin(upsilon.theta)
    func = upsilon.as_global_x(x_start, y_start)

    x_values = np.linspace(x_start, x_end, 100)
    y_values = [func(x) for x in x_values]

    ax.plot(x_values, y_values, label='upsilon')

    beta_x_end = np.cos(upsilon.beta + upsilon.theta)
    beta_y_end = np.sin(upsilon.beta + upsilon.theta)
    ax.axline((x_start, y_start), (beta_x_end, beta_y_end), color='red', label='beta')

    ax.axline((x_start, y_start), (x_end, y_end), color='green', label='span')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Paravector Visualization')
    ax.grid(True)
    ax.legend()

    ax.set_aspect('equal')

    ax.set_xlim(-0.1, upsilon.alpha + 0.1)
    ax.set_ylim(-0.3, max(y_values) + 0.2)

    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches='tight')

    plt.show()

def plot_chain(chain: Chain, path: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    x_values = np.linspace(0, chain.domain_end(0), 100)
    func = chain.as_function(0, 0)
    y_values = [func(x) for x in x_values]

    ax.plot(x_values, y_values, label='Upsilon')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Paravector Chain Visualization')
    ax.grid(True)
    ax.legend()

    ax.set_aspect('equal')

    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches='tight')

    plt.show()

def plot_global(*upsilons, path: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    for upsilon in upsilons:
        (pv, placementx, placementy) = upsilon
        x_values = np.linspace(placementx, placementx + pv.alpha * np.cos(pv.theta), 100)
        func = pv.as_global_x(placementx, placementy)
        y_values = [func(x) for x in x_values]
        ax.plot(x_values, y_values, label=f'({pv.alpha}, {pv.theta}, {pv.beta})')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Paravector Chain Visualization')
    ax.grid(True)
    ax.legend()

    ax.set_aspect('equal')

    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches='tight')

    plt.show()


#endregion
#region Examples
def paravector_example(span, theta, beta, show = True):
    upsilon = Paravector(span, theta, beta)
    if show: plot(upsilon)
    return upsilon

def vector_example(span, theta, show = True):
    upsilon = Paravector(span, theta, epsilon)
    if show: plot(upsilon)
    return upsilon

def simple_paravector(show = True): return paravector_example(4, np.pi / 4, np.pi / 4 - 0.1, show)

def sin_like_chain1(show = True):
    pi_over_4 = np.pi / 4
    pi_over_2 = np.pi / 2

    upsilon_1 = Paravector(1, pi_over_4, epsilon)
    upsilon_2 = Paravector(pi_over_2, 0, (np.pi / 4) - epsilon)
    upsilon_3 = Paravector(2, -np.pi / 4, epsilon)
    upsilon_4 = Paravector(pi_over_2, 0, -(np.pi / 4) - epsilon)

    chain = Chain([upsilon_1, upsilon_2, upsilon_3, upsilon_4])
    plot_chain(chain)
    return chain

def sin_like_chain2(show = True):
    upsilon_1 = Paravector(np.pi, 0, 1.5 * np.pi / 6)
    upsilon_2 = Paravector(np.pi, 0, -1.5 * np.pi / 6)

    chain = Chain([upsilon_1, upsilon_2])
    plot_chain(chain)
    return chain

def happy_face():
    eye1 = Paravector(2, 0,  np.pi / 4)
    eye2 = Paravector(2, 0,  np.pi / 4)
    mouth = Paravector(6, 0, -np.pi / 4)
    plot_global((eye1, 1, 2), (eye2, 3.5, 2), (mouth, 0, 0))
#endregion

if __name__ == "__main__":
    happy_face()
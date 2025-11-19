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

    def as_local_x(self, h, k):
        a = np.tan(-self.beta) / 2
        return lambda x: a * (x - h) * (x - h - self.alpha) + k

    def as_global_x(self, h, k):
        return lambda x: x * np.tan(self.theta) + self.as_local_x(h, k)(x / np.cos(self.theta))

    def as_local_y(self, h, k):
        a = np.tan(self.beta + (np.pi / 2)) / 2
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
            hs.append(hs[i-1] + self.paravectors[i-1].alpha * np.cos(self.paravectors[i-1].theta))
        return hs[-1]

    def as_function(self, h: float, k: float, x_compliant: bool = True):
        # Track the end point of each segment
        segment_starts_x = [h]
        segment_starts_y = [k]
        funcs = []

        for i, pv in enumerate(self.paravectors):
            # Get the local function for this paravector
            if x_compliant:
                local_func = pv.as_global_x(0, 0)  # Start at origin in local coords
            else:
                local_func = pv.as_global_y(0, 0)

            funcs.append(local_func)

            # Calculate where this segment ends
            next_x = segment_starts_x[-1] + pv.alpha * np.cos(pv.theta)
            next_y = segment_starts_y[-1] + local_func(pv.alpha * np.cos(pv.theta))

            segment_starts_x.append(next_x)
            segment_starts_y.append(next_y)

        def func(x):
            for j in range(len(self.paravectors)):
                x_start = segment_starts_x[j]
                x_end = segment_starts_x[j + 1]

                if x_start <= x <= x_end:
                    # Transform x to local coordinates
                    local_x = x - x_start
                    local_y = funcs[j](local_x)
                    return segment_starts_y[j] + local_y

            return None

        return func


#endregion
#region Functions
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
    ax.scatter(x_start, y_start, label='start')
    ax.scatter(x_end, y_end, label='end')

    beta_x_end = np.cos(upsilon.beta + upsilon.theta)
    beta_y_end = np.sin(upsilon.beta + upsilon.theta)
    ax.axline((x_start, y_start), (beta_x_end, beta_y_end), color='red', label='beta')

    ax.axline((x_start, y_start), (x_end, y_end), color='green', label='spen')

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

#endregion
#region Examples
def paravector_example(span, theta, beta, show = True):
    upsilon = Paravector(span, theta, beta)
    if show: plot(upsilon)
    return upsilon

def vector_example(span, theta, show = True):
    upsilon = Paravector(span, theta, 0)
    if show: plot(upsilon)
    return upsilon

def simple_paravector(show = True): return paravector_example(1, np.pi / 4, np.pi / 4 - 0.1, show)

def sin_like_chain(show = True):
    upsilon_1 = Paravector(1, np.pi / 4, np.pi / 4)
    upsilon_2 = Paravector(1, -np.pi / 4, np.pi / 4)
    upsilon_3 = Paravector(1, -np.pi / 4, np.pi / 4)
    upsilon_4 = Paravector(1, np.pi / 4, np.pi / 4)
    chain = Chain([upsilon_1, upsilon_2, upsilon_3, upsilon_4])
    plot_chain(chain)
    return chain

#endregion

# Main execution block
if __name__ == "__main__":
    sin_like_chain()

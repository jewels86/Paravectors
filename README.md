# Paravectors
3-component curved vectors for the description of systems.

## The Math
Paravectors are like polar coordinates with an extra component:

$$
\upsilon = (\alpha, \theta, \beta)
$$

$\alpha$ and $\theta$ are the span length and direction of the paravector (like the components of a normal vector), and $\beta$ represents the curvature- specifically, the angle of a line tangent at the first zero.

In the local frame, we can represent the upsilon as it's original parabola with:

$$
\upsilon(x) = ax \ \cdot \ (x - \alpha) 
$$

We can find $a$ with:

$$
f(x) = bx(x - \alpha)
$$

$$
f'(x) = \frac{d}{dx} bx(x - \alpha) = 2bx - b\alpha
$$

$$
f'(0) = -b\alpha
$$

So the angle of the line tangent at 0 is:

$$
\beta = tan^{-1}(-b\alpha)
$$

Thus, for $\upsilon(x)$:

$$
a = \frac{tan(-\beta)}{\alpha}
$$

To find the original coefficient.

### Function Approximation
We can chain paravectors:

$$
(\upsilon_1, \upsilon_2, ... \upsilon_n) = ((\alpha_1, \theta_1, \beta_1), \ (\alpha_2, \theta_2, \beta_2), ... \ (\alpha_n, \theta_n, \beta_n))
$$

First, to create an $\Upsilon(x)$ we need to define $\upsilon(x)$ in the global frame. We'll use parametric equations:

$$
x' = h + x\cos\theta - y\sin\theta
$$

$$
y' = k + x\sin\theta + y\cos\theta = \Upsilon(x)
$$

Rearranging and subsituting:

$$
0 = h + x\cos\theta - (ax(x - \alpha)\sin\theta) - x'
$$

$$
0 = h + x\cos\theta - (ax^2\sin\theta - ax\alpha\sin\theta) - x'
$$

$$
0 = h + x\cos\theta - ax^2\sin\theta + ax\alpha\sin\theta - x'
$$

$$
0 = -ax^2\sin\theta + x\cos\theta + ax\alpha\sin\theta + h - x' 
$$

$$
0 = ax^2\sin\theta - x\cos\theta - ax\alpha\sin\theta + x' - h
$$

We can use the quadratic formula to solve with $A = a\sin\theta$, $B = -(\cos\theta + a\alpha\sin\theta)$, $C = x' - h$ so that:

$$
x = \frac{-B \pm \sqrt{B^2 - 4AC}}{2A}
$$

Geometrically, $x$ is the input to $\upsilon(x)$ where it's output lines up with $x'$. Now we can solve for the global $y$:

$$
y' = k + x\sin\theta + \upsilon_n(x)\cos\theta = \Upsilon_n(x)
$$

We can chain the $\Upsilon_n$ function with others now in a piecewise.

## Uses
Paravectors allow you to compress parabolic curvature to a 3-tuple. Since they can approximate functions with the correct parameters, they can be very useful for compressing mathematical functions into chains of serializable data.

My [ParaSharp](https://github.com/jewels86/ParaSharp) project implements paravectors in C# that can be trained into serialized versions of functions using autodiff. 

## The Attached Code
You can import `paravectors` into your own project or use it from command line:
```py
from paravectors import Paravector, Chain
```
or:
```py
python paravectors.py
```
(You can change which example it runs in the if name is main block!)

Paravectors expose:
```py
class Paravector:
    alpha: float
    theta: float
    beta: float

    def as_local_x(self): ...
    def as_global_x(self, h, k): ...

    def as_local_y(self): ...
    def as_global_y(self, h, k): ...
```
Chains expose:
```py
class Chain:
    paravectors: list[Paravector]
    
    def domain_end(self, h): ...
    def as_function(self, h: float, k: float, x_compliant: bool = True): ...
```
`paravectors` also has some plotting functions and examples:
```py
def plot_local(upsilon: Paravector, path: str = None): ...
def plot(upsilon: Paravector, path: str = None): ...
def plot_chain(chain: Chain, path: str = None): ...
def plot_global(*upsilons, path: str = None): ...

def paravector_example(span, theta, beta, show = True): ...
def vector_example(span, theta, show = True): ...
def simple_paravector(show = True): ...
def sin_like_chain1(show = True): ...
def sin_like_chain2(show = True): ...
def happy_face(): ...
```

## Why I Made This
I realized you could represent curves in terms of the first derivative and got excited because I thought these would be really helpful. They are not. 

## Possible Expansions
This could be done in 3D, I assume. You could represent rotations in terms of the parabolic path of a single point, but it would be an approximation at best (elliptical relationships are different from parabolic ones). 

You could probably repeat my derivation process for the elliptical relationship but you may not end up with a quadratic to solve. Either way, paravectors approximate better the lower the eccentricity, so you can break it up into smaller segments if you'd like.

## Contributing
If you use this for anything, let me know! I'd love to see your use case.

If I made a mistake somewhere, open an issue and I'll fix it. You can also create PRs to correct the code if you'd like.

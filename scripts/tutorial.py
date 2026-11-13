from katz.prior import KatzPrior

n = 2
basis_functions = [
    ["a", "x"],
    ["sqrt", "exp", "log", "sin", "cos", "arcsin", "tanh"],
    ["+", "-", "*", "/", "pow"],
]

# Kneser-Ney smoothing with a root context is the default; add smoothing="katz",
# root_context=False, normalise=False for the Katz model of version 0.2.1 (Bartlett et al.
# (2023) used version 0.1.0, which differs in the right-child model; see README)
kp = KatzPrior(
    n, basis_functions, "../data/FeynmanEquations.csv", "../data/NewFeynman.csv"
)

for eq in ["x0**2", "sin(x0) + sin(x1)", "sin(sin(x0+x1))"]:
    p = kp.logprior(eq)
    print(eq, p)

for eq in [
    ["+", "x0", "x0"],
    ["*", "2", "x0"],
    ["+", "x0", "x1"],
    ["+", "sin", "x0", "sin", "x1"],
]:
    p = kp.logprior(eq)
    print(eq, p)

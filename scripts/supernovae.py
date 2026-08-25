import numpy as np
import pandas as pd
import sympy
from esr.fitting.likelihood import PanthLikelihood
from fit_benchmark import apply_language_prior, process_fit
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

likelihood = PanthLikelihood()
all_comp = np.arange(1, 11)

do_language = False
do_process = False
print_results = True

if do_language:
    for comp in all_comp:
        apply_language_prior(likelihood, comp, tmax=5)
        comm.Barrier()
if rank == 0:
    dirname = likelihood.out_dir + "/"

    if do_process:
        nx = len(likelihood.xvar)
        process_fit(dirname, all_comp, nx)

    if print_results:
        df = pd.read_csv(dirname + "selection_summary.csv", delimiter=";")
        print("\nYOU WILL HAVE TO INSPECT THESE YOURSELF AS DUPLICATES MAY REMAIN\n")
        for m in range(df.shape[0]):
            res = []
            for i in range(8):
                f = df[f"f{i}"][m]
                if str(f) != str(None):
                    _, eq, _ = likelihood.run_sympify(f, try_integration=False)
                    f = sympy.latex(eq)
                if m == 1:
                    loss = -np.log(df[f"loss{i}"][m] / df["loss0"][m])
                # res.append('$%s$ & %.2e'%(f, loss))
                else:
                    loss = df[f"loss{i}"][m] - df["loss0"][m]
                res.append(f"${f}$ & {loss:.2f}")
            res = f"{m+1} & " + " & ".join(res) + " \\\\"
            print(res)
            print("\\hline")

        # Now do printing for a LaTex Document to visualise results
        print("\nPrinting latex to: ", dirname + "/selection_results.tex")
        with open(dirname + "/selection_results.tex", "w") as fout:
            print("\\documentclass[12pt,fleqn]{article}", file=fout)
            print("\\usepackage{amsmath,amssymb}", file=fout)
            print("\\usepackage{color}", file=fout)
            print(
                "\\usepackage[left=1.3cm, right=1.5cm, top=1.5cm, bottom=1.5cm]{geometry}",
                file=fout,
            )
            print("\\usepackage{graphicx}\n", file=fout)
            print("\\begin{document}", file=fout)
            print("\\begin{center}", file=fout)
            print("\\section*{Supernovae Results}", file=fout)
            print("\\today{}", file=fout)
            print("\\end{center}\n", file=fout)

            for m in range(df.shape[0]):
                print(f"\n\\textbf{{Method {m+1}}}\n\n\\begin{{enumerate}}", file=fout)

                for i in range(10):
                    f = df[f"f{i}"][m]
                    if str(f) != str(None) and str(f) != "nan":
                        _, eq, _ = likelihood.run_sympify(f, try_integration=False)
                        f = sympy.latex(eq)
                    print(f"\t\\item ${f}$", file=fout)

                print("\\end{enumerate}\n", file=fout)

            print("\n\\end{document}", file=fout)

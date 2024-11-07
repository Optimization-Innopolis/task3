import numpy as np

from main import north_west_corner, vogel_approximation, russell_approximation


def run_test_cases():
    test_cases = [
        {
            "supply": [15, 25, 10],
            "demand": [5, 15, 15, 15],
            "costs": np.array([
                [4, 8, 8, 6],
                [6, 4, 3, 5],
                [5, 7, 6, 4]
            ]),
            "description": "Test Case 1: Simple balanced costs"
        },
        {
            "supply": [20, 30, 25],
            "demand": [10, 20, 15, 30],
            "costs": np.array([
                [8, 6, 10, 9],
                [9, 12, 13, 7],
                [14, 9, 16, 5]
            ]),
            "description": "Test Case 2: Moderate balanced costs"
        },
        {
            "supply": [25, 35, 15],
            "demand": [15, 25, 10, 25],
            "costs": np.array([
                [12, 14, 17, 15],
                [15, 13, 18, 10],
                [14, 16, 12, 19]
            ]),
            "description": "Test Case 3: High balanced costs"
        }
    ]

    latex_output = []

    for idx, case in enumerate(test_cases):
        print(f"Running {case['description']}")

        # Run each method
        nw_corner_sol = north_west_corner(case["supply"].copy(), case["demand"].copy())
        vogel_sol = vogel_approximation(case["supply"].copy(), case["demand"].copy(), case["costs"])
        russell_sol = russell_approximation(case["supply"].copy(), case["demand"].copy(), case["costs"])

        # LaTeX formatting
        latex_output.append(f"\\section*{{{case['description']}}}")
        latex_output.append("\\begin{table}[h!]")
        latex_output.append("\\centering")
        latex_output.append("\\begin{tabular}{|c|c|c|}")
        latex_output.append("\\hline")
        latex_output.append("Method & Initial Basic Feasible Solution & Cost \\\\ \\hline")

        def solution_to_latex(matrix):
            return "$\\begin{pmatrix}" + " \\\\ ".join(
                [" & ".join(map(str, row)) for row in matrix]
            ) + "\\end{pmatrix}$"

        # Compute total costs for each solution
        nw_cost = np.sum(nw_corner_sol * case["costs"])
        vogel_cost = np.sum(vogel_sol * case["costs"])
        russell_cost = np.sum(russell_sol * case["costs"])

        # Append each solution to LaTeX table
        latex_output.append(f"North-West Corner & {solution_to_latex(nw_corner_sol)} & ${nw_cost}$ \\\\ \\hline")
        latex_output.append(f"Vogel's Approximation & {solution_to_latex(vogel_sol)} & ${vogel_cost}$ \\\\ \\hline")
        latex_output.append(
            f"Russell's Approximation & {solution_to_latex(russell_sol)} & ${russell_cost}$ \\\\ \\hline")

        latex_output.append("\\end{tabular}")
        latex_output.append("\\end{table}")
        latex_output.append("\\vspace{1em}\n")

    # Output the LaTeX formatted results
    print("\n".join(latex_output))


# Call the function to run the tests and print the LaTeX output
run_test_cases()

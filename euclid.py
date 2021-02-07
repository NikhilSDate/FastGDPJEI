from sympy import *
import numpy as np


def length_equation(p1, p2, length):
    x1, y1 = p1
    x2, y2 = p2
    return Eq((x2 - x1) ** 2 + (y2 - y1) ** 2, length ** 2)


def angle_equation(p1, p2, p3, theta):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x_diff_1 = x1 - x3
    x_diff_2 = x2 - x3
    y_diff_1 = y1 - y3
    y_diff_2 = y2 - y3

    dot = x_diff_1 * x_diff_2 + y_diff_1 * y_diff_2
    length_1 = sqrt(x_diff_1 ** 2 + y_diff_1 ** 2)
    length_2 = sqrt(x_diff_2 ** 2 + y_diff_2 ** 2)
    return Eq(dot / (length_1 * length_2), cos(theta))


def angle_expression(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x_diff_1 = x1 - x3
    x_diff_2 = x2 - x3
    y_diff_1 = y1 - y3
    y_diff_2 = y2 - y3
    dot = x_diff_1 * x_diff_2 + y_diff_1 * y_diff_2
    length_1 = sqrt(x_diff_1 ** 2 + y_diff_1 ** 2)
    length_2 = sqrt(x_diff_2 ** 2 + y_diff_2 ** 2)
    angle = acos(simplify(dot / (length_1 * length_2)))
    return angle


def length_expression(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return simplify(sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))


def make_subsitutions(symbols, equations):
    num_subs = len(symbols) - len(equations)
    symbols_list = list(symbols)
    print(symbols_list)

    if num_subs <= 3:
        print('Problem is perfectly determinable')
    else:
        print('Problem is not perfectly determinable')

    print(num_subs)

    for i in range(3):
        for n, equation in enumerate(equations):
            equations[n] = equation.subs(symbols[i], 0)
        symbols_list.remove(symbols[i])
        print()
    subs_made = 3

    if num_subs > 3:
        for i in range(3, len(symbols)):
            sub_allowed = True
            for equation in equations:
                if simplify(equation.subs(symbols[i], 0)) == False:
                    sub_allowed = False
                    break
            if sub_allowed:
                for n, equation in enumerate(equations):
                    equations[n] = equation.subs(symbols[i], 0)
                subs_made = subs_made + 1
                symbols_list.remove(symbols[i])
            if subs_made == num_subs:
                break

    return symbols_list, equations





def solve_problem(symbols, equations, question):
    symbol_values = dict()
    for symbol in symbols:
        symbol_values[symbol] = 0
    symbols_to_solve, substituted_equations = make_subsitutions(symbols, equations)
    print(symbols_to_solve)
    solutions = solve(substituted_equations, symbols_to_solve)
    pprint(solutions)
    if len(solutions) == 0:
        print("Could not find solution")
        return

    solution=solutions[0]
    for n,symbol in enumerate(symbols_to_solve):
        symbol_values[symbol]=solution[n]
    pprint(symbol_values)
    for symbol, value in symbol_values.items():

        question = question.subs(symbol, value)
    answer = simplify(question)
    return answer


# xO, yO, xA, yA, xB, yB, xC, yC = symbols('xO yO xA yA xB yB xC yC')
#
# f1 = length_equation((xA, yA), (xO, yO), length=1)
#
# f2 = length_equation((xB, yB), (xO, yO), length=1)
# f3 = length_equation((xC, yC), (xO, yO), length=1)
#
# f4 = angle_equation((xA, yA), (xB, yB), (xO, yO), theta=pi)
# equations = [f1, f2, f3, f4]
# symbols = symbols('xO yO xA yA xB yB xC yC')
# question = angle_expression((xA,yA),(xB, yB), (xC, yC))
# print(solve_problem(symbols, equations, question))

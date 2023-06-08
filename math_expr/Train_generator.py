import itertools
import random
import json
from executor import write_figure, execute_program_line

import warnings

TOTAL_FUNCTIONS = 5  # change this to up the number of max functions
DATASET_SIZE = 10
MAX_POLY_NUMBER = 5

RANGE = [-10, 10]


def tokens_to_expression(tokens):
    """
    :param tokens: arithmetic tokens to be translated to expression e.g. ['1','2','3','4']
    :return: expression e.g. sinc(), exp(), sin(), ln()
    """
    expression = []
    poly_numbers = []

    for token in tokens:
        if token in ['1', '2', '3', '4']:
            if poly_numbers:
                poly_string = mapping['5'] + '(' + json.dumps(poly_numbers) + ')'
                expression.append(poly_string)
                poly_numbers = []
            expression.append(mapping[token])
        elif token in ['5']:
            continue
        else:
            poly_numbers.append(float(mapping[token]))

    if poly_numbers:
        poly_string = mapping['5'] + '(' + json.dumps(poly_numbers) + ')'
        expression.append(poly_string)

    return ", ".join(expression)


mapping = {
    "1": "sinc", "2": "exp", "3": "sin", "4": "ln", "5": "poly", 
    "6": "0", "7": "0.2", "8": "0.4", "9": "0.6", "10": "0.8", "11": "1.0"
}

poly_params = [list(itertools.product(range(6, 12), repeat=r)) for r in range(1, MAX_POLY_NUMBER)] # If last is 5 -> max poly parameters is 4
poly_params = [item for sublist in poly_params for item in sublist]

function_permutations = list(itertools.permutations(range(1, 6))) # generates all possible orderings of the numbers 1 to 5 in order to always use a function from 1 to 5 if possible
expressions_set, expressions, tokens_list = set(), [], []
for func_perm in function_permutations:
    for poly in poly_params:
        extended_func_perm = random.sample(func_perm * ((TOTAL_FUNCTIONS + len(func_perm) - 1) // len(func_perm)),
                                           TOTAL_FUNCTIONS)
        expression, tokens = [], []
        for func_key in extended_func_perm:
            tokens.append(str(func_key))
            if func_key == 5:
                if all([p == '6' for p in poly]):
                    continue
                expression.append(f'{mapping[str(func_key)]}({json.dumps(list(map(lambda x: float(mapping[str(x)]), poly)))})')
                tokens.extend([str(p) for p in poly])
            else:
                expression.append(f'{mapping[str(func_key)]}()')
        # Exceptions
        func_str_list = [mapping[str(func_key)] for func_key in extended_func_perm]
        exp_ln_undesired = "exp" in func_str_list and "ln" in func_str_list and abs(func_str_list.index("exp") - func_str_list.index("ln")) == 1
        sin_sinc_undesired = "sin" in func_str_list and "sinc" in func_str_list and abs(func_str_list.index("sin") - func_str_list.index("sinc")) == 1
        ln_undesired = any(func_str_list[i] == func_str_list[i+1] == 'ln' for i in range(len(func_str_list)-1))
        if exp_ln_undesired or sin_sinc_undesired or ln_undesired:
            continue
            
        expr_str = ", ".join(expression)
        if expr_str not in expressions_set:  # check for duplicate expressions
            expressions.append(expr_str)
            expressions_set.add(expr_str)
            tokens_list.append(tokens)

# Create a dataset of DATASET_SIZE images
for i in range(min(DATASET_SIZE, len(expressions))):
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            x, y = execute_program_line(expressions[i])
        write_figure(tokens_to_expression(tokens_list[i]), x, y, RANGE, RANGE, pure=True)
        # uncomment this fif you want the output to be the list input
        # write_figure(tokens_list[i], x, y, RANGE, RANGE, pure=True)
    except RuntimeWarning:
        print(f"Skipping expression: {expressions[i]} due to RuntimeWarning")
        continue

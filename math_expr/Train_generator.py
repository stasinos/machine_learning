import itertools
import random
import json
from executor import write_figure, execute_program_line

mapping = {
    "1": "sinc", "2": "exp", "3": "sin", "4": "ln", "5": "poly", 
    "6": "0", "7": "0.2", "8": "0.4", "9": "0.6", "10": "0.8", "11": "1.0"
}

poly_params = [list(itertools.product(range(6, 11), repeat=r)) for r in range(1,5)] # change this for max poly number. If last is 5 than max poly parameters is 4
poly_params = [item for sublist in poly_params for item in sublist]

function_permutations = list(itertools.permutations(range(1, 6))) # generates all possible orderings of the numbers 1 to 5 in order to always use a function from 1 to 5 if possible
total_functions = 5 # change this to up the number of max functions
expressions_set, expressions, tokens_list = set(), [], []
for func_perm in function_permutations:
    for poly in poly_params:
        extended_func_perm = random.sample(func_perm * ((total_functions + len(func_perm) - 1) // len(func_perm)), total_functions)
        expression, tokens = [], []
        for func_key in extended_func_perm:
            tokens.append(str(func_key))
            if func_key == 5:
                expression.append(f'{mapping[str(func_key)]}({json.dumps(list(map(lambda x: float(mapping[str(x)]), poly)))})')
                tokens.extend([str(p) for p in poly])
            else:
                expression.append(f'{mapping[str(func_key)]}()')
        
        expr_str = ", ".join(expression)
        if expr_str not in expressions_set: # check for duplicate expressions
            expressions.append(expr_str)
            expressions_set.add(expr_str)
            tokens_list.append(tokens)

for i in range(100):
    print(f"{expressions[i]}")
    print(f"{tokens_list[i]}")
    print("----------------")
for i in range(100):  
    x, y = execute_program_line(expressions[i])
    write_figure(tokens_list[i], x, y, [-10,10], [-10,10], pure = True) # simply change to expressions[i] if you want the name of the file to be the name of the expression

# Can be used to turn back from tokens to expressions

def tokens_to_expression(tokens):
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

# Example usage:
# expr = tokens_to_expression(['1','2','3','4'])
# print(expr)
# will print sinc(), exp(), sin(), ln()
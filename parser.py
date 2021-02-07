from statement import Statement
import euclid
print("Enter the problem")
equations=list()
symbol_set=set()
question=None
while True:
    statement_string = input('')
    if statement_string == '':
        break
    statement=Statement(statement_string)
    if statement.is_question():
        question=statement.get_equation()
    equations.append(statement.get_equation())
    symbol_set.union(statement.get_statement_symbols())
symbols_tuple=tuple(symbol_set)
print(solve)










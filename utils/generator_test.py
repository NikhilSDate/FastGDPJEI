from itertools import count
gen = count(10)
def custom_gen(generator):
    while True:
        yield (next(generator)*2)
even_gen = custom_gen(gen)

while True:
    print(next(even_gen))
# 1) Write a normal function that accepts another function as an argument. Output the result of that other function in your “normal” function.

print('/////////// \n The first task:')


def other_function(x):
    return(x ** 2)


def normal_fun(x, y):
    f_x = other_function(x)
    print('The result of other function is:')
    print(f_x)
    print('The result of the normal function is')
    print(f_x + y)
    return(None)


normal_fun(2, 2)


# 2) Call your “normal” function by passing a lambda function – which performs any operation of your choice – as an argument.
print('/////////// \n The second task:')
normal_fun(list(map(lambda x: x ** 2, [2]))
           [0], list(map(lambda y: y ** 3, [3]))[0])

# 3) Tweak your normal function by allowing an infinite amount of arguments on which your lambda function will be executed.
print('/////////// \n The third task:')


def norfuntweak(x, y, *args):
    f_x = other_function(x)
    print('The result of other function is:')
    print(f_x)
    print('The result of the normal function is')
    print(f_x + y + sum(args))
    return(None)


norfuntweak(2, 2, 3, 3)
norfuntweak(lambda x: x * 2, 2, 3, 4)

# 4) Format the output of your “normal” function such that numbers look nice and are centered in a 20 character column.

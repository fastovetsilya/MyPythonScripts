# 1 Create two variables – one with your name and one with your age
name = input('What is your name?')
age = input('What is syour age?')

# 2 Create a function which prints your data as one string


def print_data(name=name, age=age):
    print(name + ', ' + age)


# 3 Create a function which prints ANY data (two arguments) as one string
arg_1 = input('Enter any argument 1:')
arg_2 = input('Enter any argument 2:')


def print_any_data(arg_1, arg_2):
    print(arg_1 + ' ' + arg_2)


print_any_data(arg_1, arg_2)

# 4 Create a function which calculates and returns the number of decades you already lived (e.g. 23 = 2 decades)


def number_of_decades(age=age):
    print(int(age)//10)


number_of_decades()

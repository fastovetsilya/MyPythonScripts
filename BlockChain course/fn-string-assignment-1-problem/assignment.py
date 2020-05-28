# 1) Write a normal function that accepts another function as an argument. Output the result of that other function in your “normal” function.
list_of_elements = [1, 4, 6, 10]
def square(el):
    return el**2

def update_list(list_of_elements):
    updated_list = []
    for el in list_of_elements:
        updated_list.append(square(el))
    print(updated_list)
    return updated_list

update_list(list_of_elements)

# 2) Call your “normal” function by passing a lambda function – which performs any operation of your choice – as an argument.
update_list = lambda list_of_elements: [square(el) for el in list_of_elements]
print(update_list(list_of_elements))

# 3) Tweak your normal function by allowing an infinite amount of arguments on which your lambda function will be executed.
update_list = lambda *args: [square(el) for el in args]
updated_list = update_list(2, 5)
print(updated_list)

# 4) Format the output of your “normal” function such that numbers look nice and are centered in a 20 character column.
update_list = lambda *args: [square(el) for el in args]
updated_list = update_list(2, 5)
for el in updated_list:
    print('{:^20}'.format(el))
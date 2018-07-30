# 1) Create a list of “person” dictionaries with a name, age and list of hobbies for each person. Fill in any data you want.
person = [{'name': 'Kate', 'age': 29, 'hobby': 'tennis'},
          {'name': 'Jack', 'age': 48, 'hobby': 'arts'},
          {'name': 'Joanna', 'age': 14, 'hobby': 'ski'},
          {'name': 'Obbie', 'age': 29, 'hobby': 'rap'}]

# 2) Use a list comprehension to convert this list of persons into a list of names (of the persons).
list_of_names = [p['name'] for p in person]
print(list_of_names)

# 3) Use a list comprehension to check whether all persons are older than 20.
print(all([el > 20 for el in [p['age'] for p in person]]))

# 4) Copy the person list such that you can safely edit the name of the first person (without changing the original list).
# copied_persons = persons[:]
import copy
copied_persons = copy.deepcopy(person)
copied_persons[0].update({'name': 'Kira'})
print(person)
print(copied_persons)

# 5) Unpack the persons of the original list into different variables and output these variables.
Kate, Jack, Joanna, Obbie = [p for p in person]
print(Kate)
print(Jack)
print(Joanna)
print(Obbie)

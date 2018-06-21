# 1) Create a list of names and use a for loop to output the length of each name (len()).
lst = ['Max', 'Ann', 'Kate', 'John']
for name in lst:
    print(len(name))

print('Ended part 1')
print('-' * 20)

# 2) Add an if check inside the loop to only output names longer than 5 characters.
lst = ['Max', 'Ann', 'Kate', 'John', 'Joanna', 'Rukkola']
for name in lst:
    if len(name) > 5:
        print(name)

print('Ended part 2')
print('-' * 20)

# 3) Add another if check to see whether a name includes a “n” or “N” character.
lst = ['Max', 'Ann', 'Kate', 'John', 'Joanna', 'Rukkola']
for name in lst:
    if len(name) > 5 and ('N' in name or 'n' in name):
        print(name)

print('Ended part 3')
print('-' * 20)

# 4) Use a while loop to empty the list of names (via pop())
while len(lst) > 0:
    print(lst.pop())
print(lst)

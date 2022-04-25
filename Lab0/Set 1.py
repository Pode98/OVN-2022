
# Exercise 1
num1 = int(input("Enter first number: "))
num2 = int(input("Enter second number: "))

product = num1 * num2
if product > 1000:
    print("Product over a 1000. The sum is: ", num1 + num2)
else:
    print("Product under and/or equal to a 1000. The product is: ", product)


# Exercise 2
num1 = int(input("Enter the number that will start the iteration: "))
num2 = int(input("Enter the number that will finish the iteration: "))
counter = 0

if num1 >= num2:
    print("Error. Numbers invalid")
else:
    while num1 < num2:
        counter += num1
        print("Sum: ", sum)
        counter = num1
        num1 += 1


# Exercise 3 and 4
num_list = [10, 20, 3, 190, 45]
print("Are the first and the last element the same number?")
first = num_list[0]
last = num_list[-1]
if first == last:
    print("They're the same.")
else:
    print("They're different.")
print("Now I will iterate and print only the ones divisible by 5")  # Ex4 starts now
for i in num_list:
    if i % 5 == 0:
        print("The number ", i, " is divisible by 5.")


# Exercise 5
phrase = "Emma is good developer. Emma is also a writer."
count = 0
i = 0
for i in range(len(phrase)-4):
    if phrase[i:i+4] == "Emma":
        count += 1
print(count)

# Exercise 6
list1 = [1, 2, 3, 4, 5]
list2 = [6, 7, 8, 9, 10]
list3 = []

for i in range(len(list1)):
    if list1[i] % 2 != 0:
        list3.append(list1[i])
    if list2[i] % 2 == 0:
        list3.append(list2[i])
print("The new list is ", list3)

# Exercise 7
s1 = "OpenNets"
s2 = "Optical"
middle_index = int(len(s1) / 2)
print("Original strings are", s1, s2)
middle_three = s1[:middle_index] + s2 + s1[middle_index:]
print("\nAfter appending the new string in the middle", middle_three)

# Exercise 8
s1 = "America"
s2 = "Japan"
s3 = s1[0] + s2[0] + s1[int(len(s1)/2)] + s2[int(len(s2)/2)] + s1[-1] + s2[-1]
print("Mixed string: ", s3)

# Exercise 9
input_string = "P@#yn26at^&i5ve"
count_char = 0
count_num = 0
count_symb = 0
for char in input_string:
    if char.islower() or char.isupper():
        count_char += 1
    elif char.isnumeric():
        count_num += 1
    else:
        count_symb += 1
print("Chars: ", count_char, " Numbers: ", count_num, " Symbols: ", count_symb)

# Exercise 10
substring = "usa"
string = "Welcome to USA. Awesome usa, isn\"t it"
count = string.lower().count(substring)
print("The string USA appaered ", count, " times.")

# Exercise 11
input_string = "English = 78 Science = 83 Math = 68 History = 65"
words = input_string.split()
tot = 0
cont = 0
for word in words:
    if word.isnumeric():
        tot += int(word)
        cont += 1

average = tot / cont
print("The average grade is ", average, " meanwhile the sum is ", tot)

# Exercise 12
input_string = "pynativepynvepynative"
chars_dict = dict()
for char in input_string:
    count = input_string.count(char)
    chars_dict[char] = count
print(chars_dict)



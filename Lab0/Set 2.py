
# Exercise 1  questo esercizio mi ha confuso molto, in quando chiede di unire in una lista gli elementi di indice pari
# di una lista e gli elementi di indice dispari dall'altra: ora, se mi avesse chiesto di usare le posizione all'interno della lista
# avrei considerato gli attuali indici come i+1, ovvero partendo da 1 piuttosto che da 0; visto che non era specificato
# ho considerato proprio l'indice in se per il calcolo
list1 = [3, 6, 9, 12, 15, 18, 21, 56]
list2 = [4, 8, 12, 16, 20, 24, 28, 55]

print("List1 with only odd indexs: ", list1[1::2])
print("List2 with only even indexs: ", list2[::2])
list3 = []
i = 0
while i < int(len(list1)):
    list3.append(list2[i])
    list3.append(list1[i+1])
    i += 2
print("Final list: ", list3)

# Exercise 2
sampleList = [34, 54, 67, 89, 11, 43, 94]
sample_list = [34, 54, 67, 89, 11, 43, 94]
print("Original list", sample_list)
element = sample_list.pop(3)
print("List after removing element at index 4", sample_list)
sample_list.insert(2-1, element)
print("List after adding element at index 2", sample_list)
sample_list.append(element)
print("List after adding element at last", sample_list)

# Exercise 3
sample_list = [11, 45, 8, 23, 14, 12, 78, 45, 89]
print("Original list", sample_list)
length = len(sample_list)
chunk_size = int(length / 3)
start = 0
end = chunk_size
for i in range(1, chunk_size + 1):
    indexes = slice(start, end, 1)
    list_chunk = sample_list(indexes)
    print(reversed(list_chunk))
    start = end
    if i < chunk_size:
        end += chunk_size
    else:
        end = length - chunk_size

# Exercise 4
sample_list = [11, 45, 8, 11, 23, 45, 23, 45, 89]
print("Original list", sample_list)
count_dict = dict()
for item in sample_list:
    if item in count_dict:
        count_dict[item] += 1
    else:
        count_dict[item] = 1
print("Printing count of each item", count_dict)


# Exercise 5
list1 = [2, 3, 4, 5, 6, 7, 8]
list2 = [4, 9, 16, 25, 36, 49, 64]
print("First list: ", list1)
print("Second list: ", list2)
list3 = zip(list1, list2)
result_set = set(list3)
print(result_set)

# Exercise 6
set1 = {23, 42, 65, 57, 78, 83, 29}
set2 = {57, 83, 29, 67, 73, 43, 48}
intersection = set1.intersection(set2)
for item in intersection:
    set1.remove(item)

# Exercise 7
set1 = {57, 83, 29}
set2 = {57, 83, 29, 67, 73, 43, 48}
if set1.issubset(set2):
    set2.clear()
elif set2.issubset(set1):
    set1.clear()
print("The first set is: ", set1)
print("The second set is: ", set2)

# Exercise 8

roll_number = [47, 64, 69, 37, 76, 83, 95, 97]
sample_dict = {"Jhon": 47, "Emma": 69, "Kelly": 76, "Jason": 97}


text = """        
I love machine learning.
I love deep learning.
Machine learning is interesting.         
"""

lowercase = (text.lower()).strip()
print("Original text : ", lowercase)

char_list = set(lowercase)
print("Character tokens : ", char_list)

sort_list = sorted(char_list)
print("Vocabulary : " , sort_list)

char_to_id = {}
for index,char in enumerate(sort_list):
    char_to_id[char] = index

print("Character to ID : " , char_to_id)
print("\n")

encode_id = []
for char in lowercase:
    encode_id.append(char_to_id[char])
print(encode_id)

id_to_char = {}
for index,char in enumerate(sort_list):
    id_to_char[index] = char

decoded = []
for ids in encode_id:
    decoded.append(id_to_char[ids])
print("".join(decoded))
text = """
I love machine learning .
I love deep learning .
Machine learning is interesting .
"""
print(text)

lowercase = text.lower()
print(lowercase)

tokens = lowercase.split()

tokens_uni = set(tokens)
print(tokens_uni)

tokens_id = {word:index for index,word in enumerate(tokens_uni)}
print(tokens_id)

encode_id = [tokens_id[word] for word in tokens]
print(encode_id)

id_to_word = {index:word for index,word in enumerate(tokens_uni)}
print(id_to_word)

decoded_words = [id_to_word[index] for index in encode_id]
decoded_text = " ".join(decoded_words)
print(decoded_text)
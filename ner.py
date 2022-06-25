import spacy
from spacy import displacy
NER = spacy.load("en_core_web_sm")

def NamedER(text):
    
    text1= NER(text)
    print("The entities are:")
    print(text)
    for word in text1.ents:
        print(word.text,word.label_)


    
import re

def decontracted(phrase):
  """ This function expands the contractions in the text"""
    # specific
  phrase = str(phrase)
  phrase = re.sub(r"won\'t", "will not", phrase)
  phrase = re.sub(r"can\'t", "can not", phrase)

    # general
  phrase = re.sub(r"n\'t", " not", phrase)
  phrase = re.sub(r"\'re", " are", phrase)
  phrase = re.sub(r"\'s", " is", phrase)
  phrase = re.sub(r"\'d", " would", phrase)
  phrase = re.sub(r"\'ll", " will", phrase)
  phrase = re.sub(r"\'t", " not", phrase)
  phrase = re.sub(r"\'ve", " have", phrase)
  phrase = re.sub(r"\'m", " am", phrase)
  phrase = re.sub('xxxx','',phrase) #occurs many times in text may be private information which isn't useful
  return str(phrase)

def other_processing(phrase):
  """Other text processing mentioned below"""

  phrase = str(phrase)
  phrase = re.sub(r'xx*','',phrase) # Removing XXXX
  phrase = re.sub(r'\d','',phrase) # Removing numbers
  
  temp = ""

  for i in phrase.split(" "): #Removing 2 letter words
    if i!= 'no' or i!='ct':
      temp = temp + ' ' + i
    prev = i
  temp = re.sub(' {2,}', ' ',temp) #Replacing double space with single space
  temp = re.sub(r'\.+', ".", temp) #Replacing double . with single .
  temp = temp.lstrip() #Removing space at the beginning
  temp = temp.rstrip() #Removing space at the end
  return temp
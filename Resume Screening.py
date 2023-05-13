import PyPDF2 as pdf
import os
import re
import json
import numpy as np
from nltk.stem import PorterStemmer
import time
import csv
import wordninja
import tkinter as ui
from tkinter import *
from PIL import ImageTk, Image

stemmer = PorterStemmer()

path = r"C:\Users\v895a817\Downloads\Resumes"
path2 = r"C:\Users\v895a817\Downloads"
docs_arr = [] #list of all documents' text
index = {} #index containing term frequencies
dims_old = [] #list of dimensions (unique tokens) before indexing new docs
dims = [] #list of dimensions (unique tokens) after indexing new docs
doc_ids = {} #dictionary containing doc names and ids; read from file later
doc_start_index = 0
doc_vectors = [] #document vectors; read from file later
indices_to_update = [] #list containing indices to update the existing document vectors
sim_values = [] #list to hold similarity values
query_vec = [] #query vector
sum_sqre = 0
query = ""

# function to read a pdf file
def readPdf(path):
  pdfobj = open(path,'rb')
  pdfread = pdf.PdfReader(pdfobj)
  data = ""
  for x in range(len(pdfread.pages)):
    pdfpage = pdfread.pages[x]
    data += pdfpage.extract_text()
  pdfobj.close()
  return data

#function to preprocess text
def preprocesstext(text):
  text = text.lower() #lowercase
  syms = '''!()-[]{};:'"\,<>.?/$@#%^&*_~+=''' #punctuations
  nums = '0123456789'   #numbers
  #remove punctuations and numbers
  for char in text:
    if char in syms:
      text = text.replace(char, "")
    elif char in nums:
      text = text.replace(char, "")
  text = re.sub('[^A-Za-z]+', ' ', text)
  return text

#function to tokenize text
def tokenize(text):
  tokens = []
  #remove extra spaces and newlines
  tokens = wordninja.split(text)
  return tokens

#function to remove stop words
def remove_stopwords(words):
  stopwords = ['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', 'arent', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'com', 'couldn', 'couldnt', 'd', 'de', 'did', 'didn', 'didnt', 'do', 'does', 'doesn', 'doesnt', 'doing', 'don', 'dont', 'down', 'during', 'each', 'en', 'etc', 'few', 'for', 'from', 'further', 'had', 'hadn', 'hadnt', 'has', 'hasn', 'hasnt', 'have', 'haven', 'havent', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'isn', 'isnt', 'it', 'its', 'its', 'itself', 'just', 'll', 'la', 'm', 'ma', 'me', 'mightn', 'mightnt', 'more', 'most', 'mustn', 'mustnt', 'my', 'myself', 'needn', 'neednt', 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', 'shant', 'she', 'shes', 'should', 'shouldve', 'shouldn', 'shouldnt', 'so', 'some', 'such', 't', 'than', 'that', 'thatll', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', 'wasnt', 'we', 'were', 'weren', 'werent', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', 'wont', 'wouldn', 'wouldnt', 'www', 'y', 'you', 'youd', 'youll', 'youre', 'youve', 'your', 'yours', 'yourself', 'yourselves']
  words_new = []
  for word in words:
    if word not in stopwords:
      words_new.append(word)
  return words_new

#function for porter stemmer
def ps(tokens):
  for i in range(len(tokens)):
    tokens[i] = stemmer.stem(tokens[i])
  return tokens


#function to read text from docs
def read_docs(files):
  path_2 = os.path.join(path2, "docs.txt")
  if os.path.isfile(path_2):
    with open(path_2,'r') as f:
      if os.path.getsize(path_2) > 0:
        doc_ids.update(json.loads(f.read()))
  global doc_start_index
  doc_start_index = len(doc_ids)
  for file in files:
    if file not in doc_ids:
      doc_ids[file] = len(doc_ids)
      path_new = os.path.join(path, file)
      docs_arr.append(readPdf(path_new))
  with open(path_2,'w') as file:
    file.write(json.dumps(doc_ids))


#function to read past index
def read_index():
  path_new = os.path.join(path2, "index.txt")
  if os.path.isfile(path_new):
    global index
    with open(path_new, 'r') as file:
      if os.path.getsize(path_new)>0:
        data = json.loads(file.read())
        index.update(data)
  else:
    index = {}
  global dims_old
  dims_old = list(index.keys())
  dims_old.sort()

#function to update index
def indexer(docs):
  doc_id = doc_start_index
  for doc in docs:
    text = preprocesstext(doc)  # preprocessing text
    token_list = tokenize(text)  # convert text to tokens
    token_list = remove_stopwords(token_list) #remove stopwords
    token_list = ps(token_list) #Stem words using porter stemmer
    for token in token_list:
      if token not in index:
        index[token] = {}
        index[token][str(doc_id)] = 1
      else:
        if str(doc_id) not in index[token]:
          index[token][str(doc_id)]=1
        else:
          index[token][str(doc_id)] += 1
    doc_id += 1
  dims.extend([*index]) #list of latest dimensions
  dims.sort()

  #write index to file
  path_new = os.path.join(path2, "index.txt")
  with open(path_new, 'w') as file:
    file.write(json.dumps(index))

#logic for calculating term frequency
def term_freq(val):
  if val > 0:
    return (1+np.log10(val))
  else:
    return 0

#logic for calculating inverted document frequency
def idf(term,mode):
  return np.log10(len(doc_ids)/len(index[term])) #for query


#read document vectors
def read_doc_vectors():
  global doc_vectors
  if len(dims) > len(dims_old):
    path_new = os.path.join(path2,"doc_vectors.csv")
  else:
    path_new = os.path.join(path2, "norm_doc_vecs.csv")
  if os.path.isfile(path_new):
    with open(path_new, newline='\n') as f:
      reader = csv.reader(f)
      doc_vectors = list(reader)


#create document vectors for new documents
def create_doc_vectors(count):
  if len(dims) > len(dims_old):
    for i in range(count):
      doc_vectors.append(['0' for j in range(len(dims))])
    for doc_id in range(doc_start_index,len(doc_ids)):
      for token_id in range(len(dims)):
        if str(doc_id) in index[dims[token_id]]:
          tf = term_freq(index[dims[token_id]][str(doc_id)]) #term frequency
          inv_doc_freq = idf(dims[token_id],0)   #inverted document frequency
          doc_vectors[doc_id][token_id] = str(tf*inv_doc_freq)
        else:
          doc_vectors[doc_id][token_id] = str(0)

#update document vectors for existing docs
def update_doc_vectors(count):
  if len(dims) > len(dims_old):
    for ele in dims:
      if ele not in dims_old:
        indices_to_update.append(dims.index(ele))
    for doc_id in range(count):
      for val in indices_to_update:
        tf = term_freq(index[dims[val]][doc_id])
        idfreq = idf(dims[val],0)
        doc_vectors[doc_id].insert(val,str(tf*idfreq))

#write document vectors to file
def write_doc_vectors():
  if(len(doc_vectors)-doc_start_index !=0):
    path_new = os.path.join(path2, "doc_vectors.csv")
    temp_vec = []
    for vec in doc_vectors:
      temp_vec.append(','.join(vec))
    with open(path_new, 'w') as f:
      f.write('\n'.join(temp_vec))


#function to process query and generate query vectors
def process_query(text):
  text = preprocesstext(text)  # preprocessing text
  tokens = tokenize(text)  # convert text to tokens
  tokens = remove_stopwords(tokens)  # remove stopwords
  tokens = ps(tokens)  # Stem words using porter stemmer
  query_tf = np.zeros(len(dims))
  global query_vec
  query_vec = np.zeros(len(dims))
  for token in tokens:
    if token in dims:
      query_tf[dims.index(token)] = 1
  global sum_sqre
  for i in range(len(dims)):
    tf = term_freq(query_tf[i])
    idfreq = idf(dims[i],1)
    query_vec[i] = tf * idfreq
    sum_sqre += query_vec[i]*query_vec[i]
  sum_sqre = np.sqrt(sum_sqre)
  if sum_sqre != 0:
    query_vec = [j/sum_sqre for j in query_vec]
#  for k in range(len(query_vec)):
#    if query_vec[k] !=0:
#      print(str(dims[k]) + " : " + str(query_vec[k]) + "\n")


#function to normalize vectors
def normalize():
  if len(dims) > len(dims_old):
    sum_sqr = np.zeros(len(doc_vectors))
    for docid in range(len(doc_vectors)):
      for val in doc_vectors[docid]:
        sum_sqr[docid] += float(val)*float(val)
      if sum_sqr[docid] != 0:
        sum_sqr[docid] = np.sqrt(sum_sqr[docid])
      else:
        sum_sqr[docid] = 1
    for docid in range(len(doc_vectors)):
      doc_vectors[docid] = [str(round(float(i)/sum_sqr[docid], 7)) for i in doc_vectors[docid]]

    #write normalized vectors to a file
    path_new = os.path.join(path2, "norm_doc_vecs.csv")
    temp_vec = []
    for vec in doc_vectors:
      temp_vec.append(','.join(vec))
    with open(path_new, 'w') as f:
      f.write('\n'.join(temp_vec))


#function to calculate cosine similarity
def cosine_sim():
  global sim_values
  sim = 0
  doc_li = list(doc_ids.keys())
  for j in range(len(doc_vectors)):
    for i in range(len(dims)):
      sim += float(doc_vectors[j][i])*query_vec[i]
    sim_values.append([sim, doc_li[j]])
    sim = 0


#get top documents
def get_top_docs(count):
  sorted_sim = sorted(sim_values, key=lambda x:x[0], reverse = True)
  if count <= len(sim_values):
    return (sorted_sim[0:count])
  else:
    return (sorted_sim)

def start():
  start_time = time.time()
  files = os.listdir(path) #Read file names
  print("reading docs")
  s = time.time()
  read_docs(files) #read all files text into a list
  e = time.time()
  print(e-s, "sec")
  print("reading index")
  s = time.time()
  read_index()
  e = time.time()
  print(e - s, "sec")
  print("indexing")
  s = time.time()
  indexer(docs_arr) #index the text from docs
  e = time.time()
  print(e - s, "sec")
  print("reading doc vectors")
  s = time.time()
  read_doc_vectors() #Read existing document vectors from file
  e = time.time()
  print(e - s, "sec")
  print("creating doc vectors")
  s = time.time()
  create_doc_vectors(len(doc_ids)-doc_start_index) # generate document vectors for new docs
  e = time.time()
  print(e - s, "sec")
  print("updating doc vs")
  s = time.time()
  update_doc_vectors(doc_start_index) #update the document vectors for already indexed docs
  e = time.time()
  print(e - s, "sec")
  print("writing doc vs")
  s = time.time()
  write_doc_vectors() #write document vectors to file
  e = time.time()
  print(e - s, "sec")
  print("processing query")
  s = time.time()
  process_query(query) #preprocess, tokenize and index query and generate query unit vector
  e = time.time()
  print(e - s, "sec")
  print("normalizing vectors")
  s = time.time()
  normalize() #normalize document vectors to unit vector
  e = time.time()
  print(e - s, "sec")
  print("cosine sim")
  s = time.time()
  cosine_sim() #calculate cosine similarity of query with all doc vectors
  e = time.time()
  print(e - s, "sec")
  top_docs = get_top_docs(10) #get top documents based on similarity
  print(len(dims))

  end_time = time.time()
  print("total time : ", end_time-start_time)

  return top_docs



#function for UI click button task
def on_submit():
  global query
  query = input_box.get()  # Get the input from the input box
  out = start()
  output_label.delete("1.0", "end")
  for vec in out:
    output_label.insert(ui.END, vec[1])
    output_label.insert(ui.END, "\t   Similarity :")
    output_label.insert(ui.END, round(vec[0],4))
    output_label.insert(ui.END, "\n")


if __name__ == '__main__':
  #UI
  root = ui.Tk()
  root.geometry("1800x1200")
  width = root.winfo_screenwidth()
  height = root.winfo_screenheight()
  img = Image.open(os.path.join(path2,"image.png"))
  img2 = img.resize((width, height))
  bg_image = ImageTk.PhotoImage(img2)
  root.title("Resume Ranker")
  canv = Canvas(root, width=1800, height=1200)
  canv.pack(fill="both", expand=True)
  canv.create_image(0, 0, image=bg_image, anchor='nw')

  # Create the input box and submit button
  input_box = ui.Entry(root, font=("arial", 24), width=22, borderwidth=3, relief="groove")
  canv.create_window(245, 100, window=input_box)
  submit_button = ui.Button(root, text="Find Resumes", command=on_submit, font=("arial", 16))
  canv.create_window(580, 100, window=submit_button)

  output_label = ui.Text(root, font=("arial", 20), height=10, width=40, borderwidth=3, relief="groove")
  canv.create_window(347, 350, window=output_label)

  root.mainloop()






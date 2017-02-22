# ------------- IMPORTS --------------------
import sys
reload(sys)
sys.setdefaultencoding('Cp1252')
import time
import datasplit as dsp
import random
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
# --------------- START TIME -----------------

start_time = time.time()

# ----------- IMPORTED DATA -------------------

hockey = dsp.hockey             #20861
movies = dsp.movies             #22409
nba = dsp.nba                   #18422
news = dsp.news                 #21057
nfl = dsp.nfl                   #20106
politics = dsp.politics         #19694
soccer = dsp.soccer             #21363
worldnews = dsp.worldnews       #21088


‘’’
Note: To cross validate, 
I simply changed the values below. 
I performed 3-fold cross validation. 
‘’’


# TESTING SETS
test_n = 6875
hockey_test = dsp.hockey[-test_n:]
movies_test = dsp.movies[-test_n:]
nba_test = dsp.nba[-test_n:]
news_test = dsp.news[-test_n:]
nfl_test = dsp.nfl[-test_n:]
politics_test = dsp.politics[-test_n:]
soccer_test = dsp.soccer[-test_n:]
worldnews_test = dsp.worldnews[-test_n:]

# TRAINING SETS
n = 13750
hockey = dsp.hockey[:n]           
movies = dsp.movies[:n]          
nba = dsp.nba[:n]               
news = dsp.news[:n]            
nfl = dsp.nfl[:n]             
politics = dsp.politics[:n]  
soccer = dsp.soccer[:n]     
worldnews = dsp.worldnews[:n] 
'''

test_set = dsp.test_set
test_n = len(test_set)


'''


# ------------ CUSTOM FILTERING -----------------

def clean_to_list(text):
        words = word_tokenize(text)
        filtered_text = [lemmatizer.lemmatize(w) for w in words if not lemmatizer.lemmatize(w) in stop_words]
        return [value for value in filtered_text
        if '<' not in value
        and '>' not in value
        and '\\' not in value
        and '/' not in value
        and 'speaker' not in value
        and '`' not in value
        and '\'' not in value
        and len(value)>2
        and not value.isdigit()
        and 'html' not in value
        and not ('com' in value and len(value)==3)
        and not ('org' in value and len(value)==3)
        and '&quot' not in value
        and '&amp' not in value
        and '&gt' not in value # these are HTML encodings
        and not (value == 'hes')
	and not (value == 'shes')
        ]

allowed_word_types = ["N"]

# ------------- BUILD DOCUMENT LIST ---------------

print "Building documents ..."
last_time = time.time()
'''
documents = []
all_words = []

for p in hockey:
        documents.append((p,"hockey"))
        #documents.append((p,"sports"))
        words = clean_to_list(p)
        pos = nltk.pos_tag(words)
        for w in pos:
                if w[1][0] in allowed_word_types:
                        all_words.append(w[0].lower())
for p in movies:
        documents.append((p,"movies"))
        #documents.append((p,"other"))
        words = clean_to_list(p)
        pos = nltk.pos_tag(words)
        for w in pos:
                if w[1][0] in allowed_word_types:
                        all_words.append(w[0].lower())
for p in nba:
        documents.append((p,"nba"))
        #documents.append((p,"sports"))
        words = clean_to_list(p)
        pos = nltk.pos_tag(words)
        for w in pos:
                if w[1][0] in allowed_word_types:
                        all_words.append(w[0].lower())
for p in news:
        documents.append((p,"news"))
        #documents.append((p,"other"))
        words = clean_to_list(p)
        pos = nltk.pos_tag(words)
        for w in pos:
                if w[1][0] in allowed_word_types:
                        all_words.append(w[0].lower())
for p in nfl:
        documents.append((p,"nfl"))
        #documents.append((p,"sports"))
        words = clean_to_list(p)
        pos = nltk.pos_tag(words)
        for w in pos:
                if w[1][0] in allowed_word_types:
                        all_words.append(w[0].lower())
for p in politics:
        documents.append((p,"politics"))
        #documents.append((p,"other"))
        words = clean_to_list(p)
        pos = nltk.pos_tag(words)
        for w in pos:
                if w[1][0] in allowed_word_types:
                        all_words.append(w[0].lower())
for p in soccer:
        documents.append((p,"soccer"))
        #documents.append((p,"sports"))
        words = clean_to_list(p)
        pos = nltk.pos_tag(words)
        for w in pos:
                if w[1][0] in allowed_word_types:
                        all_words.append(w[0].lower())
for p in worldnews:
        documents.append((p,"worldnews"))
        #documents.append((p,"other"))
        words = clean_to_list(p)
        pos = nltk.pos_tag(words)
        for w in pos:
                if w[1][0] in allowed_word_types:
                        all_words.append(w[0].lower())


## PICKLE DOCUMENTS -- SAVE
save_classifier = open("./pickled_items/documents1.pickle", "wb")
pickle.dump(documents, save_classifier)
save_classifier.close()
## PICKLE ALL_WORDS -- SAVE
save_classifier = open("./pickled_items/all_words1.pickle", "wb")
pickle.dump(all_words, save_classifier)
save_classifier.close()
'''
## PICKLE DOCUMENTS -- OPEN
documents_f = open("./pickled_items/documents1.pickle","rb")
documents = pickle.load(documents_f)
documents_f.close()

## PICKLE ALL_WORDS -- OPEN
documents_f = open("./pickled_items/all_words1.pickle","rb")
all_words = pickle.load(documents_f)
documents_f.close()


print "\t Time taken to build documents:", time.time()-last_time, "s"

# ------------- BUILD FEATURE LIST ---------------------

print "Building feature list ..."
last_time = time.time()

'''
word_distribution = nltk.FreqDist(all_words)
number_of_features = 10000
word_features = word_distribution.most_common(number_of_features)

temp = []
for w in word_features:
	temp.append(w[0])
word_features = temp[:]

## PICKLE WORD_FEATURES -- SAVE
save_word_features = open("./pickled_items/word_features1.pickle", "wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()
'''
## PICKLE WORD_FEATURES -- OPEN
word_features_f = open("./pickled_items/word_features1.pickle", "rb")
word_features = pickle.load(word_features_f)
word_features_f.close()
#List of  3000 most common words across all documents. 


print "\t Time taken to build feature list", time.time()-last_time, "s"

# =======================  MY NAIVE BAYES ========================

vocabulary = word_features

#	Word features is the top 3000 most common words across all documents.
#	We limit our vocabulary to this subset. 
#	When predicting the class of a novel document, words not in vocabulary will be discarded.

# ----------- PROBABILITY ( CLASS ) FOR ALL 8 CLASSES -----------------

P_hockey = len(hockey)/(len(documents)+0.0)
P_movies = len(movies)/(len(documents)+0.0)
P_nba = len(nba)/(len(documents)+0.0)
P_news = len(news)/(len(documents)+0.0)
P_nfl = len(nfl)/(len(documents)+0.0)
P_politics = len(politics)/(len(documents)+0.0)
P_soccer = len(soccer)/(len(documents)+0.0)
P_worldnews = len(worldnews)/(len(documents)+0.0)

# ------------ CONCATENATE ALL DOCUMENTS FOR EACH CLASS ----------------
'''
This function returns a list of all words contained across all documents of a given class, with duplicated.
We limit these words to those in our vocabulary.
'''

def concatenate_docs(doc_class):
	total = []
	for doc in doc_class:
		words = clean_to_list(doc)
		for word in words:
			if word in vocabulary:
				total.append(word)
	return total

# ------------------------- CLASS_WORDS ------------------------------
print "Concatenating documents... "
last_time = time.time()
'''
hockey_words = concatenate_docs(hockey)
movies_words = concatenate_docs(movies)
nba_words = concatenate_docs(nba)
news_words = concatenate_docs(news)
nfl_words = concatenate_docs(nfl)
politics_words = concatenate_docs(politics)
soccer_words = concatenate_docs(soccer)
worldnews_words = concatenate_docs(worldnews)

##	PICKLE -- SAVE	
save_concat = open("./pickled_items/concat_docs_hockey1.pickle", "wb")
pickle.dump(hockey_words, save_concat)
save_concat.close()
save_concat = open("./pickled_items/concat_docs_movies1.pickle", "wb")
pickle.dump(movies_words, save_concat)
save_concat.close()
save_concat = open("./pickled_items/concat_docs_nba1.pickle", "wb")
pickle.dump(nba_words, save_concat)
save_concat.close()
save_concat = open("./pickled_items/concat_docs_news1.pickle", "wb")
pickle.dump(news_words, save_concat)
save_concat.close()
save_concat = open("./pickled_items/concat_docs_nfl1.pickle", "wb")
pickle.dump(nfl_words, save_concat)
save_concat.close()
save_concat = open("./pickled_items/concat_docs_politics1.pickle", "wb")
pickle.dump(politics_words, save_concat)
save_concat.close()
save_concat = open("./pickled_items/concat_docs_soccer1.pickle", "wb")
pickle.dump(soccer_words, save_concat)
save_concat.close()
save_concat = open("./pickled_items/concat_docs_worldnews1.pickle", "wb")
pickle.dump(worldnews_words, save_concat)
save_concat.close()

'''
## 	PICKLE -- OPEN
save_concat_f = open("./pickled_items/concat_docs_hockey1.pickle", "rb")
hockey_words = pickle.load(save_concat_f)
save_concat_f.close()
save_concat_f = open("./pickled_items/concat_docs_movies1.pickle", "rb")
movies_words = pickle.load(save_concat_f)
save_concat_f.close()
save_concat_f = open("./pickled_items/concat_docs_nba1.pickle", "rb")
nba_words = pickle.load(save_concat_f)
save_concat_f.close()
save_concat_f = open("./pickled_items/concat_docs_news1.pickle", "rb")
news_words = pickle.load(save_concat_f)
save_concat_f.close()
save_concat_f = open("./pickled_items/concat_docs_nfl1.pickle", "rb")
nfl_words = pickle.load(save_concat_f)
save_concat_f.close()
save_concat_f = open("./pickled_items/concat_docs_politics1.pickle", "rb")
politics_words = pickle.load(save_concat_f)
save_concat_f.close()
save_concat_f = open("./pickled_items/concat_docs_soccer1.pickle", "rb")
soccer_words = pickle.load(save_concat_f)
save_concat_f.close()
save_concat_f = open("./pickled_items/concat_docs_worldnews1.pickle", "rb")
worldnews_words = pickle.load(save_concat_f)
save_concat_f.close()


doc_class_to_class_words = {'hockey':hockey_words, 'movies':movies_words, 'nba':nba_words, 'news':news_words, 
				'nfl':nfl_words, 'politics':politics_words, 'soccer':soccer_words, 'worldnews':worldnews_words}

print "\t Time taken to concatenate documents:", time.time()-last_time, "s"
# ----------------- CALC P( Wk | Class ) FOR EACH CLASS ----------------
'''
word_distributions = {'hockey':nltk.FreqDist(doc_class_to_class_words['hockey']), 
			'movies':nltk.FreqDist(doc_class_to_class_words['movies']),
			'nba':nltk.FreqDist(doc_class_to_class_words['nba']),
			'news':nltk.FreqDist(doc_class_to_class_words['news']),
			'nfl':nltk.FreqDist(doc_class_to_class_words['nfl']),
			'politics':nltk.FreqDist(doc_class_to_class_words['soccer']),
			'soccer':nltk.FreqDist(doc_class_to_class_words['soccer']),
			'worldnews':nltk.FreqDist(doc_class_to_class_words['worldnews'])
			}
'''

def conditional_prob(word, doc_class):
	class_words = doc_class_to_class_words[doc_class]
	word_distribution = nltk.FreqDist(class_words)
	#word_distribution = word_distributions[doc_class]
	n_k = word_distribution[word]
	numerator = n_k + 1.0 # in case nk is 0
	denominator = len(class_words) + len(word_features)
	return numerator/denominator


def generate_probability_list(doc_class):
	probability_list = []
	counter = 0
	for word in vocabulary:
		#print counter
		probability_list.append(conditional_prob(word, doc_class))
		counter +=1
	return probability_list


## PICKLE P_LISTS -- SAVE

print "Generating probability lists ..."
last_time = time.time()
'''
problist_hockey =  generate_probability_list('hockey')
problist_movies = generate_probability_list('movies')
problist_nba = generate_probability_list('nba')
problist_news = generate_probability_list('news')
problist_nfl = generate_probability_list('nfl')
problist_politics = generate_probability_list('politics')
problist_soccer = generate_probability_list('soccer')
problist_worldnews = generate_probability_list('worldnews')

save_plist_hockey = open("./pickled_items/plist_hockey1.pickle", "wb")
pickle.dump(problist_hockey, save_plist_hockey)
save_plist_hockey.close()
save_plist_movies = open("./pickled_items/plist_movies1.pickle", "wb")
pickle.dump(problist_movies, save_plist_movies)
save_plist_movies.close()
save_plist_nba = open("./pickled_items/plist_nba1.pickle", "wb")
pickle.dump(problist_nba, save_plist_nba)
save_plist_nba.close()
save_plist_news = open("./pickled_items/plist_news1.pickle", "wb")
pickle.dump(problist_news, save_plist_news)
save_plist_news.close()
save_plist_nfl = open("./pickled_items/plist_nfl1.pickle", "wb")
pickle.dump(problist_nfl, save_plist_nfl)
save_plist_nfl.close()
save_plist_politics = open("./pickled_items/plist_politics1.pickle", "wb")
pickle.dump(problist_politics, save_plist_politics)
save_plist_politics.close()
save_plist_soccer = open("./pickled_items/plist_soccer1.pickle", "wb")
pickle.dump(problist_soccer, save_plist_soccer)
save_plist_soccer.close()
save_plist_worldnews = open("./pickled_items/plist_worldnews1.pickle", "wb")
pickle.dump(problist_worldnews, save_plist_worldnews)
save_plist_worldnews.close()
'''

## PICKLE P_LISTS -- OPEN
plist_f = open("./pickled_items/plist_hockey1.pickle", "rb")
problist_hockey = pickle.load(plist_f)
plist_f.close()

plist_f = open("./pickled_items/plist_movies1.pickle", "rb")
problist_movies = pickle.load(plist_f)
plist_f.close()

plist_f = open("./pickled_items/plist_nba1.pickle", "rb")
problist_nba = pickle.load(plist_f)
plist_f.close()

plist_f = open("./pickled_items/plist_news1.pickle", "rb")
problist_news = pickle.load(plist_f)
plist_f.close()

plist_f = open("./pickled_items/plist_nfl1.pickle", "rb")
problist_nfl = pickle.load(plist_f)
plist_f.close()

plist_f = open("./pickled_items/plist_politics1.pickle", "rb")
problist_politics = pickle.load(plist_f)
plist_f.close()

plist_f = open("./pickled_items/plist_soccer1.pickle", "rb")
problist_soccer = pickle.load(plist_f)
plist_f.close()

plist_f = open("./pickled_items/plist_worldnews1.pickle", "rb")
problist_worldnews = pickle.load(plist_f)
plist_f.close()


print "\t Time taken to generate all probability lists:", time.time()-last_time, "s"

# ---------- CONVERT P_LISTS TO DICTIONARIES ---------------

p_given_hockey = {}
p_given_movies = {}
p_given_nba = {}
p_given_news = {}
p_given_nfl = {}
p_given_politics = {}
p_given_soccer = {}
p_given_worldnews = {}

dicos = [p_given_hockey, p_given_movies, p_given_nba, p_given_news, p_given_nfl, p_given_politics, p_given_soccer, p_given_worldnews]
prob_lists = [problist_hockey, problist_movies, problist_nba, problist_news, problist_nfl, problist_politics, problist_soccer, problist_worldnews]

for i in range(0, len(dicos)):
	for j in range(0, len(vocabulary)):
		dicos[i][vocabulary[j]] = prob_lists[i][j]

print "\t Time taken to build convert probability lists to Dictionaries:", time.time()-last_time
last_time = time.time()

# ----------------- CLASSIFICATION ------------------------

#	This function takes a raw_text as input and outputs a predicted class.

def nb_classify(raw_text):
	words = clean_to_list(raw_text)
        pos = nltk.pos_tag(words)
        word_list = []
	for w in pos:
                if w[1][0] in allowed_word_types:
			word_list.append(w[0].lower())
	to_predict = [w for w in word_list if w in vocabulary]
	
	probability_hockey = P_hockey
	probability_movies = P_movies
	probability_nba = P_nba
	probability_news = P_news
	probability_nfl = P_nfl
	probability_politics = P_politics
	probability_soccer = P_soccer
	probability_worldnews = P_worldnews
	
	for w in to_predict:
		probability_hockey *= p_given_hockey[w]
		probability_movies *= p_given_movies[w]
		probability_nba *= p_given_nba[w]
		probability_news *= p_given_news[w]
		probability_nfl *= p_given_nfl[w]
		probability_politics *= p_given_politics[w]
		probability_soccer *= p_given_soccer[w]
		probability_worldnews *= p_given_worldnews[w]

	p = [probability_hockey, probability_movies, probability_nba, probability_news, probability_nfl, probability_politics, probability_soccer, probability_worldnews]
	categories = ['hockey', 'movies', 'nba', 'news', 'nfl', 'politics', 'soccer', 'worldnews']
	
	max = 0
	maxIndex = 0
	for i in range(len(p)):
		if p[i] > max:
			max = p[i]
			maxIndex = i
	
	return categories[maxIndex]

# --------------------- CLASSIFICATION STEP ---------------------------

print "Classifying test sets... \n"
last_time = time.time()


total_false_predictions = 0


ddd = {}
mistakes = []
false_predictions = 0
for t in hockey_test:
	predicted_category = nb_classify(t)
	#print predicted_category
	if predicted_category != 'hockey':
		false_predictions += 1
		total_false_predictions +=1
		mistakes.append(predicted_category)
		if predicted_category not in ddd:
			ddd[predicted_category]=1
		else:
			ddd[predicted_category] +=1

print 'hockey', false_predictions, ((1-((false_predictions+0.0)/test_n))*100)
print mistakes, '\n'
print ddd, '*****************'

ddd={}
mistakes = []
false_predictions = 0
for t in movies_test:
        predicted_category = nb_classify(t)
        #print predicted_category
        if predicted_category != 'movies':
                false_predictions += 1
		total_false_predictions +=1
		mistakes.append(predicted_category)
		if predicted_category not in ddd:
			ddd[predicted_category]=1
		else:
			ddd[predicted_category] +=1
print 'movies', false_predictions, (1-((false_predictions+0.0)/test_n))*100
print mistakes, '\n'
print ddd, '*****************'

ddd={}
mistakes = []
false_predictions = 0
for t in nba_test:
        predicted_category = nb_classify(t)
        #print predicted_category
        if predicted_category != 'nba':
                false_predictions += 1
		total_false_predictions +=1
		mistakes.append(predicted_category)
		if predicted_category not in ddd:
			ddd[predicted_category]=1
		else:
			ddd[predicted_category] +=1
print 'nba', false_predictions, (1-((false_predictions+0.0)/test_n))*100
print mistakes, '\n'
print ddd, '*****************'


ddd={}
mistakes = []
false_predictions = 0
for t in news_test:
        predicted_category = nb_classify(t)
        #print predicted_category
        if predicted_category != 'news':
                false_predictions += 1
		total_false_predictions +=1
		mistakes.append(predicted_category)
		if predicted_category not in ddd:
			ddd[predicted_category]=1
		else:
			ddd[predicted_category] +=1
print 'news', false_predictions, (1-((false_predictions+0.0)/test_n))*100
print mistakes, '\n'
print ddd, '*****************'



ddd={}
mistakes = []
false_predictions = 0
for t in nfl_test:
        predicted_category = nb_classify(t)
        #print predicted_category
        if predicted_category != 'nfl':
                false_predictions += 1
		total_false_predictions +=1
		mistakes.append(predicted_category)
		if predicted_category not in ddd:
			ddd[predicted_category]=1
		else:
			ddd[predicted_category] +=1
print 'nfl', false_predictions, (1-((false_predictions+0.0)/test_n))*100
print mistakes, '\n'
print ddd, '*****************'


ddd={}
mistakes = []
false_predictions = 0
for t in politics_test:
        predicted_category = nb_classify(t)
        #print predicted_category
        if predicted_category != 'politics':
                false_predictions += 1
		total_false_predictions +=1
		mistakes.append(predicted_category)
		if predicted_category not in ddd:
			ddd[predicted_category]=1
		else:
			ddd[predicted_category] +=1
print 'politics', false_predictions, (1-((false_predictions+0.0)/test_n))*100
print mistakes, '\n'
print ddd, '*****************'

ddd={}
mistakes = []
false_predictions = 0
for t in soccer_test:
        predicted_category = nb_classify(t)
        #print predicted_category
        if predicted_category != 'soccer':
                false_predictions += 1
		total_false_predictions +=1
		mistakes.append(predicted_category)
		if predicted_category not in ddd:
			ddd[predicted_category]=1
		else:
			ddd[predicted_category] +=1
print 'soccer', false_predictions, (1-((false_predictions+0.0)/test_n))*100
print mistakes, '\n'
print ddd, '*****************'

ddd={}
mistakes = []
false_predictions = 0
for t in worldnews_test:
        predicted_category = nb_classify(t)
        #print predicted_category
        if predicted_category != 'worldnews':
                false_predictions += 1
		total_false_predictions +=1
		mistakes.append(predicted_category)
		if predicted_category not in ddd:
			ddd[predicted_category]=1
		else:
			ddd[predicted_category] +=1
print 'worldnews', false_predictions, (1-((false_predictions+0.0)/test_n))*100
print mistakes, '\n'
print ddd, '*****************'

print 'TOTAL:', total_false_predictions, (1-((total_false_predictions+0.0)/(8*test_n)))*100

'''
for t in test_set:
	print "%s,%s"%(t[0], nb_classify(t[1]))

'''
print "\nTotal time elapsed:", time.time()-start_time, "s"





import re

#s = "Example String, <hello> world <this''''''> is a cool <String?Strang>"
#replaced = re.sub('<.*>', '',s)
#print replaced
#print s


f = open("../data/train_output.csv","r").read()
lines = f.split('\n')
del lines[0]
del lines[-1]
# Line has format: "4,politics"
ID_topic = {}
for line in lines:
	temp = line.split(',')
	ID_topic[temp[0]] = temp[1].replace('\r','')


f = open("../data/train_input.csv","r").read().replace('"\r\n', '')
lines = f.split('\n')
del lines[0]
del lines[-1]
# Line has format: "ID, text"

hockey = []
movies = []
nba = []
news = []
nfl = []
politics = []
soccer = []
worldnews = []


for line in lines:
	
	temp = line.split(',')
	ID = temp[0]
	text = ''.join(temp[1:])
	
	if ID_topic[ID] == 'hockey':
		hockey.append(text)
	elif ID_topic[ID] == 'movies':
		movies.append(text)
        elif ID_topic[ID] == 'nba':
		nba.append(text)
	elif ID_topic[ID] == 'news':
		news.append(text)
        elif ID_topic[ID] == 'nfl':
		nfl.append(text)
	elif ID_topic[ID] == 'politics':
		politics.append(text)
	elif ID_topic[ID] == 'soccer':
		soccer.append(text)
	elif ID_topic[ID] == 'worldnews':
		worldnews.append(text)
	#else:
		#print "THERE'S AN ISSUE MOTHAFUCKA"
		#break

#for h in news:
#	print h
#	print

#print "hockey = ", len(hockey)
#print "movies = ", len(movies) 
#print "nba = ", len(nba)
#print "news = ", len(news)
#print "nfl = ", len(nfl)
#print "politics = ", len(politics) 
#print "soccer = ", len(soccer)
#print "worldnews = ", len(worldnews) 


'''
hockey =  20861
movies =  22409
nba =  18422
news =  21057
nfl =  20106
politics =  19694
soccer =  21363
worldnews =  21088
'''



f = open("../data/test_input.csv","r").read().replace('"\r\n', '')
lines = f.split('\n')
del lines[0]
del lines[-1]
# Line has format: "ID, text"


test_set = []
for line in lines:
	temp = line.split(',')
	ID = temp[0]
	text = ''.join(temp[1:])

	test_set.append((ID, text))


#for t in test_set:
#	print t
#	print

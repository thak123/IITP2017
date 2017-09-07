#! /usr/bin/env python
# -*- coding: utf-8 -*-



# What this code does:
# Given a Twitter stream in JSON-to-text format, the time window size in minutes (e.g., 15 minutes)
# and the output file name, extract top 10 topics detected in the time window

# Example run:
# python twitter-topics-from-json-text-stream.py SortedTopicWiseTweets/UPelection.2.0.csv 10080 10080-mins-topics-UPE-stream.txt > details_clusters_10080_mins_topics_UPE-stream.txt
import random
import codecs
import gensim
from collections import Counter
import CMUTweetTagger
from datetime import datetime
import fastcluster
from itertools import cycle
import json
#import nltk
import numpy as np
import re
import os
import scipy.cluster.hierarchy as sch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import metrics
import string
import sys
import time
from gensim import corpora, models
from collections import defaultdict
from gensim.models import Word2Vec

#<editor-fold> Private Helper Functions
#lemmatizer=nltk.stem.WordNetLemmatizer()




def load_stopwords():
#	stop_words = nltk.corpus.stopwords.words('english')
	stop_words=[u'i', u'me', u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'you', u'your', u'yours', u'yourself', u'yourselves', u'he', u'him', u'his', u'himself', u'she', u'her', u'hers', u'herself', u'it', u'its', u'itself', u'they', u'them', u'their', u'theirs', u'themselves', u'what', u'which', u'who', u'whom', u'this', u'that', u'these', u'those', u'am', u'is', u'are', u'was', u'were', u'be', u'been', u'being', u'have', u'has', u'had', u'having', u'do', u'does', u'did', u'doing', u'a', u'an', u'the', u'and', u'but', u'if', u'or', u'because', u'as', u'until', u'while', u'of', u'at', u'by', u'for', u'with', u'about', u'against', u'between', u'into', u'through', u'during', u'before', u'after', u'above', u'below', u'to', u'from', u'up', u'down', u'in', u'out', u'on', u'off', u'over', u'under', u'again', u'further', u'then', u'once', u'here', u'there', u'when', u'where', u'why', u'how', u'all', u'any', u'both', u'each', u'few', u'more', u'most', u'other', u'some', u'such', u'no', u'nor', u'not', u'only', u'own', u'same', u'so', u'than', u'too', u'very', u's', u't', u'can', u'will', u'just', u'don', u'should', u'now', u'd', u'll', u'm', u'o', u're', u've', u'y', u'ain', u'aren', u'couldn', u'didn', u'doesn', u'hadn', u'hasn', u'haven', u'isn', u'ma', u'mightn', u'mustn', u'needn', u'shan', u'shouldn', u'wasn', u'weren', u'won', u'wouldn']


	stop_words.extend(['this','that','the','might','have','been','from',
                'but','they','will','has','having','had','how','went'
                'were','why','and','still','his','her','was','its','per','cent',
                'a','able','about','across','after','all','almost','also','am','among',
                'an','and','any','are','as','at','be','because','been','but','by','can',
                'cannot','could','dear','did','do','does','either','else','ever','every',
                'for','from','get','got','had','has','have','he','her','hers','him','his',
                'how','however','i','if','in','into','is','it','its','just','least','let',
                'like','likely','may','me','might','most','must','my','neither','nor',
                'not','of','off','often','on','only','or','other','our','own','rather','said',
                'say','says','she','should','since','so','some','than','that','the','their',
                'them','then','there','these','they','this','tis','to','too','twas','us',
                'wants','was','we','were','what','when','where','which','while','who',
                'whom','why','will','with','would','yet','you','your','ve','re','rt', 'retweet', '#fuckem', '#fuck',
                'fuck', 'ya', 'yall', 'yay', 'youre', 'youve', 'ass','factbox', 'com', '&lt', 'th','&gt',
                'retweeting', 'dick', 'fuckin', 'shit', 'via', 'fucking', 'shocker', 'wtf', 'hey', 'ooh', 'rt&amp', '&amp',
                '#retweet', 'retweet', 'goooooooooo', 'hellooo', 'gooo', 'fucks', 'fucka', 'bitch', 'wey', 'sooo', 'helloooooo', 'lol', 'smfh'])
	#turn list into set for faster search
	stop_words = set(stop_words)
	return stop_words

def normalize_text(text):
	try:
		text = text.encode('utf-8')
	except: pass
	text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(pic\.twitter\.com/[^\s]+))','', text)
	text = re.sub('@[^\s]+','', text)
	text = re.sub('#([^\s]+)', '', text)
	text = re.sub('[:;>?<=*+()/,\-#!$%\{˜|\}\[^_\\@\]1234567890’‘]',' ', text)
	text = re.sub('[\d]','', text)
	text = text.replace(".", '')
	text = text.replace("'", ' ')
	text = text.replace("\"", ' ')
	#text = text.replace("-", " ")
	#normalize some utf8 encoding
	text = text.replace("\x9d",' ').replace("\x8c",' ')
	text = text.replace("\xa0",' ')
	text = text.replace("\x9d\x92", ' ').replace("\x9a\xaa\xf0\x9f\x94\xb5", ' ').replace("\xf0\x9f\x91\x8d\x87\xba\xf0\x9f\x87\xb8", ' ').replace("\x9f",' ').replace("\x91\x8d",' ')
	text = text.replace("\xf0\x9f\x87\xba\xf0\x9f\x87\xb8",' ').replace("\xf0",' ').replace('\xf0x9f','').replace("\x9f\x91\x8d",' ').replace("\x87\xba\x87\xb8",' ')
	text = text.replace("\xe2\x80\x94",' ').replace("\x9d\xa4",' ').replace("\x96\x91",' ').replace("\xe1\x91\xac\xc9\x8c\xce\x90\xc8\xbb\xef\xbb\x89\xd4\xbc\xef\xbb\x89\xc5\xa0\xc5\xa0\xc2\xb8",' ')
	text = text.replace("\xe2\x80\x99s", " ").replace("\xe2\x80\x98", ' ').replace("\xe2\x80\x99", ' ').replace("\xe2\x80\x9c", " ").replace("\xe2\x80\x9d", " ")
	text = text.replace("\xe2\x82\xac", " ").replace("\xc2\xa3", " ").replace("\xc2\xa0", " ").replace("\xc2\xab", " ").replace("\xf0\x9f\x94\xb4", " ").replace("\xf0\x9f\x87\xba\xf0\x9f\x87\xb8\xf0\x9f", "")

	return text

def nltk_tokenize(text):
	tokens = []
	pos_tokens = []
	entities = []
	features = []

	#if len(text.strip()) > 0:
	try:
			#tokens = nltk.word_tokenize(text)
			tokens = text.split()

			for word in tokens:
				if word.lower() not in stop_words and len(word) > 1:
					features.append(word)
	except: pass
	return [tokens, pos_tokens, entities, features]

'''Assumes its ok to remove user mentions and hashtags from tweet text (normalize_text), '''
'''since we extracted them already from the json object'''
def process_json_tweet(text, fout, debug):
	features = []

	if len(text.strip()) == 0:
		return []
	text = normalize_text(text)
	#print text
	#nltk pre-processing: tokenize and pos-tag, try to extract entities
	try:
		[tokens, pos_tokens, entities, features] = nltk_tokenize(text)
	except:
		print "nltk tokenize+pos pb!"
	if debug:
		try:
			fout.write("\n--------------------clean text--------------------\n")
			fout.write(text.decode('utf-8'))
			#fout.write(text)
			fout.write("\n--------------------tokens--------------------\n")
			fout.write(str(tokens))
	#		fout.write("\n--------------------cleaned tokens--------------------\n")
	#		fout.write(str(clean_tokens))
			fout.write("\n--------------------pos tokens--------------------\n")
			fout.write(str(pos_tokens))
			fout.write("\n--------------------entities--------------------\n")
			for ent in entities:
				fout.write("\n" + str(ent).decode('utf-8'))
			fout.write("\n--------------------features--------------------\n")
			fout.write(str(features))
			fout.write("\n\n")
		except:
			#print "couldn't print text"
			pass
	return features

'''Prepare features, where doc has terms separated by comma'''
def custom_tokenize_text(text):
	REGEX = re.compile(r",\s*")
	tokens = []
	for tok in REGEX.split(text):
		#if "@" not in tok and "#" not in tok:
		if "@" not in tok:
			#tokens.append(stem(tok.strip().lower()))
			tokens.append(tok.strip().lower())
	return tokens

def spam_tweet(text):
	if 'Jordan Bahrain Morocco Syria Qatar Oman Iraq Egypt United States' in text:
		return True

	if 'Some of you on my facebook are asking if it\'s me' in text:
		return True

	if '@kylieminogue please Kylie Follow Me, please' in text:
		return True

	if 'follow me please' in text:
		return True

	if 'please follow me' in text:
		return True

	return False

def flatten_to_strings(listOfLists):
    """Flatten a list of (lists of (lists of strings)) for any level
    of nesting"""
    result = []

    for i in listOfLists:
        # Only append if i is a basestring (superclass of string)
        if isinstance(i, basestring):
            result.append(i)
        # Otherwise call this function recursively
        else:
            result.extend(flatten_to_strings(i))
    return result


def word2vectorizer(V,vectorizer):
	#load the model
	corpus_vector_collection=[]
	# model = Word2Vec.load("word2vec_twitter_model.bin")

	model=gensim.models.KeyedVectors.load_word2vec_format("word2vec_twitter_model.bin", binary=True,unicode_errors='ignore')
	# print len(corpus)

	map_index_after_cleaning = {}
	Vclean = np.zeros((1, V.shape[1]))
	for i in range(0, V.shape[0]):
		#keep sample with size at least 5
		if V[i].sum() > 4:
			Vclean = np.vstack([Vclean, V[i].toarray()])
			map_index_after_cleaning[Vclean.shape[0] - 2] = i

	Vclean = Vclean[1:,]
	#play with scaling of X
	V = Vclean
	# print model.vector_size
 	print "Xclean.shape:", Vclean.shape

	counter=0

	for doc in V:
		# print "doc:",doc
		tokens= vectorizer.inverse_transform(doc)
		#hardcoded
		pattern_vector = np.zeros(400)
		n_words = 0
		# tokens=sentence.split(",")
		# print tokens
		counter+=1
		# print counter
		# if len(tokens) > 1:
		joe=[]
		print ([joe.append((tok.split())) for tok in np.unique(tokens)])
		for t in set(flatten_to_strings(joe)):
		# [np.unique(bi_term) for bi_term in np.nditer(tokens)] :
			# print t
			try:
				if  t in model.vocab:
					vector = model[t.strip()]
					pattern_vector = np.add(pattern_vector,vector)
				else:
					pattern_vector = np.add(pattern_vector,np.zeros((400,)))
					# pattern_vector = np.add(pattern_vector,np.random.uniform(-0.25,0.25,400))
				n_words += 1
			except KeyError, e:
				continue


		pattern_vector = np.divide(pattern_vector,n_words)
		corpus_vector_collection.append(pattern_vector)
		# elif len(tokens) == 1:
		# 	try:
		# 		pattern_vector = model[tokens[0].strip()]
		# 		corpus_vector_collection.append(pattern_vector)
		#
		# 	except KeyError:
		# 		pass
	return np.array(corpus_vector_collection),map_index_after_cleaning


#</editor-fold>

'''start main'''
if __name__ == "__main__":
	# Read the Time sorted events
	file_timeordered_tweets = codecs.open(sys.argv[1], 'r', 'utf-8')
	# Set the window time frame value
	time_window_mins = float(sys.argv[2])
	# Open file for writing the output
	fout = codecs.open(sys.argv[3], 'w', 'utf-8')

	debug=0
	# Load the stop words
	stop_words = load_stopwords()


	#read tweets in time order and window them
	# Variable for checking the time start
	tweet_unixtime_old = -1

	tid_to_raw_tweet = {}

	# Contains tweets which are filtered by number of user mentions and hashtags
	window_corpus = []
	# Contains media_urls for the corressponding tweet_id
	tid_to_urls_window_corpus = {}
	# Contains mapping for tweets in window_corpus to its tweet id
	tids_window_corpus = []

	dfVocTimeWindows = {}
	# variable for keeping track of 0-3 iterations of the bin. At 4 th cycle previous values are averaged and added to the current value
	t = 0
	# variable for keeping the total number of tweets read in a window irrespective of filters
	ntweets = 0

	corpus_lda=defaultdict(list)
	# num topics
	parameter_list=[1, 2, 3, 4]
	grid = defaultdict(list)

#	fout.write("\n--------------------start time window tweets--------------------\n")
	#efficient line-by-line read of big files
	for line in file_timeordered_tweets:
		[tweet_unixtime, tweet_gmttime, tweet_id, text, hashtags, users, urls, media_urls, nfollowers, nfriends] = eval(line)
		# Check if spam tweet
		if spam_tweet(text):
			continue

		if tweet_unixtime_old == -1:
			tweet_unixtime_old = tweet_unixtime

#  		#while this condition holds we are within the given size time window
		if (tweet_unixtime - tweet_unixtime_old) < time_window_mins * 60:
			ntweets += 1

			features = process_json_tweet(text, fout, debug)
			# create tweet representation with user mentions hashtags and tweet-text
			tweet_bag = ""
			try:
				for user in set(users):
					tweet_bag += "@" + user.decode('utf-8').lower() + ","
				for tag in set(hashtags):
					if tag.decode('utf-8').lower() not in stop_words:
						tweet_bag += "#" + tag.decode('utf-8').lower() + ","
				for feature in features:
					tweet_bag += feature.decode('utf-8') + ","
			except:
				#print "tweet_bag error!", tweet_bag, len(tweet_bag.split(","))
				pass

			#print tweet_bag.decode('utf-8')
			if len(users) < 3 and len(hashtags) < 3 and len(features) > 3 and len(tweet_bag.split(",")) > 4 and not str(features).upper() == str(features):
				tweet_bag = tweet_bag[:-1]

				window_corpus.append(tweet_bag)
				tids_window_corpus.append(tweet_id)
				tid_to_urls_window_corpus[tweet_id] = media_urls
				tid_to_raw_tweet[tweet_id] = text
				#print urls_window_corpus
		else:
				dtime = datetime.fromtimestamp(tweet_unixtime_old).strftime("%d-%m-%Y %H:%M")
				print "\nWindow Starts GMT Time:", dtime, "\n"
				print "start time processing",str(datetime.now()),"\n"
				tweet_unixtime_old = tweet_unixtime

				#increase window counter
				t += 1

 				#first only cluster tweets
				print("Corpus Length",len(window_corpus))
				print("Min DF",len(window_corpus)*0.0025)
				vectorizer = CountVectorizer(tokenizer=custom_tokenize_text, binary=True, min_df=max(int(len(window_corpus)*0.0025),10), ngram_range=(2,3))
 				V = vectorizer.fit_transform(window_corpus)
				#function to convert text to vector
				X,map_index_after_cleaning=word2vectorizer(V,vectorizer)
 			# 	map_index_after_cleaning = {}
				#hardcoded
				print "X",X
#  				Xclean = np.zeros((1, 400))
#  				for i in range(0, X.shape[0]):
#  					#keep sample with size at least 5
#  					if X[i].sum() > 4 or True:
#  						Xclean = np.vstack([Xclean, X[i]])
#  						map_index_after_cleaning[Xclean.shape[0] - 2] = i
# #   					else:
#   						print "OOV tweet:"

 			# 	Xclean = Xclean[1:,]
				#print "len(articles_corpus):", len(articles_corpus)
				print "total tweets in window:", ntweets
				#print "len(window_corpus):", len(window_corpus)
				print "X.shape:", X.shape
 			# 	print "Xclean.shape:", Xclean.shape
 				#print map_index_after_cleaning
				#play with scaling of X
				# X = Xclean
				Xdense = np.matrix(X).astype('float')
				X_scaled = preprocessing.scale(Xdense)
				X_normalized = preprocessing.normalize(X_scaled, norm='l2')
				#transpose X to get features on the rows
				#Xt = X_scaled.T
# 				#print "Xt.shape:", Xt.shape
				'''
 				vocX = vectorizer.get_feature_names()
 				#print "Vocabulary (tweets):", vocX
 				#sys.exit()

 				boost_entity = {}
 				pos_tokens = CMUTweetTagger.runtagger_parse([term.upper() for term in vocX])
 				#print "detect entities", pos_tokens
 				for l in pos_tokens:
 					term =''
 					for gr in range(0, len(l)):
 						term += l[gr][0].lower() + " "
  					if "^" in str(l):
 						boost_entity[term.strip()] = 2.5
 					else:
 				 		boost_entity[term.strip()] = 1.0

				dfX = X.sum(axis=0)
 				#print "dfX:", dfX
 				dfVoc = {}
 				wdfVoc = {}
 				boosted_wdfVoc = {}
 				keys = vocX
 				vals = dfX
 				for k,v in zip(keys, vals):
 					dfVoc[k] = v
 				for k in dfVoc:
 					try:
 						dfVocTimeWindows[k] += dfVoc[k]
 						avgdfVoc = (dfVocTimeWindows[k] - dfVoc[k])/(t - 1)
					except:
 						dfVocTimeWindows[k] = dfVoc[k]
 						avgdfVoc = 0

 					wdfVoc[k] = (dfVoc[k] + 1) / (np.log(avgdfVoc + 1) + 1)
					try:
						boosted_wdfVoc[k] = wdfVoc[k] * boost_entity[k]
					except:
						boosted_wdfVoc[k] = wdfVoc[k]

				print "sorted wdfVoc*boost_entity:"
				print sorted( ((v,k) for k,v in boosted_wdfVoc.iteritems()), reverse=True)
				'''
				#Hclust: fast hierarchical clustering with fastcluster
				#X is samples by features
				#distMatrix is sample by samples distances
				distMatrix = pairwise_distances(X_normalized, metric='cosine')

				#cluster tweets
 				print "fastcluster, average, cosine"
 				L = fastcluster.linkage(distMatrix, method='average')

				#for dt in [0.3, 0.4, 0.5, 0.6, 0.7]:
				#for dt in [0.5]:
				dt = 0.5
				print "hclust cut threshold:", dt
#				indL = sch.fcluster(L, dt, 'distance')
				indL = sch.fcluster(L, dt*distMatrix.max(), 'distance')
				#print "indL:", indL
				freqTwCl = Counter(indL)
				print "n_clusters:", len(freqTwCl)
				print(freqTwCl)
#				print "silhoutte: ", metrics.silhouette_score(distMatrix, indL, metric="precomputed")
				allowSiloutte=False
				for freqTwClkey,freqTwClCount in freqTwCl.iteritems():
					if(freqTwClCount>1):
						allowSiloutte = True
						break

				if allowSiloutte:
					print "silhoutte: ", metrics.silhouette_score(distMatrix, indL, metric="precomputed")

				npindL = np.array(indL)

#				print "top50 most populated clusters, down to size", max(10, int(X.shape[0]*0.0025))
				freq_th = max(10, int(X.shape[0]*0.0025))
				cluster_score = {}
	 			for clfreq in freqTwCl.most_common(50):
	 				cl = clfreq[0]
 					freq = clfreq[1]
 					cluster_score[cl] = 0
 					if freq >= freq_th:
 	 					#print "\n(cluster, freq):", clfreq
	 					clidx = (npindL == cl).nonzero()[0].tolist()
						cluster_centroid = X[clidx].sum(axis=0)
						#print "centroid_array:", cluster_centroid
						try:
							# cluster_tweet = vectorizer.inverse_transform(cluster_centroid)
							#get the words closest to center
							sim_word_list=model.most_similar(positive=[cluster_centroid], topn=20)

							pos_tokens = CMUTweetTagger.runtagger_parse([term.upper() for term[0] in sim_word_list])
			 				#print "detect entities", pos_tokens
							score=0
			 				for l in pos_tokens:
			 					term =''

			 					for gr in range(0, len(l)):
			 						term += l[gr][0].lower() + " "
			  					if "^" in str(l):
			 						score += 2.5
			 					else:
			 				 		score += 1.0


							cluster_score[cl] =score

						except: pass
						cluster_score[cl] /= freq
					else: break

				sorted_clusters = sorted( ((v,k) for k,v in cluster_score.iteritems()), reverse=True)
				print "sorted cluster_score:"
		 		print sorted_clusters

		 		ntopics = 20
		 		headline_corpus = []
		 		orig_headline_corpus = []
		 		headline_to_cluster = {}
		 		headline_to_tid = {}
		 		cluster_to_tids = {}

				for score,cl in sorted_clusters[:ntopics]:
#		 		for score,cl in sorted_clusters:
					#print "\n(cluster, freq):", cl, freqTwCl[cl]
					clidx = (npindL == cl).nonzero()[0].tolist()
					first_idx = map_index_after_cleaning[clidx[0]]
					keywords = window_corpus[first_idx]
					fout.write("New Cluster \n\n")
					# Code for LDA
					for member in clidx:
						member_headline=[]
						member_keywords = window_corpus[map_index_after_cleaning[member]]
						fout.write(tid_to_raw_tweet[tids_window_corpus[map_index_after_cleaning[member]]].replace("\n", ' ').replace("\t", ' ')+"\n")
						for k in member_keywords.split(","):
							if not '@' in k and not '#' in k:
#								# headline += k + ","
								member_headline.append(k.lower())
#							member_headline.append(k.lower())
						corpus_lda[cl].extend([member_headline])
					fout.write("Cluster Ends\n")
					dictionary = corpora.Dictionary(corpus_lda[cl])
					corpus = [dictionary.doc2bow(text) for text in corpus_lda[cl]]
#					cp = random.sample(corpus,len(corpus_lda[cl]))
#					 # split into 80% training and 20% test sets
#					p = int(len(cp) * .5)
#					cp_train = cp[0:p]
#					cp_test = cp[p:]
					  # for num_topics_value in num_topics_list:
#					for parameter_value in parameter_list:
#					print "starting pass for parameter_value = %.3f" % parameter_value
					start_time = time.time()

#						ldamodel = gensim.models.ldamodel.LdaModel(cp_train,id2word = dictionary,num_topics=parameter_value)
					ldamodel = gensim.models.hdpmodel.HdpModel(corpus,  id2word = dictionary)

					# show elapsed time for model
					elapsed = time.time() - start_time
					print "Elapsed time: %s" % elapsed

#					perplex = ldamodel.bound(cp_test)
#					print "Perplexity: %s" % perplex
#					grid[parameter_value].append(perplex)

#						per_word_perplex = np.exp2(-perplex / sum(cnt for document in cp_test for _, cnt in document))
#						print "Per-word Perplexity: %s" % per_word_perplex
#						grid[parameter_value].append(per_word_perplex)
					print("\niteration start")
					print(ldamodel.print_topics(num_topics=20, num_words=10))
					print("iteration over")

					# Code Ends
					orig_headline_corpus.append(keywords)
					headline = ''
					for k in keywords.split(","):
						if not '@' in k and not '#' in k:
							headline += k + ","
					headline_corpus.append(headline[:-1])
					headline_to_cluster[headline[:-1]] = cl
					headline_to_tid[headline[:-1]] = tids_window_corpus[first_idx]

					tids = []
					for i in clidx:
						idx = map_index_after_cleaning[i]
						tids.append(tids_window_corpus[idx])
					#   						try:
					#   							print window_corpus[map_index_after_cleaning[i]]
					#   						except: pass
					cluster_to_tids[cl] = tids

				#sys.exit()
				window_corpus = []
				tids_window_corpus = []
				tid_to_urls_window_corpus = {}
				tid_to_raw_tweet = {}
				ntweets = 0
				if t == 4:
					dfVocTimeWindows = {}
					t = 0

				#fout.write("\n--------------------start time window tweets--------------------\n")
				#fout.write(line)
	print "end time processing",str(datetime.now()),"\n"
	file_timeordered_tweets.close()
	fout.close()

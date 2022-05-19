#####################
### For BEA Paper ###
#####################
import glob
import pickle
import spacy
from spacy.tokens import Doc
from spacy.language import Language
### First, we have to deal with tokenization - we want Spacy to treat our data as pre-tokenized ###
# Split on whitespace
class WhitespaceTokenizer(object):
	def __init__(self, vocab):
		self.vocab = vocab

	def __call__(self, text):
		words = text.split(' ')
		# All tokens 'own' a subsequent space character in this tokenizer
		spaces = [True] * len(words)
		return Doc(self.vocab, words=words, spaces=spaces)

#Return one sentence per doc (Force spacy to use our sentence tokenization)
#note that this may cause problems after it has been defined once. If it throws an error, skip this component
@Language.component("single_sent") 
def custom_sent(doc): 
	for token in doc:
		if token.i == 0:
			doc[token.i].is_sent_start = True
		else:
			doc[token.i].is_sent_start = False
	return(doc)

def spacy_tag(pos_sents,model): #model = "en_core_web_trf" or "en_core_web_sm"
	print("Start Spacy")
	nsentst = len(pos_sents)
	nsents = 0
	smsent = 0
	tagged = []
	#load spacy model, set tokenization
	print("OK1")
	nlp = spacy.load(model)
	print("OK2")
	nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
	print("OK3")
	#nlp.add_pipe("single_sent", before='parser') #ensure sentence tokenization is the same as input text
	nlp.add_pipe("single_sent", after='tagger') #ensure sentence tokenization is the same as input text
	print("OK4")

	for sent in pos_sents:
		nsents += 1
		smsent += 1
		if smsent == 100:
			print("Processing",nsents,"of",nsentst,"sentences with Spacy model",model)
			smsent = 0
		taggeds = []
		doc = nlp(" ".join(tag_strip(sent))) #process data
		for token in doc:
			taggeds.append((token.text,token.tag_)) #add to sentence level list
		tagged.append(taggeds) #add sentence to corpus level list
	return(tagged)

def dep_strip(dep_sent):
	outl = []
	for token in dep_sent:
		#print(token)
		outl.append(token["token"])
	return(outl)

def spacy_parse_cpos(dep_sents,model): #model = "en_core_web_trf" or "en_core_web_sm"

	nsentst = len(dep_sents)
	nsents = 0
	smsent = 0
	tagged = []
	#load spacy model, set tokenization
	nlp_tagger = spacy.load(l1l2comb_trf) #load best tagger!!!
	nlp = spacy.load(model)
	nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
	nlp.add_pipe("single_sent", before='parser') #ensure sentence tokenization is the same as input text
	print(nlp.pipe_names)
	nlp.remove_pipe("tagger")
	nlp.add_pipe("tagger",source=nlp_tagger,before = 'single_sent')
	print(nlp.pipe_names)
	for sent in dep_sents:
		nsents += 1
		smsent += 1
		if smsent == 100:
			print("Processing",nsents,"of",nsentst,"sentences with Spacy model",model)
			smsent = 0
		taggeds = []
		doc = nlp(" ".join(dep_strip(sent))) #process data
		for token in doc:
			idx = str(token.i + 1)
			if token.i == token.head.i and token.dep_ == "ROOT":
				hidx = str(0)
			else:
				hidx = str(token.head.i + 1)
			if token.dep_ == "ROOT":
				token.dep_ = "root"
			lblhd = "_".join([token.dep_,hidx])
			taggeds.append({"idx":idx,"token":token.text,"xpos": token.tag_,"label":token.dep_,"hidx" : hidx,"label+hidx":lblhd}) #add to sentence level list
		tagged.append(taggeds) #add sentence to corpus level list
	return(tagged)

def prs(tp,fn,fp): #precision, recall, f1  simple
	if tp == 0:
		return({"recall" : 0,"precision" : 0,"f1" : 0})
	else:
		recall = tp/(tp + fn)
		precision =  tp/(tp + fp)
		f1 =  2 * ((precision*recall)/(precision + recall))
		return({"recall" : recall,"precision" : precision,"f1" : f1})

def total_compile(posd,ttgl): #posd
	tc,tp,fn,fp = 0,0,0,0 #initial states
	for x in posd:
		if x not in ["All_Tags","Lexical_Tags"]:
			if ttgl == False or x in ttgl:
				#print(x)
				tc += posd[x]["TC"]
				tp += posd[x]["TP"]
				fn += posd[x]["FN"]
				fp += posd[x]["FP"]
	total = prs(tp,fn,fp)
	total["TC"] = tc
	total["TP"] = tp
	total["FN"] = fn
	total["FP"] = fp
	return(total)

def dep_compile(gold_corp,tgr_list): #single gold corpus
	outd = {}
	for tgr in tgr_list:
		nmstr = tgr[2] #name for dictionary
		tagger = tgr[0] #tagger function
		model = tgr[1] #model name
		if model == False:
			tagged = tagger(gold_corp)
		else:
			tagged = tagger(gold_corp,model)
		outd[nmstr] = dep_prec_rec2(tagged,gold_corp)
	return(outd)

#this is for a summary of output across taggers and corpora:
def dep_compile_full(gold_corpora,tgr_list): #[(gold_corpus,"gold_corpus_nm"),]
	outd = {}
	for gold_corp, gc_name in gold_corpora: #iterate through corpora
		outd[gc_name] = dep_compile(gold_corp,tgr_list)
	return(outd)
#deppr

def total_compile_dep(posd,ttgl,acctype): #posd
	tc,pwsum,rwsum,f1wsum = 0,0,0,0 #initial states
	for x in posd:
		if x not in ["All_Tags","Lexical_Tags"]:
			if ttgl == False or x in ttgl:
				#print(x)
				tc += posd[x]["TC"]
				pwsum += posd[x][acctype]["precision"] * posd[x]["TC"] #get weighted sum of precision
				rwsum += posd[x][acctype]["recall"] * posd[x]["TC"] #get weighted sum of recall
				f1wsum += posd[x][acctype]["f1"] * posd[x]["TC"] #get weighted sum of f1
	total = {acctype : {}} #make dictionary levels match input dictionary
	total["TC"] = tc
	total[acctype]["precision"] = pwsum/tc #get weighted average
	total[acctype]["recall"] = rwsum/tc #get weighted average
	total[acctype]["f1"] = f1wsum/tc #get weighted average
	return(total)

def dep_table_maker_summary(posfd,tagsl,acctype = "LAS",tinfo = 'f1',rnd = 3,sep = "\t",outn = False):
	#tagsl = [x for x in tagslin] #copy list
	outl = [] #This will be organized by tag
	tcl = sorted(list(posfd.keys())) #target corpus list
	ttl = sorted(list(posfd[tcl[0]].keys())) #target tagger list (based on what is available for the first corpus in TCL)
	outl.append(["Tag","Corpus","nTokens"]+ttl) #include header
	for posd in posfd:
		for d in posfd[posd]:
			posfd[posd][d]["Lexical_Tags"] = total_compile_dep(posfd[posd][d],tagsl,acctype) #calculate lexical tags
			posfd[posd][d]["All_Tags"] = total_compile_dep(posfd[posd][d],False,acctype) #calculate lexical tags
	
	tagsl = ["All_Tags","Lexical_Tags"] + tagsl #add lexical tags to beginning of list
	#print(tagsl)
	for tag in tagsl: #Iterate through tags
		for idx,crp in enumerate(tcl): #iterate through corpora
			if idx == 0:
				if tag in posfd[crp][ttl[0]]:
					line = [tag,crp,posfd[crp][ttl[0]][tag]["TC"]] #make first column the tag, also add number of items
				else:
					line = [tag,crp,0] #make first column the tag, also add number of items
			else:
				if tag in posfd[crp][ttl[0]]:
					line = ["",crp,posfd[crp][ttl[0]][tag]["TC"]] #make empty first column, also add number of items
				else:
					line = [tag,crp,0] #make first column the tag, also add number of items
			for tgr in ttl: #for tagger in target tagger list
				print(tgr,crp,tag)
				if tag in posfd[crp][tgr]:
					print("OK1")
					if rnd != False:
						print("OK2")
						line.append(round(posfd[crp][tgr][tag][acctype][tinfo],rnd)) #round output
						print("OK3")
					else:
						print("OK4")
						line.append(posfd[crp][tgr][tag][acctype][tinfo]) #round output
				else:
					line.append(0) #if tag isn't in corpus
			outl.append(line)
	return(outl)

def pos_reader_conllu(conllus,onto = False): #list of conllu files
	outl = []
	for fname in conllus:
		sents = open(fname).read().split("\n\n") #split into sentences
		for sent in sents:
			skip = False
			if len(sent) < 1:
				continue
			pos_sent = []
			for toks in sent.split("\n"):
				if toks[0] == "#":
					continue
				else:
					items = toks.split("\t")
					idx = items[0]
					if "-" in idx or "." in idx:
						continue
					if items[4] == "XX":
						skip = True
						continue
					if onto == False:
						if items[4] == "_": print(fname.split("/")[-1])
						pos_sent.append((items[1],items[4]))
					else:
						if items[3] == "_": print(fname.split("/")[-1])
						pos_sent.append((items[1],items[3]))
			if skip == False:
				outl.append(pos_sent)
	print(len(outl),"sentences in corpus")
	return(outl)

def tag_strip(sent):
	toks = []
	for word,tag in sent:
		toks.append(word)
	return(toks)

def pos_compile(gold_corp,tgr_list): #single gold corpus
	outd = {}
	for tgr in tgr_list:
		nmstr = tgr[2] #name for dictionary
		tagger = tgr[0] #tagger function
		model = tgr[1] #model name
		print(nmstr)
		if model == False:
			print("no model name")
			tagged = tagger(gold_corp)
		else:
			print(nmstr)
			tagged = tagger(gold_corp,model)
		outd[nmstr] = tag_prec_rec(tagged,gold_corp)
	return(outd)

#this is for a summary of output across taggers and corpora:
def pos_compile_full(gold_corpora,tgr_list): #[(gold_corpus,"gold_corpus_nm"),]
	outd = {}
	for gold_corp, gc_name in gold_corpora: #iterate through corpora
		outd[gc_name] = pos_compile(gold_corp,tgr_list)
	return(outd)

def flat_lister(lols):
	fl = []
	for sent in lols: 
		if isinstance(sent[0],str): #if list is already flat (e.g., if TreeTagger is used):
			fl.append({'word':sent[0],'pos':sent[1]})
		else:
			for tok in sent:
				fl.append({'word':tok[0],'pos':tok[1]})
	return(fl)

def flat_lister_dep(lols):
	fl = []
	for sent in lols:
		for tok in sent:
			fl.append(tok)
	return(fl)

def pprint(dictentry):
	for x in dictentry:
		print(x)

def freq_add(d,item):
	if item not in d:
		d[item] = 1
	else:
		d[item] += 1

def pred_accuracy(pred,gold):
	c = 0
	f = 0

	for idx, x in enumerate(pred):
		#print(x,gold[idx]["pos"])
		if x[1] == gold[idx][1]:
			c+=1
		else:
			f+=1
	return(c/(c+f))

#code for precision, recall, and F1 formulas
def prec_rec(accuracy_dict):
	accuracy_dict["TC"] = accuracy_dict["TP"] + accuracy_dict["FN"]
	if accuracy_dict["TP"] + accuracy_dict["FN"] == 0:
		accuracy_dict["recall"] = 0
	else:
		accuracy_dict["recall"] = accuracy_dict["TP"]/(accuracy_dict["TP"] + accuracy_dict["FN"])

	if accuracy_dict["TP"] +accuracy_dict["FP"] == 0:
		accuracy_dict["precision"] = 0
	else:
		accuracy_dict["precision"] = accuracy_dict["TP"]/(accuracy_dict["TP"] +accuracy_dict["FP"])
	if accuracy_dict["precision"] == 0 and accuracy_dict["recall"] == 0:
		accuracy_dict["f1"] = 0
	else:
		accuracy_dict["f1"] = 2 * ((accuracy_dict["precision"] * accuracy_dict["recall"])/(accuracy_dict["precision"] + accuracy_dict["recall"]))

#code to apply the above function to a full dataset
def tag_prec_rec(tested,gold): #takes list of sentences as input:
	tag_d = {}

	if isinstance(tested[0],(list,tuple)) == True:
		tested = flat_lister(tested)
		gold = flat_lister(gold)

	for idx, item in enumerate(gold):
		### convert formats, as needed: ###
		if type(item) == str:
			item = {"pos" : item}

		# print(tested[idx])
		if type(tested[idx]) == str:
			tested[idx] = {"pos" : tested[idx]}

		### update tag dictionary as needed ###
		if item["pos"] not in tag_d:
			tag_d[item["pos"]] = {"TP":0,"FP":0,"FN":0}
		if tested[idx]["pos"] not in tag_d:
			tag_d[tested[idx]["pos"]] = {"TP":0,"FP":0,"FN":0}

		### tabulate accuracy ###
		if item["pos"] == tested[idx]["pos"]:
			tag_d[item["pos"]]["TP"] += 1
		else:
			tag_d[item["pos"]]["FN"] += 1
			tag_d[tested[idx]["pos"]]["FP"] += 1

	for x in tag_d:
		prec_rec(tag_d[x])

	return(tag_d)

def deppr (tcr,gtc,ttc): # total correct, gold total count, tested total count; calculation of precision and recall for dependency information
	if ttc == 0:
		p = 0
	else:
		p = tcr/ttc
	if gtc == 0:
		r = 0
	else:
		r = tcr/gtc
	if p+r == 0:
		f1 = 0
	else:
		f1 = (2*(p*r))/(p+r)
	return({"precision" : p,"recall":r,"f1":f1})

def dep_prec_rec2(tested,gold,vars = ("hidx","label","label+hidx")): #takes sentences as input, also takes names for (head,label,label+head)
	ulas = {} #for unlabeled attachment
	las = {} #labeled attachment score
	ls = {} #label score (ignoring head)
	gold_countd = {} #for keeping track of number of hits per tag in gold data
	tested_countd = {} #for keeping track of number of hits per tag in tested data

	#convert data as needed:
	if isinstance(tested,dict) == True:
		tested = dtol(tested) #convert from dictionary to list
		gold = dtol(gold) #convert from dictionary to list
	
	#print(tested)
	if isinstance(tested[0],list) == True:
		tested = flat_lister_dep(tested) #get rid of sentence breaks
		gold = flat_lister_dep(gold) #get rid of sentence breaks

	for goldl,testedl in zip(gold,tested): #iterate through tokens
		#print(goldl["token"],goldl["label"],testedl["token"],testedl["label"])
		### update tag dictionaries as needed ###
		if goldl["label"] not in ulas:
			ulas[goldl["label"]] = {"TP":0,"FP":0} #correct hit, incorrect hit
			las[goldl["label"]] = {"TP":0,"FP":0} #correct hit, incorrect hit
			ls[goldl["label"]] = {"TP":0,"FP":0} #correct hit, incorrect hit
			gold_countd[goldl["label"]] = 0 #Total counts (gold)
			tested_countd[goldl["label"]] = 0 #Total counts (tested)
		if testedl["label"] not in ulas:
			ulas[testedl["label"]] = {"TP":0,"FP":0} #correct hit, incorrect hit
		if testedl["label"] not in las:
			las[testedl["label"]] = {"TP":0,"FP":0} #correct hit, incorrect hit
		if testedl["label"] not in ls:
			ls[testedl["label"]] = {"TP":0,"FP":0} #correct hit, incorrect hit
		if testedl["label"] not in gold_countd:
			gold_countd[testedl["label"]] = 0 #Total counts (gold)
		if testedl["label"] not in tested_countd:
			tested_countd[testedl["label"]] = 0 #Total counts (tested)

		### tabulate accuracy ###
		#total counts
		gold_countd[goldl["label"]] += 1 #add one to total count for label in gold
		tested_countd[testedl["label"]] += 1 #add one to total count for label in tested

		#unlabeled
		#print(goldl["hidx"],goldl["label+hidx"],goldl["label"],testedl["hidx"],testedl["label+hidx"],testedl["label"])
		if goldl["hidx"] == testedl["hidx"]:
			ulas[testedl["label"]]["TP"] += 1
		else:
			ulas[testedl["label"]]["FP"] += 1
		#head + id
		if goldl["label+hidx"] == testedl["label+hidx"]:
			las[testedl["label"]]["TP"] += 1
		else:
			las[testedl["label"]]["FP"] += 1
		#label only
		if goldl["label"] == testedl["label"]:
			ls[testedl["label"]]["TP"] += 1
		else:
			ls[testedl["label"]]["FP"] += 1
	print("OK")
	outd = {} #output dictionary
	for x in gold_countd: #iterate through tags
		#print(x)
		outd[x] = {}
		outd[x]["TC"] = gold_countd[x] #number of actual hits in gold corpus
		outd[x]["ULAS"] = deppr(ulas[x]["TP"],gold_countd[x],tested_countd[x])
		outd[x]["LAS"] = deppr(las[x]["TP"],gold_countd[x],tested_countd[x])
		outd[x]["LS"] = deppr(ls[x]["TP"],gold_countd[x],tested_countd[x])
	return(outd)

def conllu_extract_dep(conllu,d = False,check = False): #takes list of list of strings as input
	outl = [] #holder for outlist
	#print(conllu)
	for tok in conllu.split("\n"):
		if tok[0] == "#":
			continue
		items = tok.split("\t")
		idx = items[0]
		if "-" in idx or "." in idx: #skip split idx, orphan idx
			#print(tok)
			continue
		word = items[1]
		lbl = items[7] #dep label
		xpos = items[4] #postag
		if check == True and lbl == "_":
			print(conllu.split("\n")[0])
		head = items[6] #dep head idx
		lblhd = "_".join([items[7],items[6]])
		if d == False: #output tuple
			outl.append((word,lbl,head,lblhd))
		else: #output dict
			outl.append({"token":word,"xpos":xpos,"label":lbl, "hidx":head, "label+hidx":lblhd})
	return(outl)

def conllu_extract_dep_full(conllulol,d = True,check = False):
	outl = []
	for fnm in conllulol:
		print("Processing",fnm.split("/")[-1])
		for sent in open(fnm).read().strip().split("\n\n"):
			outl.append(conllu_extract_dep(sent, d,check))
	return(outl)

def pos_table_maker(posd,tagsl,header = ["TC","TP","FP","FN","precision","recall","f1"],rnd = 3,sep = "\t",outn = False):
	#tagsl = [x for x in tagslin] #copy list
	outl = []
	tl = sorted(list(posd.keys()))
	for d in posd:
		posd[d]["Lexical_Tags"] = total_compile(posd[d],tagsl) #calculate lexical tags
		posd[d]["All_Tags"] = total_compile(posd[d],False) #calculate lexical tags
	
	tagsl = ["All_Tags","Lexical_Tags"] + tagsl #add lexical tags to beginning of list
	print(tagsl)
	for tag in tagsl:
		for idx,tgr in enumerate(tl):
			if idx == 0:
				line = [tag,tgr] #make first column the tag
			else:
				line = ["",tgr] #make empty first column
			for item in header:
				if tag in posd[tgr]:
					if rnd != False:
						line.append(round(posd[tgr][tag][item],rnd)) #round output
				else:
					line.append(0) #if tag isn't in corpus
			outl.append(line)
	if outn != False: #write to file if filename is provided
		outf = open(outn,"w")
		outf.write("Tag" + sep + "Tagger" + sep + sep.join([str(x) for x in header])) #write header
		for rows in outl:
			outf.write("\n"+ sep.join([str(x) for x in rows]))
		outf.flush()
		outf.close()
	return(outl)

#this is for a summary of output across taggers and corpora:
def pos_table_maker_summary(posfd,tagsl,tinfo = 'f1',rnd = 3,sep = "\t",outn = False):
	#tagsl = [x for x in tagslin] #copy list
	outl = [] #This will be organized by tag
	tcl = sorted(list(posfd.keys())) #target corpus list
	ttl = sorted(list(posfd[tcl[0]].keys())) #target tagger list (based on what is available for the first corpus in TCL)
	outl.append(["Tag","Corpus","nTokens"]+ttl) #include header
	for posd in posfd:
		for d in posfd[posd]:
			posfd[posd][d]["Lexical_Tags"] = total_compile(posfd[posd][d],tagsl) #calculate lexical tags
			posfd[posd][d]["All_Tags"] = total_compile(posfd[posd][d],False) #calculate lexical tags
	
	tagsl = ["All_Tags","Lexical_Tags"] + tagsl #add lexical tags to beginning of list
	#print(tagsl)
	for tag in tagsl: #Iterate through tags
		for idx,crp in enumerate(tcl): #iterate through corpora
			if idx == 0:
				if tag in posfd[crp][ttl[0]]:
					line = [tag,crp,posfd[crp][ttl[0]][tag]["TC"]] #make first column the tag, also add number of items
				else:
					line = [tag,crp,0] #make first column the tag, also add number of items
			else:
				if tag in posfd[crp][ttl[0]]:
					line = ["",crp,posfd[crp][ttl[0]][tag]["TC"]] #make empty first column, also add number of items
				else:
					line = [tag,crp,0] #make first column the tag, also add number of items
			for tgr in ttl: #for tagger in target tagger list
				print(tgr,crp,tag)
				if tag in posfd[crp][tgr]:
					print("OK1")
					if rnd != False:
						print("OK2")
						line.append(round(posfd[crp][tgr][tag][tinfo],rnd)) #round output
						print("OK3")
					else:
						print("OK4")
						line.append(posfd[crp][tgr][tag][tinfo]) #round output
				else:
					line.append(0) #if tag isn't in corpus
			outl.append(line)
	return(outl)

#write report generated by pos_table_maker_summary()
def wpos_report(reportl, outn, header = True,sep = "\t"):
	outf = open(outn,"w")
	if header != True: #if first line isn't header
		outf.write(sep.join([str(x) for x in header])) #write header
		for rows in reportl:
			outf.write("\n"+ sep.join([str(x) for x in rows]))
	else:
		outf.write(sep.join([str(x) for x in reportl[0]])) #write header
		for rows in reportl[1:]:
			outf.write("\n"+ sep.join([str(x) for x in rows]))
	outf.flush()
	outf.close()

def basic_stats(locorp,extract_funct = conllu_extract_dep_full): #get nsents,ntokens
	outd = {}
	for x in locorp:
		simp_name = x.split("/")[-1]
		reg_name = simp_name.split("-")[0]
		if reg_name not in outd:
			outd[reg_name] = {}
		data = extract_funct([x]) #conllu_extract_dep_full
		outd[reg_name][simp_name] = {"nsents":len(data),"ntokens":len(flat_lister_dep(data))}
	for reg in outd:
		nsents = 0
		ntokens = 0
		for div in outd[reg]:
			nsents += outd[reg][div]["nsents"]
			ntokens += outd[reg][div]["ntokens"]
		outd[reg][reg] = {"nsents":nsents,"ntokens":ntokens}
	return(outd)

def wbasic_stats(statsd,outname):
	outf = open(outname,"w")
	outl = []
	for reg in statsd:
		print(reg)
		for div in statsd[reg]:
			print(div)
			outl.append("\t".join([div,str(statsd[reg][div]["nsents"]),str(statsd[reg][div]["ntokens"])]))
	outf.write("\n".join(outl))
	outf.flush()
	outf.close()

##########################################
### Variables that need to be defined: ###
##########################################
goldpath = '/conllu_path/' #UPDATE THIS: path to folder with gold annotated .conllu files
spacy_path = '/spacy_models_path/' #UPDATE THIS: path to folder with spacy models
report_path = '/path_for_report_folder/' #UPDATE THIS: Path to where reports should be written

####################
### POS Analyses ###
####################

#target tags for accuracy analysis
target_tags_fs = ['NN', 'IN', 'DT', 'PRP', 'JJ', 'RB', 'NNP', 'VB', 'NNS', 'CC', 'VBP', 'VBD', 'VBZ', 'TO', 'CD', 'VBN', 'VBG', 'PRP$', 'MD', 'WRB', 'WDT', 'UH', 'WP', 'RP', 'JJR', 'EX', 'NNPS', 'JJS', 'RBR', 'FW', 'PDT', 'RBS']


### Target Gold Corpora ###
esl_wr_gold_dev = pos_reader_conllu(glob.glob(goldpath + "ESL_Written/*dev.conllu")) #500
esl_sp_gold_dev = pos_reader_conllu(glob.glob(goldpath + "L2*/*dev.conllu")) #741
ud_ewt_gold_dev = pos_reader_conllu(glob.glob(goldpath + "UD_English-EWT/*dev.conllu")) #1998
ud_gum_gold_dev = pos_reader_conllu(glob.glob(goldpath + "*/en_gum-ud-dev.conllu")) #1960; GUM + GUM Reddit

esl_wr_gold_test = pos_reader_conllu(glob.glob(goldpath + "ESL_Written/*test.conllu")) #500
esl_sp_gold_test = pos_reader_conllu(glob.glob(goldpath + "L2*/*test.conllu")) #741
ud_ewt_gold_test = pos_reader_conllu(glob.glob(goldpath + "UD_English-EWT/*test.conllu")) #2076
ud_gum_gold_test = pos_reader_conllu(glob.glob(goldpath + "*/en_gum-ud-test.conllu")) #1990; GUM + GUM Reddit

#trained on POS sets:

#Trained with all SL2T POS data (as applicable):
l1comb_trf_small = spacy_path + "en_ud_L1_combined_trf-0.0.1/en_ud_L1_combined_trf/en_ud_L1_combined_trf-0.0.1"
l1l2comb_trf_small_pos = spacy_path + "en_ud_L1L2_combined_trf-0.0.1/en_ud_L1L2_combined_trf/en_ud_L1L2_combined_trf-0.0.1"
l1l2ecomb_trf_small_pos = spacy_path + "en_ud_L1L2e_combined_trf-0.0.1/en_ud_L1L2e_combined_trf/en_ud_L1L2e_combined_trf-0.0.1"

gc_list = [(esl_sp_gold_dev,"L2 Spoken Dev"),(esl_wr_gold_dev,"L2 Written Dev"),(ud_ewt_gold_dev,"EWT Dev"),(ud_gum_gold_dev,"GUM Dev"),(esl_sp_gold_test,"L2 Spoken Test"),(esl_wr_gold_test,"L2 Written Test"),(ud_ewt_gold_test,"EWT Test"),(ud_gum_gold_test,"GUM Test")]

tgr_listsp_small_pos = [(spacy_tag,l1comb_trf_small,"Spacy_trf UD"),(spacy_tag,l1l2comb_trf_small_pos,"Spacy_trf UD + L2"),(spacy_tag,l1l2ecomb_trf_small_pos,"Spacy_trf UD + L2 Equal")]

spacy_tagger_results_20220330 = pos_compile_full(gc_list,tgr_listsp_small_pos)
pickle.dump(spacy_tagger_results_20220330,open(report_path + 'spacy_small_results.pickle',"wb"))
spacy_report_20220330 = pos_table_maker_summary(spacy_tagger_results_20220330,target_tags_fs)
wpos_report(spacy_report_20220330,report_path + "pos_summary.tsv")

### Dependency Analyses ###

#target dependency tags:
tdeps = ["case","nsubj","det","root","advmod","amod","obj","obl","nmod","mark","conj","cc","compound","aux","cop","nmod:poss","advcl","xcomp","acl:relcl","ccomp","nummod","flat","acl","aux:pass","parataxis","appos","nsubj:pass","discourse","compound:prt","expl","fixed","dep","obl:tmod","obl:npmod","nmod:tmod","iobj","csubj","list","det:predet","reparandum","goeswith","nmod:npmod","cc:preconj","vocative","obl:agent","orphan","dislocated","csubj:pass","flat:foreign"]

esl_wr_gold_dep_test = conllu_extract_dep_full(glob.glob(goldpath + "ESL_Written/*test.conllu")) #500
esl_sp_gold_dep_test = conllu_extract_dep_full(glob.glob(goldpath + "L2*_dep/*test.conllu")) #232
ud_ewt_gold_dep_test = conllu_extract_dep_full(glob.glob(goldpath + "UD_English-EWT/*test.conllu")) #2001
ud_gum_gold_dep_test = conllu_extract_dep_full(glob.glob(goldpath + "*/en_gum-ud-test.conllu")) #1960; GUM + GUM Reddit


l1comb_trf = spacy_path + "en_ud_L1_combined_trf-0.0.1/en_ud_L1_combined_trf/en_ud_L1_combined_trf-0.0.1"
l1l2comb_trf = spacy_path + "en_ud_L1L2_combined_trf-0.0.1/en_ud_L1L2_combined_trf/en_ud_L1L2_combined_trf-0.0.1"
l1l2ecomb_trf = spacy_path + "en_ud_L1L2e_combined_trf-0.0.1/en_ud_L1L2e_combined_trf/en_ud_L1L2e_combined_trf-0.0.1"

prs_list = [(spacy_parse_cpos,l1comb_trf,"Spacy_L1_UD_TRF"),(spacy_parse_cpos,l1l2comb_trf,"Spacy_L1L2_UD_TRF"),(spacy_parse_cpos,l1l2ecomb_trf,"Spacy_L1L2e_UD_TRF")] #list of (tagger,model,name)
gdc_list = [(esl_sp_gold_dep_test,"L2 Spoken"),(esl_wr_gold_dep_test,"L2 Written"),(ud_ewt_gold_dep_test,"EWT"),(ud_gum_gold_dep_test,"GUM")]

dep_comp_test = dep_compile_full(gdc_list,prs_list)
pickle.dump(dep_comp_test,open(report_path + 'spacy_dep_results.pickle',"wb"))
dep_las_tbl = dep_table_maker_summary(dep_comp_test,tdeps,acctype = "LAS")
# dep_ulas_tbl = dep_table_maker_summary(dep_comp,tdeps,acctype = "ULAS")
# dep_ls_tbl = dep_table_maker_summary(dep_comp,tdeps,acctype = "LS")



wpos_report(dep_las_tbl, report_path + "dep_las_summary_f1.tsv")
# wpos_report(dep_ulas_tbl, report_path + "dep_ulas_summary_f1.tsv")
# wpos_report(dep_ls_tbl, report_path + "dep_ls_summary_f1.tsv")

### Get basic stats:

bea_lists = glob.glob(goldpath + "_POS Sets/*.conllu") + glob.glob(goldpath + "L2_spok*/*.conllu") + glob.glob(goldpath + "UD*/*.conllu") + glob.glob(goldpath + "ESL_Written/*.conllu")
bea_stats = basic_stats(bea_lists)

wbasic_stats(bea_stats,report_path + "bea_basic_stats_dep.tsv")
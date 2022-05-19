#####################
### For BEA Paper ###
#####################

### POS Analyses ###

### Target Gold Corpora ###
goldpath = '/Users/kristopherkyle/Desktop/Corpora/Accuracy Project/'
esl_wr_gold_dev = pos_reader_conllu(glob.glob(goldpath + "ESL_Written/*dev.conllu")) #500
esl_sp_gold_dev = pos_reader_conllu(glob.glob(goldpath + "L2*/*dev.conllu")) #741
ud_ewt_gold_dev = pos_reader_conllu(glob.glob(goldpath + "UD_English-EWT/*dev.conllu")) #1998
ud_gum_gold_dev = pos_reader_conllu(glob.glob(goldpath + "*/en_gum-ud-dev.conllu")) #1960; GUM + GUM Reddit

esl_wr_gold_test = pos_reader_conllu(glob.glob(goldpath + "ESL_Written/*test.conllu")) #500
esl_sp_gold_test = pos_reader_conllu(glob.glob(goldpath + "L2*/*test.conllu")) #741
ud_ewt_gold_test = pos_reader_conllu(glob.glob(goldpath + "UD_English-EWT/*test.conllu")) #2076
ud_gum_gold_test = pos_reader_conllu(glob.glob(goldpath + "*/en_gum-ud-test.conllu")) #1990; GUM + GUM Reddit

#POS models:
spacy_path = '/Users/kristopherkyle/Desktop/Programming/SpacyModels/'
#trained on POS sets:

#Trained with all SL2T POS data (as applicable):
l1comb_trf_small = spacy_path + "en_ud_L1_combined_trf-0.0.1/en_ud_L1_combined_trf/en_ud_L1_combined_trf-0.0.1"
l1l2comb_trf_small_pos = spacy_path + "en_ud_L1L2_combined_trf-0.0.1/en_ud_L1L2_combined_trf/en_ud_L1L2_combined_trf-0.0.1"
l1l2ecomb_trf_small_pos = spacy_path + "en_ud_L1L2e_combined_trf-0.0.1/en_ud_L1L2e_combined_trf/en_ud_L1L2e_combined_trf-0.0.1"

gc_list = [(esl_sp_gold_dev,"L2 Spoken Dev"),(esl_wr_gold_dev,"L2 Written Dev"),(ud_ewt_gold_dev,"EWT Dev"),(ud_gum_gold_dev,"GUM Dev"),(esl_sp_gold_test,"L2 Spoken Test"),(esl_wr_gold_test,"L2 Written Test"),(ud_ewt_gold_test,"EWT Test"),(ud_gum_gold_test,"GUM Test")]

tgr_listsp_small_pos = [(spacy_tag,l1comb_trf_small,"Spacy_trf UD"),(spacy_tag,l1l2comb_trf_small_pos,"Spacy_trf UD + L2"),(spacy_tag,l1l2ecomb_trf_small_pos,"Spacy_trf UD + L2 Equal")]

spacy_tagger_results_20220330 = pos_compile_full(gc_list,tgr_listsp_small_pos)
pickle.dump(spacy_tagger_results_20220330,open('/Volumes/GoogleDrive/My Drive/2019_20_LL_POS/Datasets/annotators/2021 zzz Final/20220330_bea/spacy_small_results_20220330.pickle',"wb"))
spacy_report_20220330 = pos_table_maker_summary(spacy_tagger_results_20220330,target_tags_fs)
wpos_report(spacy_report_20220330,"/Volumes/GoogleDrive/My Drive/2019_20_LL_POS/Datasets/annotators/2021 zzz Final/20220330_bea/spacy_small_pos_summary_20220330.tsv")

### Dependency Analyses ###

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

goldpath = '/Users/kristopherkyle/Desktop/Corpora/Accuracy Project/'
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
pickle.dump(dep_comp_test,open('/Volumes/GoogleDrive/My Drive/2019_20_LL_POS/Datasets/annotators/2021 zzz Final/20220330_bea/spacy_dep_results_20220330.pickle',"wb"))
dep_las_tbl = dep_table_maker_summary(dep_comp_test,tdeps,acctype = "LAS")
# dep_ulas_tbl = dep_table_maker_summary(dep_comp,tdeps,acctype = "ULAS")
# dep_ls_tbl = dep_table_maker_summary(dep_comp,tdeps,acctype = "LS")


report_path = "/Volumes/GoogleDrive/My Drive/2019_20_LL_POS/Datasets/annotators/2021 zzz Final/20220330_bea/"
wpos_report(dep_las_tbl, report_path + "dep_las_summary_f1_20220330.tsv")
# wpos_report(dep_ulas_tbl, report_path + "dep_ulas_summary_f1_20220315.tsv")
# wpos_report(dep_ls_tbl, report_path + "dep_ls_summary_f1_20220315.tsv")

# test = spacy_parse_cpos(esl_sp_gold_dep_test,l1l2comb_trf)
# for x in test:
# 	print(x)

### Get basasic stats:

bea_lists = glob.glob(goldpath + "_POS Sets/*.conllu") + glob.glob(goldpath + "L2_spok*/*.conllu") + glob.glob(goldpath + "UD*/*.conllu") + glob.glob(goldpath + "ESL_Written/*.conllu")
bea_stats = basic_stats(bea_lists)

wbasic_stats(bea_stats,report_path + "bea_basic_stats_dep.tsv")
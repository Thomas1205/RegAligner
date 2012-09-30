include common/Makefile.common
INCLUDE += -I common/ -I $(CBCPATH)Clp/src/ -I $(CBCPATH)CoinUtils/src/ -I $(CBCPATH)Clp/inc/ -I $(CBCPATH)CoinUtils/inc/ -I ../../linprog/ -I $(CBCPATH)Cbc/inc/ -I $(CBCPATH)Cbc/src/ -I $(CBCPATH)/Osi/inc/ -I $(CBCPATH)/Osi/src/Osi/ -I $(CBCPATH)/Osi/src/ -I $(CBCPATH)/Clp/src/OsiClp/ -I $(CBCPATH)Cgl/src/ 


#if you have CBC, outcomment this
#DEBUGFLAGS += -DHAVE_CONFIG_H -DHAS_CBC
#OPTFLAGS   += -DHAVE_CONFIG_H -DHAS_CBC
#CBCLINK = $(CBCPATH)Clp/src/.libs/libClp.so $(CBCPATH)Cgl/src/.libs/libCgl.so $(CBCPATH)Cbc/src/.libs/libCbc.so $(CBCPATH)CoinUtils/src/.libs/libCoinUtils.so $(CBCPATH)/Clp/src/OsiClp/.libs/libOsiClp.so $(CBCPATH)Osi/src/Osi/.libs/libOsi.so $(CBCPATH)Osi/src/OsiGrb/.libs/libOsiGrb.so -llapack


#if you have GZSTREAM, outcomment these lines:
#DEBUGFLAGS += -DHAS_GZSTREAM
#OPTFLAGS += -DHAS_GZSTREAM
#GZLINK = thirdparty/libgzstream.a -lz
#INCLUDE += -I thirdparty/

all : $(DEBUGDIR) $(OPTDIR) regaligner_swb.debug.L64 regaligner_swb.opt.L64 extractvoc.opt.L64 plain2indices.opt.L64 common/lib/commonlib.debug common/lib/commonlib.opt

common/lib/commonlib.opt : 
	cd common; make; cd -

common/lib/commonlib.debug :
	cd common; make; cd -


extractvoc.opt.L64 : extract_vocabulary.cc common/lib/commonlib.opt
	$(LINKER) $(OPTFLAGS) $(INCLUDE) extract_vocabulary.cc common/lib/commonlib.opt $(GZLINK) -o $@

plain2indices.opt.L64 : plain2indices.cc common/lib/commonlib.opt
	$(LINKER) $(OPTFLAGS) $(INCLUDE) plain2indices.cc common/lib/commonlib.opt common/$(DEBUGDIR)/makros.o $(GZLINK) -o $@


regaligner_swb.debug.L64 : regaligner_swb.cc common/lib/commonlib.debug $(DEBUGDIR)/training_common.o $(DEBUGDIR)/ibm1_training.o $(DEBUGDIR)/ibm2_training.o $(DEBUGDIR)/ibm3_training.o $(DEBUGDIR)/ibm4_training.o $(DEBUGDIR)/hmm_training.o common/$(DEBUGDIR)/stringprocessing.o common/$(DEBUGDIR)/combinatoric.o  $(DEBUGDIR)/alignment_computation.o $(DEBUGDIR)/singleword_fertility_training.o $(DEBUGDIR)/alignment_error_rate.o $(CBCLINK) $(DEBUGDIR)/alignment_error_rate.o $(DEBUGDIR)/corpusio.o 
	$(LINKER) $(DEBUGFLAGS) $(INCLUDE) regaligner_swb.cc common/lib/commonlib.debug $(DEBUGDIR)/training_common.o $(DEBUGDIR)/ibm1_training.o $(DEBUGDIR)/ibm2_training.o $(DEBUGDIR)/ibm3_training.o $(DEBUGDIR)/ibm4_training.o $(DEBUGDIR)/hmm_training.o common/$(OPTDIR)/matrix.o  common/$(DEBUGDIR)/combinatoric.o $(DEBUGDIR)/alignment_computation.o  $(DEBUGDIR)/singleword_fertility_training.o $(DEBUGDIR)/alignment_error_rate.o $(DEBUGDIR)/corpusio.o common/$(DEBUGDIR)/fileio.o $(CBCLINK) $(GZLINK) -ldl -lm -lc -lz  -o $@

regaligner_swb.opt.L64 : regaligner_swb.cc common/lib/commonlib.opt $(OPTDIR)/training_common.o $(OPTDIR)/ibm1_training.o $(OPTDIR)/ibm2_training.o $(OPTDIR)/ibm3_training.o $(OPTDIR)/ibm4_training.o $(OPTDIR)/hmm_training.o common/$(OPTDIR)/stringprocessing.o common/$(OPTDIR)/combinatoric.o  $(OPTDIR)/alignment_computation.o $(OPTDIR)/singleword_fertility_training.o $(OPTDIR)/alignment_error_rate.o  $(CBCLINK) $(OPTDIR)/alignment_error_rate.o $(OPTDIR)/corpusio.o 
	$(LINKER) $(OPTFLAGS) $(INCLUDE) regaligner_swb.cc common/lib/commonlib.debug $(OPTDIR)/training_common.o $(OPTDIR)/ibm1_training.o $(OPTDIR)/ibm2_training.o $(OPTDIR)/ibm3_training.o $(OPTDIR)/ibm4_training.o $(OPTDIR)/hmm_training.o common/$(OPTDIR)/matrix.o  common/$(OPTDIR)/combinatoric.o $(OPTDIR)/alignment_computation.o  $(OPTDIR)/singleword_fertility_training.o common/$(OPTDIR)/fileio.o $(OPTDIR)/alignment_error_rate.o $(OPTDIR)/corpusio.o  $(CBCLINK) $(GZLINK) -ldl -lm -lc -lz  -o $@



include common/Makefile.finish

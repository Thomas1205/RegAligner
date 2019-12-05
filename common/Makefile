include Makefile.common

all: $(LIB) $(DEBUGDIR) $(OPTDIR) $(LIB)/commonlib.debug $(LIB)/commonlib.opt $(OPTDIR)/vector.o $(OPTDIR)/matrix.o $(OPTDIR)/tensor.o

$(LIB)/commonlib.debug: $(DEBUGDIR)/fileio.o $(DEBUGDIR)/stringprocessing.o $(DEBUGDIR)/application.o $(DEBUGDIR)/timing.o $(OPTDIR)/vector.o $(DEBUGDIR)/matrix.o $(DEBUGDIR)/tensor.o $(DEBUGDIR)/makros.o $(DEBUGDIR)/combinatoric.o 
	ar rs $@ $(DEBUGDIR)/fileio.o $(DEBUGDIR)/stringprocessing.o $(DEBUGDIR)/application.o $(DEBUGDIR)/timing.o $(OPTDIR)/vector.o $(DEBUGDIR)/matrix.o $(DEBUGDIR)/tensor.o  $(DEBUGDIR)/makros.o  $(DEBUGDIR)/combinatoric.o

$(LIB)/commonlib.opt: $(OPTDIR)/fileio.o $(OPTDIR)/stringprocessing.o $(OPTDIR)/application.o $(OPTDIR)/timing.o $(OPTDIR)/vector.o $(OPTDIR)/matrix.o $(OPTDIR)/tensor.o  $(OPTDIR)/makros.o  $(OPTDIR)/combinatoric.o 
	ar rs $@ $(OPTDIR)/fileio.o $(OPTDIR)/stringprocessing.o $(OPTDIR)/application.o $(OPTDIR)/timing.o $(OPTDIR)/vector.o $(OPTDIR)/matrix.o $(OPTDIR)/tensor.o  $(OPTDIR)/makros.o  $(OPTDIR)/combinatoric.o

clean:
	rm -f $(DEBUGDIR)/*.o 
	rm -f $(OPTDIR)/*.o
	rm -f lib/* 

include Makefile.finish


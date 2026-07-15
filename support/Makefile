PACKAGE=problemtools
PROGRAMS=default_validator interactive

all:
	$(foreach prog,$(PROGRAMS),$(MAKE) -C $(prog) &&) true

install: all
	install -d $(DESTDIR)
	$(foreach prog,$(PROGRAMS),install $(prog)/$(prog) $(DESTDIR) &&) true
	install default_grader/default_grader $(DESTDIR)
	cp viva/viva.jar $(DESTDIR)
	install viva/viva.sh $(DESTDIR)

clean:
	$(foreach prog,$(PROGRAMS),$(MAKE) -C $(prog) clean;)

all:
	make -C support

builddeb:
	dpkg-buildpackage -us -uc -tc -b

clean:
	make -C support clean
	rm -rf problemtools.egg-info build

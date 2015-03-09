all: html

html:
	install -d _build
	rm -rf _build/env
	virtualenv --system-site-packages _build/env
	(cd .. && ./doc/_build/env/bin/python setup.py install)
	./_build/env/bin/python `which sphinx-build` -b html . _build/html

clean:
	rm -rf _build

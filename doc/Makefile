all: html

html:
	install -d _build
	rm -rf _build/env
	virtualenv --system-site-packages _build/env
	(cd .. && ./doc/_build/env/bin/python setup.py install)
	./_build/env/bin/python `which sphinx-build` -b html . _build/html

gh-pages: clean html
	rm -rf _build/gh-pages
	mkdir -p _build/gh-pages
	git -C _build/gh-pages init
	git -C _build/gh-pages checkout -b gh-pages
	rsync -a _build/html/ _build/gh-pages/
	touch _build/gh-pages/.nojekyll
	git -C _build/gh-pages add .
	git -C _build/gh-pages commit -m "Generated from sources" -a
	git branch -D gh-pages || true
	git fetch $(CURDIR)/_build/gh-pages
	git branch gh-pages FETCH_HEAD
	rm -rf _build/gh-pages
	@echo ""
	@echo "The gh-pages branch is now updated, you can now push it to github: git push -f origin gh-pages"
	@echo ""

clean:
	rm -rf _build

.PHONY: all html gh-pages clean

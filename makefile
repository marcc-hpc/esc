.PHONY: serve serveback notebooks build
all: 
serveback:
	# this might not work if the screen is dead
	cd docs && if ! (screen -ls | egrep "[0-9]+.jekyll-serve\.+tached"); then screen -LdmS jekyll-serve bundle exec jekyll serve -H 0.0.0.0; else echo "already running"; fi
serve:
	cd docs && bundle exec jekyll serve -H 0.0.0.0
notebooks:
	make -C notebooks
build: 
	cd docs && bundle exec jekyll build

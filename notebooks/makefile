COLLECTION_SPOT:=../docs/_notebooks
NOTEBOOKS:=$(shell find . ! -path '*/.ipynb_checkpoints/*' -name "*.ipynb")
NOTEBOOKS_RENDERED:=$(patsubst %.ipynb,$(COLLECTION_SPOT)/%.md,$(NOTEBOOKS))
all: $(NOTEBOOKS_RENDERED)
$(COLLECTION_SPOT)/%.md: %.ipynb
	@mkdir -p $(shell dirname $@)
	jupyter nbconvert --to markdown --config notebook_config.py $<

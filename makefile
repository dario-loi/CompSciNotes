

MARKDOWN_FILES = $(wildcard *.md)
PANDOC_FLAGS = --filter pandoc-plot --top-level-division=chapter --toc


all: pdf merge

## Convert all markdown files to pdf
pdf: $(MARKDOWN_FILES:.md=.pdf)

%.pdf: %.md
	@echo "Converting $< to $@"
	@pandoc $(PANDOC_FLAGS)  $< -o $@

merge: semester.pdf

semester.pdf: $(MARKDOWN_FILES)
	@echo "Merging all markdown files to $@"
	@pandoc $(PANDOC_FLAGS)  $(MARKDOWN_FILES) -o $@

clean:
	@rm -f *.pdf

.PHONY: clean


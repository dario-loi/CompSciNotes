

MARKDOWN_FILES = $(wildcard *.md)
PANDOC_FLAGS = --top-level-division=chapter --toc -H format.tex -V colorlinks=true -V linkcolor=blue -V urlcolor=red


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
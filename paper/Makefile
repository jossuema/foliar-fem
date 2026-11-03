# Builds the LXAI camera-ready. The .sty files live in sty/ because BasicTeX
# does not ship environ, and the NeurIPS styles are not in TeX Live.
#
#   make           -> paper.pdf, and copies it to foliar_fem_lxai_cameraready.pdf
#   make check     -> local-path leaks, PDF metadata, 8-page body limit.
#                     Runs on the versioned deliverable by default, so it works
#                     on a fresh clone without building anything.
#   make figs      -> regenerates the seven paper figures
#   make cost      -> the cost table of the three levels
#
# paper.tex is the accepted CAMERA-READY: it carries 'final' and the \author
# block, so its check is not the anonymity one but the leak check (local paths)
# plus the page limit.

PDF ?= foliar_fem_lxai_cameraready.pdf

all: paper.pdf

paper.pdf: paper.tex refs.bib $(wildcard fig/*.pdf)
	cp sty/neurips_2025.sty sty/environ.sty sty/trimspaces.sty .
	pdflatex -interaction=nonstopmode paper
	bibtex paper
	pdflatex -interaction=nonstopmode paper
	pdflatex -interaction=nonstopmode paper
	pdflatex -interaction=nonstopmode paper
	cp paper.pdf foliar_fem_lxai_cameraready.pdf
	@$(MAKE) --no-print-directory check PDF=paper.pdf

# There is no anonymity left to verify, but a local path or the machine
# username inside the PDF is still a leak.
check:
	@echo "--- LXAI camera-ready · $(PDF) ---"
	@echo "local path or username leaks in the PDF"
	@strings $(PDF) | grep -icE "/Users/|scratchpad" \
	  | awk '{ if ($$1>0) print "  FAIL: " $$1 " matches"; else print "  OK" }'
	@echo "metadata (in the final version it MUST be filled in)"
	@pdfinfo $(PDF) | grep -E "^(Author|Title):" \
	  | awk -F: '{ v=$$2; gsub(/^[ \t]+|[ \t]+$$/,"",v); if (v=="") print "  MISSING: " $$1 " empty"; else print "  OK: " $$0 }'
	@echo "pages (limit 8, not counting references or appendices)"
	@python3 checkpages.py $(PDF) 8

figs:
	python3 scripts/figs.py

cost:
	python3 scripts/cost.py --repeats 7

clean:
	rm -f paper.aux paper.bbl paper.blg paper.log paper.out paper.pdf \
	      *.sty pg-*.png
	@echo "  (foliar_fem_lxai_cameraready.pdf is NOT removed: it is the deliverable)"

.PHONY: all check figs cost clean

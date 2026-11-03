#!/usr/bin/env python3
"""Check the CFP page limit: the BODY must fit in N pages, not counting
references or appendices.

    python3 checkpages.py [pdf] [limit]      # defaults: paper.pdf and 8

Looking at which page 'References' starts on is not enough: if it starts at the
very top of page N+1, the body ended on page N and that is fine.

The appendices come AFTER the bibliography, so the sentinel is still
'References': everything after it -- bibliography and appendices -- falls
outside the count, which is what the CFP asks for.
"""
import subprocess
import sys

pdf = sys.argv[1] if len(sys.argv) > 1 else 'paper.pdf'
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 8

n = int(subprocess.run(['pdfinfo', pdf], capture_output=True, text=True)
        .stdout.split('Pages:')[1].split()[0])
for pg in range(1, n + 1):
    lines = subprocess.run(['pdftotext', '-f', str(pg), '-l', str(pg), pdf, '-'],
                           capture_output=True, text=True).stdout.split('\n')
    if not any(l.strip() == 'References' for l in lines):
        continue
    i = next(k for k, l in enumerate(lines) if l.strip() == 'References')
    # bare numbers are the line numbering of submission mode
    body = [l for l in lines[:i] if l.strip() and not l.strip().isdigit()]
    last = pg if body else pg - 1
    ok = last <= limit
    print(f"  {n} pages | references from p{pg} | the body ends on p{last} -> "
          + ("OK" if ok else f"OVER BY {last - limit} page(s)"))
    sys.exit(0 if ok else 1)
print("  could not find the References section")
sys.exit(1)

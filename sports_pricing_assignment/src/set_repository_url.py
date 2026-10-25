"""Insert your published repository link into the article before rebuilding it."""
from pathlib import Path
from urllib.parse import urlparse
import sys

if len(sys.argv)!=2:
    raise SystemExit('Usage: python src/set_repository_url.py https://github.com/USERNAME/REPOSITORY')
url=sys.argv[1].strip()
u=urlparse(url)
if u.scheme!='https' or u.netloc not in {'github.com','gitlab.com'} or len(u.path.strip('/').split('/'))<2 or any(c in url for c in '{}\\%#'):
    raise SystemExit('Supply an HTTPS GitHub/GitLab repository URL without a fragment or special TeX characters.')
root=Path(__file__).resolve().parents[1]
(root/'paper/repository.tex').write_text('Code, synthetic data, run outputs and manuscript source are available at \\url{'+url+'}.\n')
print('Updated paper/repository.tex. Rebuild paper/main.tex to update the PDF.')

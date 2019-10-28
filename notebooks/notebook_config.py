
# via https://gist.github.com/tgarc/7d6901858ef708030c19

# python 2/3 compatibility
try: from urllib.parse import quote
except ImportError: from urllib2 import quote
import os,sys

fn = None
for arg in sys.argv:
    if arg.endswith('.ipynb'):
        fn = arg.split('.ipynb')[0]
        break

c = get_config()
c.NbConvertApp.export_format = 'markdown'
c.MarkdownExporter.template_path = [os.path.dirname(__file__)]
c.MarkdownExporter.template_file = 'jupyter'
c.Application.verbose_crash = True

def path2support(path):
    """Turn a file path into a URL"""
	# modify this function to point your images to a custom path
	# by default this saves all images to a directory 'images' in 
	# the root of the blog directory
    return '{{ BASE_PATH }}/images/' + os.path.basename(path)
def path2support(path): 
    # to get the correct path we must remove one root folder
    fn = os.path.sep.join(path.split(os.path.sep)[1:])
    print(fn)
    return fn
c.MarkdownExporter.filters = {'path2support': path2support}

if fn:
    c.NbConvertApp.output_base = fn.lower().replace(' ', '-')
    c.FilesWriter.build_directory = os.path.realpath('../docs/_notebooks')

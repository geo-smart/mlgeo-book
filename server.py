from http.server import HTTPServer, SimpleHTTPRequestHandler
import sys
import os

# GLOBALS
build = True
silent = False
directory = "book/_build/html"
geosmart = r"""
   _____            _____ __  __          _____ _______ 
  / ____|          / ____|  \/  |   /\   |  __ \__   __|
 | |  __  ___  ___| (___ | \  / |  /  \  | |__) | | |   
 | | |_ |/ _ \/ _ \\___ \| |\/| | / /\ \ |  _  /  | |   
 | |__| |  __/ (_) |___) | |  | |/ ____ \| | \ \  | |   
  \_____|\___|\___/_____/|_|  |_/_/    \_\_|  \_\ |_|   
"""

def output(data):
  if not silent:
    print(data)

# Found this funky little function on stack overflow:
# https://stackoverflow.com/questions/40419276/python-how-to-print-text-to-console-as-hyperlink
def link(uri, label=None):
    if label is None: 
        label = uri
    parameters = ''

    # OSC 8 ; params ; URI ST <name> OSC 8 ;; ST 
    escape_mask = '\033]8;{};{}\033\\{}\033]8;;\033\\'

    return escape_mask.format(parameters, uri, label)

def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler):
  server_address = ("", 8000)
  httpd = server_class(server_address, handler_class)
  httpd.serve_forever()

class NoExtensionHandler(SimpleHTTPRequestHandler):
  def do_GET(self):
    print("\nServing path:", self.path)
    self.path = directory + self.path
    print("Prepending directory:", self.path)

    SimpleHTTPRequestHandler.do_GET(self)

def main():
  global directory

  if not silent:
    clear = lambda: os.system('cls' if os.name=='nt' else 'clear')
    clear()
  output(geosmart)

  if build:
    output("Attempting to build jupyter notebook:")
    os.system("jb build book")
  else:
    output("Detected '--no-build' flag, skipping jb build.")
  
  # path = os.getcwd()
  # directory = ""

  # sub_folders = ["binder", "book", "conda"]
  # if any(folder in path for folder in sub_folders):
  #   output(f"Detected execution starting in non-root directory, adjusting relative paths.")
  #   directory = ".."
  # else:
  #   output("Execution starting in root directory, using normal paths.")
  # directory += "/book/_build/html"

  output(f"Serving book from {directory} directory...")
  output("Navigate to localhost:8000 in your browser to view the book!")
  output("Or, if your terminal supports links: " + link("http://localhost:8000/"))
  run(HTTPServer, NoExtensionHandler)

if len(sys.argv) == 1:
  build = True
  silent = False
  main()

elif len(sys.argv) == 2:
  flags = ["--no-build", "--nb"]

  if sys.argv[1] in flags:
    build = False
    silent = False
    main()
  else:
    print(f"Invalid flag provided. Expected one of {str(flags)}, got '{sys.argv[1]}'")

else:
  print(f"Invalid number of arguments. Expected at most 1, got {len(sys.argv) - 1}")
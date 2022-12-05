import json
import sys
import os
import yaml

def cleanBook(book_in, book_out):
  divider = "-" * max(len(book_in), len(book_out))
  
  print(divider)
  print("Running book cleaning for file: ")
  print(book_in, "\n")

  # Grabbing the file data and reading it as a json
  with open(book_in, 'r') as file:
    data = json.load(file) # returns a dict

    if not data:
      print("Failed to open file. Aborting.")
      sys.exit("Successfully terminated execution.")
    else:
      print("Data retrieved successfully.")

  # Reading through the file data and modifying appropriately
  cells = data["cells"]
  prev_marked = False
  for i in range(len(cells)):
    print(f"\nScanning cell #{i} of {len(cells)}")

    if cells[i]["cell_type"] == "code" and prev_marked:
      print("Marked 'code' cell found, clearing content...")
      cells[i]["source"] = ["# implement answer here"]
      prev_marked = False

    if cells[i]["cell_type"] == "markdown":
      print("Cell is 'markdown' type, checking source...")
      content = cells[i]["source"]
      if (len(content) > 0 and content[0] == "```{admonition} Student response section\n"):
        print("Student response marker found.")
        prev_marked = True
      else:
        print("No marker found.")

  # Writing the modified data
  with open(book_out, 'w', encoding='utf-8') as out_file:
    print("\nAttempting to write to output file...\n")
    json.dump(data, out_file, ensure_ascii=False, indent=2)

  print("Completed book cleaning, output file: ")
  print(book_out)
  print(divider)

# MAIN STARTS HERE
print("\nExamining table of contents for jupyter books")
print("=============================================")

with open("./book/_toc.yml") as raw:
  toc = yaml.safe_load(raw)
  # could add exclusion pattern?

books = []
for part in toc["parts"]:
  print("\nChecking chapters contained in", part["caption"])
  for chapter in part["chapters"]:
    print("Chapter data:", chapter)
    if "file" in chapter.keys():
      path = "./book/" + chapter["file"] + ".ipynb"
      if os.path.exists(path):
        books.append(path)
        print("> Added path:", path)

print(f"\nFound {len(books)} jupyter notebooks to clean.")
print("Starting cleaning process...")

for book in books:
  # six for length of .ipynb extension
  output = book[:-6] + "_cleaned.ipynb"
  cleanBook(book, output)

  # TODO: REMOVE LATER!! FOR TESTING PURPOSES ONLY!!
  os.remove(output)

print("\n=================================================")
print(f"Finished book cleaning for all {len(books)} books\n")
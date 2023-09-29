import shutil
import os

def replaceFile(target, replacement):
  print(f"\nAttempting to replace {target} with {replacement}...")

  try:
    shutil.copyfile(replacement, target) # copy the content of the source file to the destination file
    os.remove(replacement) # delete the source file
    
    print(f"Content from '{replacement}' has been copied to '{target}', '{replacement}' has been deleted.")
  except FileNotFoundError:
    print(f"One or both of {target, replacement} does not exist.")
  except Exception as e:
    print(f"An unknown error occurred: {e}")

replaceFile("README.md", "STUDENT_README.md")
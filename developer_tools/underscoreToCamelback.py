import sys

def findOccurences(s, ch):
  """
  Returns a list of the indexes of s in which ch occurs
  @ In, s, string, string to be searched in
  @ In, ch, string, character to be searched for
  @ Out, list, list, list of indexes
  """
  return [i for i, letter in enumerate(s) if letter == ch]

def convert(filename, output):
  """
  Converts underscore variable name to camelback
  @ In, filename, string, path to file to be converted
  @ In, output, string, path to converted file
  @ Out, None
  """
  line_list = []
  with open(filename, 'r') as file:
    for line in file:
      if '_' in line:
        letter_list = []
        index_list = findOccurences(line, '_')
        for i in index_list:
          if line[i+1].isalnum()
            letter_list.append(line[i + 1])
        for letter in letter_list:
          line = line.replace('_' + letter, letter.capitalize())
      line_list.append(line)

  with open(output, 'w') as outputfile:
    for item in line_list:
        outputfile.write(item)

convert(sys.argv[1], sys.argv[2])
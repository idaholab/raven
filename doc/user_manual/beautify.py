# author: maljdp
# date: 1/22/15
# A work in progress script to auto-format the tex documents to my personal
# style.
# This document stands as a testament that scripting is not always faster...

from glob import glob
import re

texFiles = glob('*.tex')
lineWidth = 80
indentSize = 2

def GetIndent(string):
  indentCount  = string.count('{') + string.count('\\begin{itemize}')
  indentCount -= string.count('}') + string.count('\\end{itemize}')
  if(indentCount > 0):
    return indentCount*indentSize*' '
  else:
    return ''

for f in texFiles:
  with open(f) as myFile:
    newString = ''
    indent = ''
    codeListing = False
    openBraces = 0
    closeBraces = 0
    for line in myFile:
      commentLine = False
      # Remove trailing whitespace
      line = line.rstrip()

#      # If there was room on the last line, put more stuff there, but only if
#      # the last line is a letter
#      if len(newString) > 0 and newString[-1] not in ['\n']:
#        if newString[-1] in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ' and not line.strip().startswith('\\end{') and not line.strip().startswith('\\begin{'):
#          temp = newString.rsplit('\n',1)
#          if len(temp) == 2:
#            newString = temp[0] + '\n'
#            print(temp[1].strip())
#            line = temp[1].strip() + line
#        else:
#          newString += '\n' # It must be a special line, put the newline back in

      indent = GetIndent(newString)

      # Blank lines represent new paragraphs and should be preserved, however we
      # don't need more than one blank line
      if line == '' and newString.split('\n')[-1] == '':
        newString += '\n'
        continue
      elif line == '':
        newString += '\n'
        continue

      # Do not attempt to format any commented lines
      if line.strip().startswith('%') or line.strip().startswith('\\alfoa') \
         or line.strip().startswith('\\maljdan'):
        newString += line + '\n'
        continue

      # If it is a code listing, leave it alone, don't try to format it
      if line.strip().startswith('\\begin{lstlisting}'):
        newString += line + '\n'
        codeListing = True
        continue
      elif line.strip().startswith('\\end{lstlisting}'):
        newString += line + '\n'
        codeListing = False
        continue
      elif codeListing:
        newString += line + '\n'
        continue

      # If the line's length  > lineWidth, then start storing lineWidth
      # components
      if len(indent+line) > lineWidth:
        currentLine = indent
        sep = ''
        tokens = filter(None,line.split(' '))
        for token in tokens:
          if len(currentLine + sep + token) > lineWidth:
            #If we haven't got anywhere just print it
            if currentLine == indent:
              newString += currentLine + token + '\n'
              indent = GetIndent(newString)
              currentLine = indent
              sep = ''
            else:
              newString += currentLine + '\n'
              indent = GetIndent(newString)
              currentLine = indent + token
              sep = ' '
          else:
            currentLine += sep+token
            sep = ' '
          # No matter what happens if we have reached the end of a sentence,
          # then we need a new line
          if currentLine.endswith('.') and 'e.g.' not in token and 'i.e.' not in token:
            newString += currentLine + '\n'
            indent = GetIndent(newString)
            newString += indent + '%\n'
            currentLine = indent
            sep = ''
        # There is something left in the buffer
        if currentLine != indent:
            newString += currentLine + '\n'
      else:
        if line.strip().startswith('\\begin{'):
          newString += line + '\n'
#          indent += indentSize*' '
        elif line.strip().startswith('\\end{'):
#          indent = indent[0:-indentSize]
          newString += line + '\n'
        else:
          # If we got here, it is a ``normal'' line, so we will look for periods
          # and split the lines there
          currentLine = indent
          sep = ''
          tokens = filter(None,line.split(' '))
          for token in tokens:
            currentLine += sep+token
            sep = ' '
            # No matter what happens if we have reached the end of a sentence,
            # then we need a new line
            if currentLine.endswith('.') and 'e.g.' not in token and 'i.e.' not in token:
              newString += currentLine + '\n'
              indent = GetIndent(newString)
              newString += indent + '%\n'
              currentLine = indent
              sep = ''
          # There is something left in the buffer
          if currentLine != indent:
            newString += currentLine + '\n'

#      openBraces = line.count('{')
#      closeBraces = line.count('}')
#      if(openBraces > closeBraces):
#        indent += (openBraces-closeBraces)*indentSize*' '
#      if(openBraces < closeBraces):
#        indent = indent[0:-indentSize*(closeBraces-openBraces)]

  while '%\n%\n' in newString:
    newString = newString.replace('%\n%\n','%\n')
  while ' %\n\n' in newString:
    newString = newString.replace(' %\n\n','\n\n')
  while '\n%\n\n' in newString:
    newString = newString.replace('\n%\n\n','\n\n')
  while '\n\n\n' in newString:
    newString = newString.replace('\n\n\n','\n\n')

  newString = newString.replace('\\textit{Default = ','\\default{')
  newString = re.sub(r'\$<(.*?)>\$', r'\xmlNode{\1}', newString)
  newString = re.sub(r'\textbf{\textit{(.*?)\.?}}\.', r' \xmlDesc{\1}', newString)
  newString = newString.replace('\\xmlDesc{, ', ', \\xmlDesc{')

  newFile = f + '2'
  fout = open(newFile,'w')
  fout.write(newString)
  fout.close()


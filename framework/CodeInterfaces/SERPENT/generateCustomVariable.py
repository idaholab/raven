
file = './aux-input-files/isoFile'
isotopes = []

with open(file, 'r') as isofile:
  for line in isofile:
    isotopes.append(line.strip())

customVariableFile = './aux-input-files/customVariable.xml'
with open(customVariableFile, 'w') as cvf:
  for iso in isotopes:
    cvf.write('<variable name="f' + iso + '"/> \n')

featureSpace = './aux-input-files/featureIsotopes.xml'
with open(featureSpace, 'w') as fs:
  fs.write('<Group name="featureSpace">')
  for iso in isotopes:
    fs.write('f'+iso+',')
  fs.write(' deptime </Group>')

targetSpace = './aux-input-files/targetIsotopes.xml'
with open(targetSpace, 'w') as ts:
  ts.write('<Group name="targetSpace">')
  for iso in isotopes:
    if iso == isotopes[-1]:
      ts.write('d'+iso)
    else:
      ts.write('d'+iso+',')
  ts.write('</Group>')

targetSpaceKeff = './aux-input-files/targetIsotopesKeff.xml'
with open(targetSpaceKeff, 'w') as tsk:
  tsk.write('<Group name="targetSpace">')
  for iso in isotopes:
    tsk.write('d'+iso+',')
  tsk.write('eocKeff, bocKeff </Group>')

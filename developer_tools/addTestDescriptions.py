import os
from datetime import datetime

testDict = {}

testFile = open('tests')

prefix = 'framework/ROM/MSR.'

authors = {
'aaron.epiney': 'epinas',
'andrea.alfonsi': 'alfoa',
'bandini.alessandro': 'banda',
'bounces': 'moosebuild',
'congjian.wang': 'wangc',
'cristian.rabiti': 'crisr',
'diego.mandelli': 'mandd',
'carlo.parisi': '',
'charles.jouglard': '',
'jun.chen': 'chenj',
'jongsuk.kim': 'kimj',
'joshua.cogliati': 'cogljj',
'jjc': 'cogljj',
'kingfive': 'rinai',
'ivan.rinaldi': 'rinai',
'maljdp': 'maljdan',
'maljovec': 'maljdan',
'michael.pietrykowski': '',
'robert.kinoshita': 'bobk',
'scott.schoen': '',
'sujong.yoon': '',
'ramazan.sen': 'senrs',
'sonat.sen': 'senrs',
'paul.talbot': 'talbpaul',
'talbotpne': 'talbpaul',
'taoyiliang': 'talbpaul',
'andrew.slaughter': 'slauae',
'benjamin.spencer': 'spenbw',
'brian.alger': 'algebk',
'cody.permann': 'permcj',
'codypermann': 'permcj',
'david.andrs': 'andrsd',
'haihua.zhao': 'zhaoh',
'jason.hales': 'halejd',
'jason.miller': 'milljm',
'joseph.nielsen': 'nieljw'
# cohn.72
# derek.gaston:
# jw.peterson
}

while True:
	line = testFile.readline()
	if not line:
		break
	line = line.strip()
	if line.startswith('[./'):
		token = line.replace('[./','').replace(']','')
		nextLine = testFile.readline().strip()
		while not nextLine.startswith('input'):
			nextLine = testFile.readline()
			if not nextLine:
				break
			nextLine = nextLine.strip()
		if not nextLine:
			break
		inputFile = nextLine.split(' = ')[1].replace('\'', '')
		testDict[inputFile] = prefix + token

testFile.close()

for key,value in testDict.items():
	print(key,value)
	tokens = key.rsplit('.',1)[0].split('_')
	if 'smooth' in tokens:
		smooth = True
		kernel = tokens[2]
	else:
		smooth = False
		kernel = tokens[1]
	text = os.popen('git log %s' % key).read()

	revisions = []
	lines = text.split('\n')
	i = 0
	while i < len(lines):
		line = lines[i]
		if line.startswith('Author:'):
			author = line.rsplit('<',1)[1].rsplit('@',1)[0].lower()
			if author in authors:
				author = authors[author]
			i += 1
			line = lines[i]
			date = line.split(':',1)[1][:-6].strip()
			date = datetime.strptime(date, '%a %b %d %H:%M:%S %Y').strftime('%Y-%m-%d')
			i += 2
			line = lines[i]
			description = ''
			while not line.startswith('commit'):
				description += line
				i += 1
				if i >= len(lines):
					break
				line = lines[i]
			revisions.append((author,date,description))
		i += 1

	author,date,description = revisions.pop()
	revisions.reverse()
	inputFile = open(key,'r')
	lines = [line for line in inputFile]
	inputFile.close()
	for i,line in enumerate(lines):
		if 'Simulation' in line:
			break

	newLines = []
	newLines.append('  <TestInfo>\n')
	newLines.append('    <name>%s</name>\n' % value)
	newLines.append('    <author>%s</author>\n' % author)
	newLines.append('    <created>%s</created>\n' % date)
	newLines.append('    <classesTested>SupervisedLearning.MSR</classesTested>\n')
	newLines.append('    <description>\n')
	newLines.append('       An example of using the Morse-Smale regression reduced order model with\n')
	if kernel == 'SVM':
		newLines.append('       a support vector machine.\n')
	else:
		newLines.append('       a %s kernel function for the kernel density estimator.\n' % kernel)
	if smooth:
		newLines.append('       This is a smoothed version of MSR where local models are blended\n')
		newLines.append('       together.\n')
	newLines.append('\n')
	newLines.append('       Note, all of the tests in MSR operate on a 2D input domain with\n')
	newLines.append('       the goal of fitting a single Gaussian bump. The input dimensions are\n')
	newLines.append('       of largely different scales and one dimension is off-centered from the\n')
	newLines.append('       origin to ensure that normalization is being handled correctly.\n')
	newLines.append('    </description>\n')
	newLines.append('    <revisions>\n')
	for author,date,description in revisions:
		newLines.append('      <revision author="%s" date="%s">%s</revision>\n' % (author,date,description))
	newLines.append('      <revision author="maljdan" date="2017-01-19">Adding this test description.</revision>\n')
	newLines.append('    </revisions>\n')
	newLines.append('  </TestInfo>\n')

	lines = lines[:(i+1)] + newLines + lines[(i+1):]

	inputFile = open(key,'w')
	inputFile.write(''.join(lines))
	inputFile.close()
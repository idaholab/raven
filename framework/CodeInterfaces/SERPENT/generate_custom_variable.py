
file = './aux-input-files/iso_file'
isotopes = []

with open(file, 'r') as isofile:
    for line in isofile:
        isotopes.append(line.strip())

custom_variable_file = './aux-input-files/custom_variable.xml'
with open(custom_variable_file, 'w') as cvf:
    for iso in isotopes:
        cvf.write('<variable name="f' + iso + '"/> \n')

feature_space = './aux-input-files/feature_isotopes.xml'
with open(feature_space, 'w') as fs:
    fs.write('<Group name="feature_space">')
    for iso in isotopes:
        fs.write('f'+iso+',')
    fs.write(' deptime </Group>')

target_space = './aux-input-files/target_isotopes.xml'
with open(target_space, 'w') as ts:
    ts.write('<Group name="target_space">')
    for iso in isotopes:
        if iso == isotopes[-1]:
            ts.write('d'+iso)
        else:
            ts.write('d'+iso+',')
    ts.write('</Group>')

target_space_keff = './aux-input-files/target_isotopes_keff.xml'
with open(target_space_keff, 'w') as tsk:
    tsk.write('<Group name="target_space">')
    for iso in isotopes:
        tsk.write('d'+iso+',')
    tsk.write('eoc_keff, boc_keff </Group>')

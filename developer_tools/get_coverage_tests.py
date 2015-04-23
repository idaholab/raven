skip_these = ['test_rom_trainer.xml','../../framework/TestDistributions.py']
do_tests = []
testsFile = file('tests','r')
for line in testsFile:
  if line.strip().startswith('input'):
    newtest = line.split('=')[1].strip().strip("'")
    if newtest not in skip_these: do_tests.append(newtest)

print ' '.join(do_tests)

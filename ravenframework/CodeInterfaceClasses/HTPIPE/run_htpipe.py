
if __name__=='__main__':
  import sys
  import subprocess
  args = sys.argv
  executable = args[args.index('-e')+1] if '-e' in args else None
  inputFileName = args[args.index('-i')+1] if '-i' in args else None
  cmd = [executable,]
  p = subprocess.run(cmd, input=f"2\n{inputFileName}\nn\n".encode(), stdout=subprocess.PIPE)
  print(p.stdout.decode())

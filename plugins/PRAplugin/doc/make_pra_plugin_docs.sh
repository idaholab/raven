
VERB=0
for i in "$@"
do
  if [[ $i == "--verbose" ]]
  then
    VERB=1
    echo Entering verbose mode...
  fi
done

if git describe
then
  git describe | sed 's/_/\\_/g' > new_version.tex
  echo "\\\\" >> new_version.tex
  git log -1 --format="%H %an\\\\%aD" . >> new_version.tex
  if diff new_version.tex version.tex
  then
    echo No change in version.tex
  else
    mv new_version.tex version.tex
  fi
fi

echo Building
if [[ 1 -eq $VERB ]]
then
  make; MADE=$?
else
  make > /dev/null; MADE=$?
fi

if [[ 0 -eq $MADE ]]
then
  echo ...Successfully made docs
else
  echo ...Failed to make docs
  exit -1
fi 

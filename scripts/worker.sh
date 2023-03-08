search=$(find configs -name "*.sh")

for line in $search
do
  if [[ "$line" != *"ignore"* ]]
  then
    echo "===Running script $line"
    # source $line
    status=$?
    if [[ $status == 123 ]]
    then
      echo "    ***Script $line skipped"
    fi
  fi
done
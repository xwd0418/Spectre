search=$(find configs -name "*.sh" -o -name "*.yml")

for line in $search
do
  if [[ "$line" != *"ignore"* ]]
  then
    echo "=== Running script $line"
    if [[ "$line" == *".sh" ]]
    then
      echo "Ima source dis shit $line"
      # source $line
    else
      echo "Ima python dis shit $line"
      # python train_concise.py --config $line
    fi
    status=$?
    if [[ $status == 123 ]]
    then
      echo "    ***Script $line skipped"
    fi
  fi
done
count=0
for file in *.mat; do
    newname=$(printf "%d" $count)
    mv "$file" "${newname}.mat"
    ((count++))
done
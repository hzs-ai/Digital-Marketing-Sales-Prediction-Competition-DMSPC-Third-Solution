#!/bin/bash
for file in `ls *.py`	
do
	echo 'Now run file' $file
	python3 $file
done

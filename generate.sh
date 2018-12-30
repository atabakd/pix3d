#!/bin/bash

# Remember to run another bash script to remove the '1' added by blender on the generation
# e.g. 00002.png became 000021.png

for i in `seq 0 10069`; #10069 is the number of items in the dataset
do
    echo '******************************************************************'
    echo '                   Generating ' $i' of 10069                      '
    echo '******************************************************************'
    ../../../blender-2.79b-linux-glibc219-x86_64/blender --verbose 0 --background --python demo.py -- --anno_idx $i
done

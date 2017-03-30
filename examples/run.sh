num=$1
python ./genlandmarks.py data/$num $num
./fit-model  -s 2048 -i data/$num -l data/$num.pts -o out/$num
cd out
meshlab $num.obj
# ./fit-model  -i data/u1.png -l data/u1.lm -o u1

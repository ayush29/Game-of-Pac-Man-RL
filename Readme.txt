Files & Folders
1. pacman: Folder contains my implementation of OpenAI gym compatible pacman environment
2. learning.py: Implementation of SARASA and QLEarning Algorithms
3. test.py: Python script to train, play and evaluate pacman agents.
4. Output: Output file and plot
5. Game_of_Pacman.pdf : Explanation of Assumptions and Evaluation Results


Required packages
1. python3
2. gym
3. Numpy
4. Install my implementation of pacman environment using "pip install -e pacman"

How to run?
Run "python test.py -h" to know about command line arguments required then run with suitable values. 

To Reproduce the reported results run :
"python test.py -eps1 5000 -eps2 100 -gs 5 -f 10 -lr 0.4 -gm 0.99 -epn 0.9 -ms 2500 > pacman_output.txt "



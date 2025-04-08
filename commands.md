# Pacman commands
## Pacman controlled manually
python pacman.py

## List of all options arguments to pacman.py
-l : sets layout type (`testClassic`, `smallClassic`, `mediumClassic`,
`trappedClassic`, `minimaxClassic`)  
-a : sets arguments for agents  

-g : type of ghost agent (`DirectionalGhost` are smarter than default one)  
-k : sets number of ghosts  
-a depth=3: sets depth in graph traversal algorithm (only for minimax and variants)  

-n : sets number of trials to play  
--frameTime 0 : sets frametime to 0 to accelerate game  
-q : turn off graphical layout (to accelerate game sessions)  

For more complete details on commands, type:
python pacman.py -h

## Pacman with reflex agent
python pacman.py -p ReflexAgent  
python pacman.py -p ReflexAgent -g DirectionalGhost  
python pacman.py -p ReflexAgent -l testClassic  
python pacman.py -p ReflexAgent -l testClassic  
python pacman.py --frameTime 0 -p ReflexAgent -k 1  
python pacman.py --frameTime 0 -p ReflexAgent -k 2  
python pacman.py -p ReflexAgent -l openClassic -n 10 -q  


## Pacman with minimax agent (to be implemented by students)
python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4  
python pacman.py -p MinimaxAgent -l trappedClassic -a depth=3  

## Pacman with alphabeta agent (to be implemented by students)
python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic  
python pacman.py -p AlphaBetaAgent -l trappedClassic -a depth=3 -q -n 10  

## Pacman with expectiminimax agent (to be implemented by students)
python pacman.py -p ExpectimaxAgent -l trappedClassic -a depth=3 -q -n 10  
python pacman.py -l smallClassic -p ExpectimaxAgent -a evalFn=better -q -n 10  
python pacman.py -l contestClassic -p ContestAgent -g DirectionalGhost -q -n 10  

## Pacman with MachineLearning agent (to be implemented by students)
python pacman.py -l mediumClassic -p MachineLearningAgent -q -n 500  

## Tests to submit for grading
python pacman.py -l mediumClassic -p ExpectimaxAgent -q -n 500 -a depth=3,evalFn=better  
python pacman.py -l mediumClassic -p ExpectimaxAgent -g DirectionalGhost -q -n 75 -a depth=3,evalFn=better  
python pacman.py -l mediumClassic -p MachineLearningAgent -q -n 500  
python pacman.py -l mediumClassic -p MachineLearningAgent -g DirectionalGhost -q -n 75  

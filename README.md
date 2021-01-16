# navigation
This project is a solution for navigation problem, where an unity environment is controled by artificial brain.

This environment have 37 box states and 4 discrete actions.

The environment is considered solved if the mean of last 100 episodes is more than or equal 13 yellow bananas collecteds.

A requirements file is disponibilized for install necessary libs, just use pip install -r requirements.txt
download unity agent and paste in navigation directory. https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip

Run Navigation.ipynb on jupyter notebook and follow instructs.

For solve this a reinforcement learning networks was implemented, the hiperameters was seted by comparing score graph.
After train is possible to see an agent collecting yellow bananas.





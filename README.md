# Installation Guide
## Installing Prerequisites  
- Install [Anaconda](https://docs.anaconda.com/anaconda/install/index.html)
- Open terminal and make sure git is installed or else install [github](https://anaconda.org/anaconda/git)
- Create a working directory and name it the way you want ( exemple: /working_directory). Access it through the terminal.
- Git clone my project’s repository in working directory (*write your command after the "$" sign*):
  > ~/working_directory$ git clone https://github.com/oscarjenot/pdm_oscar_jenot.git
- Go to the pulled git repo, copy and activate my conda environment (named flatnessRL) by running (this should take some time) :
  > ~/working_directory$ cd pdm_oscar_jenot  
  > ~/working_directory/pdm_oscar_jenot$ conda env create -f flatnessRL.yaml  
  > ~/working_directory/pdm_oscar_jenot$ conda activate flatnessRL  
- Install OpenAI [Gym](https://gym.openai.com/docs/) by running :
  > ~/working_directory/pdm_oscar_jenot$ pip install gym  
  > ~/working_directory/pdm_oscar_jenot$ git clone https://github.com/openai/gym  
  > ~/working_directory/pdm_oscar_jenot$ cd gym  
  > ~/working_directory/pdm_oscar_jenot$ pip install -e .  
  > ~/working_directory/pdm_oscar_jenot$ cd ..  

## Installing and registering the projects environement
  
The directory /pdm_oscar_jenot/ should have the files or directories : agent, pdm, gym, Master’s Project Report.pdf, flatnessRL.yaml
- Install the project’s environment in Gym’s toolkit:
  - Cut the directory /working_directory/pdm_oscar_jenot/PDM  
  - Paste it in /working_directory/pdm_oscar_jenot/gym/gym/envs  
- Register project’s environments in Gym’s toolkit  
  - Add paths for newly created environments in the “SOURCES.txt” file
    - Open /working_directory/pdm_oscar_jenot/gym/gym.egg-info/SOURCES.txt
    - Add the lines (in alphabetic order):
      > gym/envs/pdm/__ init__.py  
      > gym/envs/pdm/crane_env.py  
      > gym/envs/pdm/crane_env_1.py  
      > gym/envs/pdm/crane_env_2.py  
  - Register environment in the “__ init__.py” file
    - Open /working_directory/pdm_oscar_jenot/gym/gym/envs/__ init__.py
    - Add the lines (respect the indent : tab = 4 spaces):
      > register(  
      > &nbsp;&nbsp;&nbsp;&nbsp; id='crane-v0',  
      > &nbsp;&nbsp;&nbsp;&nbsp; entry_point='gym.envs.pdm:CraneEnv',  
      > &nbsp;&nbsp;&nbsp;&nbsp; max_episode_steps=2000,  
      > )  
      > <br />
      > register(  
      > &nbsp;&nbsp;&nbsp;&nbsp; id='crane-v1',  
      > &nbsp;&nbsp;&nbsp;&nbsp; entry_point='gym.envs.pdm:CraneEnv1',  
      > &nbsp;&nbsp;&nbsp;&nbsp; max_episode_steps=2000,  
      > )  
      > <br />
      > register(  
      > &nbsp;&nbsp;&nbsp;&nbsp; id='crane-v2',  
      > &nbsp;&nbsp;&nbsp;&nbsp; entry_point='gym.envs.pdm:CraneEnv2',  
      > &nbsp;&nbsp;&nbsp;&nbsp; max_episode_steps=2000,  
      > )  
  - Add the imports of the new environment in the second “__init__.py” file in the pdm directory (should already be done):
    - Open /working_directory/pdm_oscar_jenot/gym/gym/envs/PDM/__ init__.py
    - Add the lines (respect the indent : tab = 4 spaces):
      > from gym.envs.pdm.crane_env import CraneEnv
      > from gym.envs.pdm.crane_env_1 import CraneEnv1
      > from gym.envs.pdm.crane_env_2 import CraneEnv2
      
## Creating a new environement and runing demos

- If you wish to create a new RL environment, just add it the the /working_directory/pdm_oscar_jenot/gym/gym/envs/PDM directory and follow the previous steps for its correct registration
- To run a simulation of the environments
  - From terminal go to the directories : /working_directory/pdm_oscar_jenot/agent/James001 or James002
  - Run python test.py
    > ~/working_directory/pdm_oscar_jenot/agent/James001$ python3 test.py
- To open the training agents or the presentation training demo (James001.ipynb, James002.ipynb, James003.ipynb, Presentation001.ipynb, etc files
  - From terminal go to /working_directory/pdm_oscar_jenot/agent and run:
    > ~/working_directory/pdm_oscar_jenot/agent$ jupyter notebook
  -Browse and open the wanted files
  
## Questions and recommandations

For any questions or tips about the project, installation, etc, your may contact me at: 
oscar.jenot@alumni.epfl.ch
  
    
    
    

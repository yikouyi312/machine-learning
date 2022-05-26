# machine-learning
**Setting**

- **Mac Guide**
  - Step 1: Installing Miniconda
     - Open terminal
     - IF DEFAULT SHELL IS NOT BASH, change shell to bash by running: `**_chsh -s /bin/bash_**`, Otherwise, skip this step
     - Download the installer by either:
       - Downloading the installer directly from `**_https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-MacOSX-x86_64.sh_**` 
            and moving the downloaded file into your root directory
       - Running: `**_curl -sL https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-MacOSX-x86_64.sh > Miniconda3-py39_4.10.3-MacOSX-x86_64.sh_**`
       - Running: `**_wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-MacOSX-x86_64.sh_**`
    - Run: `**_bash Miniconda3-py39_4.10.3-MacOSX-x86_64.sh_**`
    - Follow the on screen instructions and accept everything as the default versions
    - Close and reopen terminal
    - Type the command: conda list to check that it installed correctly!
  - Step 2: Setting up a conda environment
    - Run: `**_conda create -n myenvname python=3.9.6 numpy=1.21.2 scipy=1.7.1_**` Where myenvname is replaced by whatever you want to name your environment. 
    - If the above doesn’t work, run `**_conda config --append channels conda-forge_**` and then rerun the first step
  - Step 3: Using your conda environment
    - Run: `**_conda activate myenvname_**` to switch into your environment, which will be indicated by the start of your terminal line having (myenvname) at the front of it: 
      Note: This command must be run every time you start a new terminal session!
    - If you ever want to switch to your base environment or a different environment, run: `**_conda deactivate_**`
	


- **Windows Guide**
  - Step 1: Installing Miniconda
    - Download the miniconda installer from `**_https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Windows-x86_64.exe_**`
    - Double-click the downloaded .exe file to begin the installation
    - Follow the on screen instructions and accept everything as the default versions
    - From the start menu, open Anaconda Prompt (this will basically let you do linux-y commands on your windows machine)
    - Type the command: `**_conda list_**` to check that it installed correctly!
  - Step 2: Setting up a conda environment
    - Run: `**_conda create -n myenvname python=3.9.6 numpy=1.21.2 scipy=1.7.1_**` Where myenvname is replaced by whatever you want to name your environment. 
    - If the above doesn’t work, run `**_conda config --append channels conda-forge_**` and then rerun the first step
  - Step 3: Using your conda environment
    - Run: `**_conda activate myenvname_**` to switch into your environment, which will be indicated by the start of your anaconda prompt line having (myenvname) at the front of it:
      Note: This command must be run every time you start a new Anaconda Prompt session!
    - If you ever want to switch to your base environment or a different environment, run: `**_conda deactivate_**`
	


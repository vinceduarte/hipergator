# hipergator
CIS6930

## Dependencies

- Anaconda https://docs.anaconda.com/anaconda/install/linux/
    
    Install using the GUI application
    ``` bash ~/Downloads/Anaconda3-2021.05-Linux-x86_64.sh ```
    
    Initialize:
    ``` source <path to conda>/bin/activate ```
    ``` conda init ```

- Virtual enviornment https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwi7v87dl7bzAhVDTTABHdn4CeUQFnoECAsQAQ&url=https%3A%2F%2Fdocs.python.org%2F3%2Flibrary%2Fvenv.html&usg=AOvVaw1SQ6VGTcJCX7W6wOs1SpnV

    1. Using Anaconda https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/
        Create the virtual enviornment if it does not exist:
        ``` conda create -n yourenvname python=x.x anaconda ```

        Activate the virtual enviornment in terminal:
        ``` conda activate venv ```
        If done correctly, (venv) will show in the terminal.

        Deactivate the virtual enviornment:
        ``` conda deactivate venv ```

        Remove the virtual enviornment if you need a new one:
        ``` conda remove -n venv -all ```
    
    2. Using python out of the box
        Create the virtual enviornment if it does not exist:
        ``` python3 -m venv ./venv ```

        Activate the virtual enviornment in terminal:
        ``` source /path/to/venv/bin/activate ```
        If done correctly, (venv) will show in the terminal.

        Deactivate the virtual enviornment:
        ``` deactivate .venv ```

        Remove the virtual enviornment if you need a new one:
        ``` sudo rm -rf venv ```

    Install dependencies:

    - PyTorch-1.8.1
    CPU ONLY
    ``` conda install pytorch==1.8.1 torchvision==0.9.0 torchaudio==0.8.0 cpuonly -c pytorch ```

- Matplotlib https://matplotlib.org/stable/users/installing.html

- scikit-learn https://scikit-learn.org/stable/install.html

TODO:
Need to Add (Code):
 - Return prediction for a given input
 - Check for overfitting !!!!!!
 - ~~Support for multilabel and final
 - Integration for HiperGator
 - ~~Integration for game
 - Accuracy Part II (Since this is really a multi-label classification problem that is being treated as a regression problem, please report the accuracy of the multi-label regressor as if it were a classifier, by rounding the output of each regression output to either 0 or 1 and comparing with the testing data)
Must write up:
 - Classifier and Regression Analysis (see bolded section)
 - Written Report
 - Video!

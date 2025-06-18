# GitHub tutorial for the Rhone group https://materials-intelligence.com/

## Motivation
Everyone in the Rhone group codes. Most of us started with Professor Rhone's onboarding exercise, which is the content of this repository. One day, let's say we get a very nice score on magnetic moment using RandomForestRegressor. We may want to keep the corresponding set of descriptors, hyperparameters, and even better, the current stage of our notebook/colab. But at the same time, also want to modify the program to achieve a better score. 

One option is to make a copy of the code. We will have `ML_2D_exercises.ipynb` and `ML_2D_exercises.ipynb(1).ipynb` in our folder. If another change is made, we will have `ML_2D_exercises.ipynb(2).ipynb`, `ML_2D_exercises.ipynb(3).ipynb`, and so on. As you can see, this approach can become overwhelming due to the ever-increasing number of files in our folder. Also, a week later, one may have trouble remembering which file, `ML_2D_exercises.ipynb(2).ipynb` or `ML_2D_exercises.ipynb(3).ipynb` contains the better result.

On the other hand, if we only keep one copy and make a mistake, we might need to press `Ctrl + C` or `Command + C` many times to get back the stage that was working, and there is no guarantee we will.

The solution is GitHub https://github.com/, and more fundamentally, **git** https://git-scm.com/. *git* allows you to save the current stage of a folder. In which you can put `.iypnb`, `.py`, `.txt`, and even `.pptx`. git will track everything in your folder, and let you go back to any previous version. The means we only need one `ML_2D_exercises.ipynb` in our folder, yet we can track its progress. Even better, this version control is done on you local machine. That is, as long as your computer has battery, you can access anything you saved in your git repository.

*github* on the other hand is a website for people to share programs. The previous versions of a program can be tracked, just like in git. So, rather than sending your `.ipynb` file through Webex or email, you can put your code on github, and people can get an continuously updated version of your program.

## Getting Started
Download git at https://git-scm.com/downloads. If you use Window, I strongly recommend installing Window Subsystem for Linux (WSL) https://learn.microsoft.com/en-us/windows/wsl/install. You get all the features of Linux plus a smooth experience using git and github.

continues...
  

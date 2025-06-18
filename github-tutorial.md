# GitHub tutorial for the Rhone group https://materials-intelligence.com/

*A step-by-step guide to collaborative GitHub workflows, tailored for researchers new to version control and open-source collaboration.*

---

## Table of Contents

1. [Motivation](#motivation)
2. [Getting Started](#getting-started)
3. [Forking vs. Cloning](#forking-vs-cloning)
4. [Branching Strategy](#branching-strategy)
5. [Keeping Your Branch Up-to-Date](#keeping-your-branch-up-to-date)
6. [Pushing & Pull Requests](#pushing--pull-requests)
7. [Code Review & Merging](#code-review--merging)

## Motivation
Everyone in the Rhone group codes. Most of us started with Professor Rhone's onboarding exercise, which is the content of this repository. One day, let's say we get a very nice score on magnetic moment using RandomForestRegressor. We may want to keep the corresponding set of descriptors, hyperparameters, and even better, the current stage of our notebook/colab. But at the same time, also want to modify the program to achieve a better score. 

One option is to make a copy of the code. We will have `ML_2D_exercises.ipynb` and `ML_2D_exercises.ipynb(1).ipynb` in our folder. If another change is made, we will have `ML_2D_exercises.ipynb(2).ipynb`, `ML_2D_exercises.ipynb(3).ipynb`, and so on. As you can see, this approach can become overwhelming due to the ever-increasing number of files in our folder. Also, a week later, one may have trouble remembering which file, `ML_2D_exercises.ipynb(2).ipynb` or `ML_2D_exercises.ipynb(3).ipynb` contains the better result.

On the other hand, if we only keep one copy and make a mistake, we might need to press `Ctrl + C` or `Command + C` many times to get back the stage that was working, and there is no guarantee we will.

The solution is GitHub https://github.com/, and more fundamentally, **git** https://git-scm.com/. *git* allows you to save the current stage of a folder. In which you can put `.iypnb`, `.py`, `.txt`, and even `.pptx`. git will track everything in your folder, and let you go back to any previous version. The means we only need one `ML_2D_exercises.ipynb` in our folder, yet we can track its progress. Even better, this version control is done on you local machine. That is, as long as your computer has battery, you can access anything you saved in your git repository.

*github* on the other hand is a website for people to share programs. The previous versions of a program can be tracked, just like in git. So, rather than sending your `.ipynb` file through Webex or email, you can put your code on github, and people can get an continuously updated version of your program.

## Getting Started
Download git at https://git-scm.com/downloads. If you use Window, I strongly recommend installing Window Subsystem for Linux (WSL) https://learn.microsoft.com/en-us/windows/wsl/install. You get all the features of Linux plus a smooth experience using git and github.

Open up a terminal and do `git clone https://github.com/TingwenZhang/magnetic-materials-2d.git`, to copy this repository to your local machine.

Then, move to the project directory:
```
cd magnetic-materials-2d
```

You can skip the the `conda` steps in https://github.com/TingwenZhang/magnetic-materials-2d/tree/main#how-to-install-magnetic-materials-2d-locally, if you are not familar with conda virtual environment. We will focus on using `git` first.

Type `code .` in your terminal to open the repository in VSCode, or your favorite text editor.

continues...

## Forking vs. Cloning

- **Fork**: Creates your own copy of a repository on GitHub (use when you don’t have write access).
- **Clone**: Copies a repository (fork or original) to your local machine.

**To fork and clone**:

1. On GitHub, click **Fork** (top-right).
2. Clone your fork locally:
   ```bash
   git clone git@github.com:<your-username>/magnetic-materials-2d.git
   cd magnetic-materials-2d
   ```
3. Add the upstream remote (original repo):
   ```bash
   git remote add upstream https://github.com/TingwenZhang/magnetic-materials-2d.git
   ```

## Branching Strategy

Use branches to isolate work:

1. **Create a branch** per feature/fix:
   ```bash
   git checkout -b feat/new-descriptor
   ```
2. **Commit frequently** with clear messages:
   ```bash
   git add data_processing.py
   git commit -m "Add electronegativity descriptor function"
   ```
3. **Keep branches small and focused.**

## Branching Strategy

Use branches to isolate work:

1. **Create a branch** per feature/fix:
   ```bash
   git checkout -b feat/new-descriptor
   ```
2. **Commit frequently** with clear messages:
   ```bash
   git add data_processing.py
   git commit -m "Add electronegativity descriptor function"
   ```
3. **Keep branches small and focused.**

## Keeping Your Branch Up-to-Date

Sync regularly with `main`:

```bash
git checkout main
git fetch upstream
git pull upstream main

git checkout feat/new-descriptor
# Option A: Rebase
git rebase main
# Option B: Merge
git merge main
```

- **Rebase**: cleaner, linear history.
- **Merge**: preserves merge commits.

## Pushing & Pull Requests

1. **Push your branch** to your fork:
   ```bash
   git push origin feat/new-descriptor
   ```
2. On GitHub, click **Compare & pull request**.
3. Complete the PR template:
   - **What** does this change do?
   - **Why** is it needed?
   - Related issues or blockers?
4. Assign reviewers (e.g., `@DrRhone`).
5. Address feedback by pushing additional commits to the same branch.

## Code Review & Merging

- **Reviewers**:

  - Check for clarity, functionality, style, and tests.
  - Use GitHub’s inline comments.

- **Authors**:

  - Update code, respond to comments, and push fixes.

**After approval**:

1. Use **Squash & merge** to consolidate commits (preferred).
2. Delete the branch.
3. Pull the latest `main` locally.

---
  ## Best Practices

- Write clear, imperative commit messages.
- Keep each commit focused on one change.
- Test locally before pushing.
- Document major changes in `README.md`.
- Use meaningful branch names: `feat/…`, `fix/…`, `docs/…`.
- Review PRs promptly and constructively.

---

## Further Resources

- **Pro Git Book**: [https://git-scm.com/book/en/v2](https://git-scm.com/book/en/v2)
- **GitHub Learning Lab**: [https://lab.github.com/](https://lab.github.com/)
- **GitHub Actions Docs**: [https://docs.github.com/en/actions](https://docs.github.com/en/actions)

---

*Happy collaborating!*

- **Scikit-Package**: [https://scikit-package.github.io/scikit-package/index.html](https://scikit-package.github.io/scikit-package/index.html) — Tools and practices for building reusable scientific Python packages (by Columbia University).

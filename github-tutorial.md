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

One option is to make a copy of the code. We will have `ML_2D_exercises.ipynb` and `ML_2D_exercises(1).ipynb` in our folder. If another change is made, we will have `ML_2D_exercises(2).ipynb`, `ML_2D_exercises(3).ipynb`, and so on. As you can see, this approach will become overwhelming due to the ever-increasing number of files in our folder. Also, a week later, one may have trouble remembering which file, `ML_2D_exercises(2).ipynb` or `ML_2D_exercises(3).ipynb` contains the better result.

On the other hand, if we only keep one copy and make a mistake, we might need to press `Ctrl + C` or `Command + C` many times to get back the stage that was working, and there is no guarantee we will.

The solution is GitHub https://github.com/, and more fundamentally, **git** https://git-scm.com/. **git** allows you to save the current stage of a folder, in which you can put `.ipynb`, `.py`, `.txt`, and even `.pptx`. git will track everything in your folder, and let you go back to any previous version. The means you only need one `ML_2D_exercises.ipynb` in your folder, yet its progress is tracked. Even better, this version control is done on you local machine. That is, as long as your computer has battery, you can access anything you saved in your git repository.

*github* on the other hand is a website for people to share programs. The previous versions of a program can be tracked, just like in git. So, rather than sending your `.ipynb` file through Webex or email, you can put your code on github, and people can get a continuously updated version of your program.

## Getting Started
Download git at https://git-scm.com/downloads. If you use Window, I strongly recommend installing Window Subsystem for Linux (WSL) https://learn.microsoft.com/en-us/windows/wsl/install. You get all the features of Linux plus a smooth experience using git and github.

# Getting Started with the Magnetic Materials 2D Repository

## Clone the Repository

Open a terminal and run:

```
git clone https://github.com/TingwenZhang/magnetic-materials-2d.git
cd magnetic-materials-2d
```

This copies the repository to your local machine and moves you into the project directory.

## Skip Conda for Now

You can skip the `conda` setup steps mentioned in the [install guide](https://github.com/TingwenZhang/magnetic-materials-2d/tree/main#how-to-install-magnetic-materials-2d-locally) if you're not familiar with virtual environments. We'll focus on using `git`.

## Open in VSCode

To open the repository in VSCode, type:

```
code .
```

Or open it in your preferred text editor.

## Edit the Onboarding Notebook

Navigate to:

```
src/notebooks/ML_2D_exercises.ipynb
```

This notebook contains the onboarding exercise.

Make a simple edit—for example, add a line like:

```python
print("Hello, world!")
```

Save the file.

## Commit Your Changes

In your terminal (make sure you're still in the `magnetic-materials-2d` folder), run:

```
git add .
git commit -m "modified the onboarding exercise"
```

This stages and commits your change, storing a new version in the git history.

## View Git History

To see previous commits:

```
git log
```

You'll see a list of commits with their hashes. To view a previous version, copy the commit hash (e.g., `abc1234`) and run:

```
git checkout abc1234
```

This puts your repository in a "detached HEAD" state, showing the contents as they were at that commit. You can now open the notebook and explore the older version **(You should see `print("Hello, world!")` disappears from your notebook)**.

## Return to the Main Branch

To go back to the most recent version on the main branch:

```
git checkout main
```
**(You should see `print("Hello, world!")` re-appears in your notebook)**

## Compare Changes

To compare your latest commit with the one before it:

```
git diff HEAD^ HEAD
```

This shows exactly what was changed.

## Continue Practicing

- Try editing the notebook again and repeating the `git add`, `git commit`, and `git log` steps.
- Use `git status` to check which files have been modified or staged.
- Explore `git diff` to see changes before committing.

You're now using `git` to track, compare, and explore versions of your onboarding exercise.


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
- **Scikit-Package**: [https://scikit-package.github.io/scikit-package/index.html](https://scikit-package.github.io/scikit-package/index.html) — Tools and practices for building reusable scientific Python packages (by Columbia University).
  
---

*Happy collaborating!*


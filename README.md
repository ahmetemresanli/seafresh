# SeaFresh

This is a shared public repository for the team. Every teammate is expected to add their own code here, following a fork-based workflow.

## How to contribute your code

1. **Fork** this repository on GitHub (button top-right of the repo page).
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/<your-username>/seafresh.git
   cd seafresh
   git remote add upstream https://github.com/juandronm/seafresh.git
   ```
3. **Create a folder with your name or project name** at the root of the repo, e.g.:
   ```
   seafresh/
     juan/
     maria/
     detection-model/
   ```
   Put all your code, scripts, and data references inside your own folder. Do not edit other people's folders.
4. **Create a branch, commit, and push to your fork**:
   ```bash
   git checkout -b add-<your-name>-code
   git add <your-folder>
   git commit -m "Add <your-name> code"
   git push origin add-<your-name>-code
   ```
5. **Open a Pull Request** from your fork's branch into `juandronm/seafresh:main`.
6. Before starting new work, sync your fork with the latest changes:
   ```bash
   git checkout main
   git pull upstream main
   git push origin main
   ```

## How to run your own code

Each contributor's folder should be runnable on its own. Inside **your** folder:

1. **Add a `requirements.txt`** (or `environment.yml`) listing the exact packages your code needs.
2. **Create and activate a virtual environment** before installing anything:
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS/Linux
   source venv/bin/activate
   ```
3. **Install your dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run your code** from inside your folder, e.g.:
   ```bash
   python main.py
   ```
5. If your code needs extra setup (API keys, models to download, config files), add a short `README.md` inside your own folder explaining those steps — don't assume everyone knows how to run it.

## Notes

- Do not commit secrets, API keys, `.env` files, virtual environment folders (`venv/`), or large model/data files. Add them to `.gitignore` instead.
- Keep your folder self-contained so others can run your code without touching the rest of the repo.
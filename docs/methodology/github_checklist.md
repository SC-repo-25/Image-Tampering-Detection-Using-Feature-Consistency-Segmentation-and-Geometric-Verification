# GitHub Final Submission Checklist

- [ ] Repository is **PUBLIC**
- [ ] Repository root contains `README.md`
- [ ] `statement.md` exists at root
- [ ] `requirements.txt` exists at root
- [ ] Source code is complete (`src/`, `main.py`, `config.py`)
- [ ] Tests exist and pass (`pytest tests/ -v`)
- [ ] Project runs from terminal (`python main.py --input ...`)
- [ ] No GUI dependency
- [ ] Dataset instructions are clear (`data/README.md`)
- [ ] No unnecessary large files committed (raw datasets excluded — see
      `.gitignore` and README §13 licensing note)
- [ ] `.gitignore` exists
- [ ] No API keys/passwords/secrets anywhere in the repo
- [ ] Screenshots/results included where appropriate
      (`outputs/visualizations/` sample images, or embed a few in README)
- [ ] Documentation is complete (`docs/`)
- [ ] Report is complete (`report/project_report.md` → converted to
      `report/project_report.pdf`)
- [ ] Repository root URL is correct: exactly
      `https://github.com/<github-username>/<repo-name>`
      — never `/tree/main/` or `/blob/main/`

## Before pushing

```bash
git init
git add .
git commit -m "Initial commit: CV image tampering detection system"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

## Recommended: add a couple of real screenshots to the README

```bash
# after running main.py on a sample image, embed e.g.:
# outputs/visualizations/<name>_09_final_heatmap.png
```

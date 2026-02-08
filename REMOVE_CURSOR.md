# Commands to Remove Cursor from GitHub

Run these commands in order:

```bash
cd /Users/yliu3y/Desktop/Passwords

# Create fresh history
git checkout --orphan clean-main
git add .
TREE=$(git write-tree)
COMMIT=$(echo "initial commit" | git commit-tree $TREE)
git reset --hard $COMMIT

# Replace main branch
git branch -D main
git branch -m main

# Force push (this replaces ALL history on GitHub)
git push --force origin main

# Verify it's clean
git log --format=fuller
git show HEAD --format=fuller
```

After running these, check GitHub:
- https://github.com/ybro2011/celebrity-lookalike-finder/commits/main
- https://github.com/ybro2011/celebrity-lookalike-finder/graphs/contributors

If you still see Cursor, wait 10 minutes and hard refresh (Cmd+Shift+R).

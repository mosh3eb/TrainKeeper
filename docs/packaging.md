# Packaging Hygiene

When distributing a ZIP from macOS, Finder can add `__MACOSX/` and `._*` metadata files.
These can cause install errors such as duplicate `pyproject.toml` sections.

Before publishing a ZIP, ensure these files are removed:

```bash
rm -rf __MACOSX
find . -name "._*" -delete
```

Then rebuild the ZIP from the clean directory.

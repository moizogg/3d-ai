# Synthetic frame fixtures

Tiny sharp grid images for Phase 2 tests and CLI smoke runs.

Regenerate:

```bash
python -c "from pathlib import Path; from PIL import Image, ImageDraw; root=Path('.'); ..."
```

Or re-run any test that builds its own temp fixtures.

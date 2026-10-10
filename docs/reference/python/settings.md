# Settings

`VPP_*` environment variables (or a `.env` file) are loaded into
`Settings`. The [configuration reference](../../configuration.md) lists
every variable with its default and explains how they interact.

```python
from vpp.settings import get_settings

settings = get_settings()
print(settings.database_url)
```

::: vpp.settings
